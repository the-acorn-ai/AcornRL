import textarena as ta 
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import torch
import os
import time
import gc
import threading
from acornrl.collectors.base import Collector
from acornrl.agents.base import Agent

class MultiGPUTextArenaCollector(Collector):
    """
    A collector that ensures balanced utilization across all available GPUs
    by explicitly controlling the assignment of work.
    """
    
    def __init__(
        self,
        env_ids: List[str],
        batch_size: int = 16,
        max_steps_per_episode: int = 100,
        **kwargs
    ):
        """
        Initialize a balanced GPU collector.
        
        Args:
            env_ids: List of TextArena environment IDs to collect from
            batch_size: Number of episodes to collect per batch
            max_steps_per_episode: Maximum number of steps per episode
            **kwargs: Additional parameters
        """
        super().__init__(batch_size=batch_size)
        self.env_ids = env_ids
        self.max_steps_per_episode = max_steps_per_episode
        
        # Set up GPU detection
        self.num_gpus = torch.cuda.device_count()
        self.gpu_available = self.num_gpus > 0
        
        if self.gpu_available:
            print(f"Found {self.num_gpus} GPU(s)")
            # List available GPUs and get memory info
            self.gpu_mem_info = []
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                print(f"  GPU {i}: {gpu_name} with {gpu_mem_total:.2f} GB memory")
                self.gpu_mem_info.append({"name": gpu_name, "mem_total": gpu_mem_total})
        else:
            print("No GPUs detected, using CPU only")
            self.gpu_mem_info = []
        
        # Set up GPU load tracking
        self.gpu_tasks = {i: 0 for i in range(self.num_gpus)} if self.gpu_available else {}
        self.gpu_lock = threading.Lock()
        
        # Calculate appropriate batch size
        self.batch_size = min(batch_size, num_gpus * 4) if self.gpu_available else batch_size
        print(f"Using batch size of {self.batch_size}")
    
    def _get_least_busy_gpu(self):
        """Get the GPU with the fewest active tasks."""
        with self.gpu_lock:
            if not self.gpu_tasks:
                return 0
            return min(self.gpu_tasks, key=self.gpu_tasks.get)
    
    def _increment_gpu_tasks(self, gpu_idx):
        """Increment the task count for a GPU."""
        with self.gpu_lock:
            self.gpu_tasks[gpu_idx] = self.gpu_tasks.get(gpu_idx, 0) + 1
    
    def _decrement_gpu_tasks(self, gpu_idx):
        """Decrement the task count for a GPU."""
        with self.gpu_lock:
            self.gpu_tasks[gpu_idx] = max(0, self.gpu_tasks.get(gpu_idx, 0) - 1)
    
    def _print_gpu_utilization(self):
        """Print current GPU utilization info."""
        if not self.gpu_available:
            return
            
        try:
            print("\nCurrent GPU utilization:")
            for i in range(self.num_gpus):
                # Get memory stats
                free_mem = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory
                used_mem = total_mem - free_mem
                percent_used = (used_mem / total_mem) * 100
                active_tasks = self.gpu_tasks.get(i, 0)
                
                print(f"  GPU {i}: {percent_used:.1f}% memory used, {active_tasks} active tasks")
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
    
    def collect(self, agent: Agent, num_episodes: int) -> List[Dict[str, Any]]:
        """
        Collect data with balanced GPU utilization.
        
        Args:
            agent: Agent to use for generating actions
            num_episodes: Number of episodes to collect
            
        Returns:
            List of collected step data
        """
        collected_data = []
        active_episodes = 0
        
        # Create model copies for each GPU to avoid moving models
        # between devices (which can cause issues with memory fragmentation)
        gpu_models = {}
        
        if self.gpu_available and hasattr(agent, 'model'):
            print("Initializing model on each GPU...")
            import copy
            
            for i in range(self.num_gpus):
                # Create a copy of the agent for each GPU
                gpu_agent = copy.deepcopy(agent)
                
                # Move model to the specific GPU and ensure it stays there
                device = f"cuda:{i}"
                
                # Use specific context for this operation
                with torch.cuda.device(i):
                    # Clear GPU memory before loading model
                    torch.cuda.empty_cache()
                    
                    # Move model to device
                    gpu_agent.model = gpu_agent.model.to(device)
                    
                    # Store in dictionary
                    gpu_models[i] = gpu_agent
                    
                    # Perform a small forward pass to ensure the model is loaded and properly initialized
                    if hasattr(gpu_agent.model, 'tokenizer'):
                        tokenizer = gpu_agent.model.tokenizer
                        small_input = tokenizer("test", return_tensors="pt").to(device)
                        try:
                            with torch.no_grad():
                                _ = gpu_agent.model(small_input.input_ids)
                            print(f"  Initialized model on GPU {i}")
                        except Exception as e:
                            print(f"  Warning: Couldn't run initialization test on GPU {i}: {e}")
            
            print("All GPU models initialized successfully")
        else:
            # Single model mode (either CPU or no model attribute)
            gpu_models = {-1: agent}  # -1 represents CPU
        
        # Track which environments are assigned to which GPUs
        env_gpu_map = {}
        
        # Set up a queue for environments
        env_queue = deque()
        
        # Track episode data for each environment
        all_episode_data = {}
        
        # Set initial batch size based on available resources
        batch_size = min(self.batch_size, num_episodes)
        
        # Print initial GPU utilization
        self._print_gpu_utilization()
        
        with tqdm(total=num_episodes, desc="Collecting episodes") as episode_pbar:
            # Initialize environments
            for env_idx in range(min(batch_size, num_episodes)):
                # Create and initialize environment
                env = ta.make(np.random.choice(self.env_ids) if isinstance(self.env_ids, list) else self.env_ids)
                env = ta.wrappers.LLMObservationWrapper(env=env)
                env.reset(num_players=2)
                
                player_id, observation = env.get_observation()
                
                # Assign unique ID to this environment
                env_id = env_idx
                
                # Assign to the least busy GPU
                if self.gpu_available:
                    gpu_idx = self._get_least_busy_gpu()
                    self._increment_gpu_tasks(gpu_idx)
                else:
                    gpu_idx = -1  # CPU
                
                env_gpu_map[env_id] = gpu_idx
                print(f"Environment {env_id} assigned to {'CPU' if gpu_idx < 0 else f'GPU {gpu_idx}'}")
                
                # Initialize episode data for this environment
                all_episode_data[env_id] = []
                
                # Add to queue for processing
                env_queue.append({
                    "env": env,
                    "env_id": env_id,
                    "done": False,
                    "current_player_id": player_id,
                    "current_observation": observation,
                    "step": 0
                })
                
                active_episodes += 1
                
            # Process environments until all episodes are complete
            with tqdm(total=num_episodes * self.max_steps_per_episode, desc="Total steps", leave=False) as step_pbar:
                env_counter = 0  # For periodically printing GPU stats
                
                while env_queue and (active_episodes < num_episodes or len(env_queue) > 0):
                    # Process environments one by one to avoid memory spikes
                    env_data = env_queue.popleft()
                    
                    env = env_data["env"]
                    env_id = env_data["env_id"]
                    player_id = env_data["current_player_id"]
                    observation = env_data["current_observation"]
                    step = env_data["step"]
                    
                    # Get the assigned GPU for this environment
                    gpu_idx = env_gpu_map[env_id]
                    
                    # Use the model assigned to this GPU
                    device_agent = gpu_models.get(gpu_idx, gpu_models.get(-1))
                    
                    # Process with the assigned model
                    try:
                        if gpu_idx >= 0:
                            # Set CUDA context for this operation
                            with torch.cuda.device(gpu_idx):
                                action, reasoning = device_agent(observation=observation)
                        else:
                            # CPU processing
                            action, reasoning = device_agent(observation=observation)
                            
                    except Exception as e:
                        print(f"Error processing on {'CPU' if gpu_idx < 0 else f'GPU {gpu_idx}'}: {e}")
                        # Fall back to a generic response if needed
                        action = "default_action"
                        reasoning = "Error occurred during processing"
                    
                    # Record step data
                    all_episode_data[env_id].append({
                        "player_id": player_id,
                        "observation": observation,
                        "reasoning": reasoning,
                        "action": action,
                        "step": step,
                    })
                    
                    # Execute step in environment
                    done, info = env.step(action=action)
                    step_pbar.update(1)
                    
                    if not done and step < self.max_steps_per_episode - 1:
                        # Get next observation
                        next_player_id, next_observation = env.get_observation()
                        
                        # Put back in queue with updated state
                        env_queue.append({
                            "env": env,
                            "env_id": env_id,
                            "done": False,
                            "current_player_id": next_player_id,
                            "current_observation": next_observation,
                            "step": step + 1
                        })
                    else:
                        # Episode completed
                        # Decrease task count for this GPU
                        if gpu_idx >= 0:
                            self._decrement_gpu_tasks(gpu_idx)
                        
                        # Get rewards
                        rewards = env.close()
                        
                        # Add rewards to all steps in this episode
                        for step_data in all_episode_data[env_id]:
                            step_data["final_reward"] = rewards[step_data["player_id"]]
                            step_data["full_length"] = len(all_episode_data[env_id])
                        
                        # Add to collected data
                        collected_data.extend(all_episode_data[env_id])
                        
                        episode_pbar.update(1)
                        
                        # Start a new episode if needed
                        if active_episodes < num_episodes:
                            # Create and initialize new environment
                            env = ta.make(np.random.choice(self.env_ids) if isinstance(self.env_ids, list) else self.env_ids)
                            env = ta.wrappers.LLMObservationWrapper(env=env)
                            env.reset(num_players=2)
                            
                            player_id, observation = env.get_observation()
                            
                            # Assign new unique ID
                            env_id = active_episodes
                            
                            # Assign to the least busy GPU
                            if self.gpu_available:
                                gpu_idx = self._get_least_busy_gpu()
                                self._increment_gpu_tasks(gpu_idx)
                            else:
                                gpu_idx = -1  # CPU
                                
                            env_gpu_map[env_id] = gpu_idx
                            print(f"Environment {env_id} assigned to {'CPU' if gpu_idx < 0 else f'GPU {gpu_idx}'}")
                            
                            # Initialize episode data for this environment
                            all_episode_data[env_id] = []
                            
                            # Add to queue for processing
                            env_queue.append({
                                "env": env,
                                "env_id": env_id,
                                "done": False,
                                "current_player_id": player_id,
                                "current_observation": observation,
                                "step": 0
                            })
                            
                            active_episodes += 1
                    
                    # Periodically print GPU stats
                    env_counter += 1
                    if env_counter % 5 == 0:
                        self._print_gpu_utilization()
                    
                    # Force garbage collection to prevent memory issues
                    if gpu_idx >= 0:
                        with torch.cuda.device(gpu_idx):
                            torch.cuda.empty_cache()
        
        # Final GPU utilization report
        self._print_gpu_utilization()
        
        return collected_data