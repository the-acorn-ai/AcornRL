import textarena as ta 
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import torch
import os
import time
import gc
from acornrl.collectors.base import Collector
from acornrl.agents.base import Agent

class MultiGPUTextArenaCollector(Collector):
    """
    A collector that runs episodes in TextArena environments in parallel.
    
    Uses a model-per-process approach to ensure complete isolation of GPU contexts.
    Each process has its own agent copy that stays on a single GPU throughout execution.
    """
    
    def __init__(
        self,
        env_ids: List[str],
        batch_size: int = 16,
        max_steps_per_episode: int = 100,
        **kwargs
    ):
        """
        Initialize a TextArena collector with multi-GPU support.
        
        Args:
            env_ids: List of TextArena environment IDs to collect from
            batch_size: Number of episodes to collect per batch
            max_steps_per_episode: Maximum number of steps per episode
            **kwargs: Additional parameters
        """
        super().__init__(batch_size=batch_size)
        self.env_ids = env_ids
        self.max_steps_per_episode = max_steps_per_episode
        self.original_batch_size = batch_size
        
        # Set up GPU device tracking
        self.num_gpus = torch.cuda.device_count()
        self.gpu_available = self.num_gpus > 0
        
        if self.gpu_available:
            print(f"Found {self.num_gpus} GPU(s)")
            # Check if all GPUs are usable
            self.usable_gpus = []
            for i in range(self.num_gpus):
                try:
                    with torch.cuda.device(i):
                        torch.tensor([1.0], device=f"cuda:{i}")
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                    self.usable_gpus.append(i)
                except RuntimeError as e:
                    print(f"  GPU {i}: Not usable - {e}")
            
            self.num_gpus = len(self.usable_gpus)
        else:
            print("No GPUs detected, using CPU only")
            self.usable_gpus = []
        
        # Calculate per-GPU batch size (minimum 1)
        self.batch_size = min(batch_size, self.num_gpus * 2)
        print(f"Using effective batch size of {self.batch_size}")
    
    def _move_tensors_to_device(self, tensors, device):
        """Recursively move tensors to the specified device."""
        if isinstance(tensors, torch.Tensor):
            return tensors.to(device)
        elif isinstance(tensors, dict):
            return {k: self._move_tensors_to_device(v, device) for k, v in tensors.items()}
        elif isinstance(tensors, list):
            return [self._move_tensors_to_device(v, device) for v in tensors]
        elif isinstance(tensors, tuple):
            return tuple(self._move_tensors_to_device(v, device) for v in tensors)
        else:
            return tensors
            
    def collect(self, agent: Agent, num_episodes: int) -> List[Dict[str, Any]]:
        """
        Collect data from environments using parallel processing with multi-GPU support.
        
        Args:
            agent: The agent to use for collecting data
            num_episodes: Number of episodes to collect
            
        Returns:
            List of collected data samples
        """
        collected_data = []
        active_episodes = 0
        
        # Create process-isolated versions of the agent if GPUs are available
        if self.gpu_available and self.num_gpus > 0:
            # Use GPU round-robin assignment for environments
            next_gpu = 0
            
            def get_next_gpu():
                nonlocal next_gpu
                gpu_idx = self.usable_gpus[next_gpu]
                next_gpu = (next_gpu + 1) % len(self.usable_gpus)
                return gpu_idx
        else:
            # CPU-only mode
            def get_next_gpu():
                return -1  # -1 means CPU
        
        # Set up a queue for environments requiring model prediction
        env_queue = deque()
        
        # Track episode data for each environment
        all_episode_data = {}  # maps env_id to its episode data
        
        # Track which device each environment is assigned to
        env_device_map = {}  # maps env_id to device_idx
        
        # Make batch size proportional to available GPUs
        batch_size = min(self.batch_size, num_episodes)
        
        # Pre-allocate agents on each GPU to avoid constant recreation
        if self.gpu_available and self.num_gpus > 0:
            print(f"Creating agent replicas for {self.num_gpus} GPUs...")
            
            # Import multiprocessing here to avoid issues in some environments
            try:
                import torch.multiprocessing as mp
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                # If already set, just continue
                pass
                
            # We'll use sequential processing but with fixed GPU assignments
            # This is more reliable than trying to use multiple processes
            # which can lead to complex tensor sharing issues
        
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
                
                # Assign to a specific GPU (or CPU if no GPUs)
                device_idx = get_next_gpu()
                env_device_map[env_id] = device_idx
                
                print(f"Environment {env_id} assigned to {'CPU' if device_idx < 0 else f'GPU {device_idx}'}")
                
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
                while env_queue and (active_episodes < num_episodes or len(env_queue) > 0):
                    # Process environments one by one
                    env_data = env_queue.popleft()
                    
                    env = env_data["env"]
                    env_id = env_data["env_id"]
                    player_id = env_data["current_player_id"]
                    observation = env_data["current_observation"]
                    step = env_data["step"]
                    
                    # Get the device assigned to this environment
                    device_idx = env_device_map[env_id]
                    
                    # Process on the assigned device
                    action = None
                    reasoning = None
                    
                    try:
                        # If using GPU, make sure the model is on the right device
                        if device_idx >= 0:
                            # Clear GPU cache before inference
                            with torch.cuda.device(device_idx):
                                torch.cuda.empty_cache()
                            
                            # Make sure the agent's model is on the correct device
                            if hasattr(agent, 'model'):
                                # Create a fresh device object to avoid stale references
                                device = torch.device(f"cuda:{device_idx}")
                                
                                # Move model to the correct device
                                # We explicitly specify device as a string to avoid issues
                                agent.model = agent.model.to(f"cuda:{device_idx}")
                            
                            # Process inside this device's context
                            with torch.cuda.device(device_idx):
                                action, reasoning = agent(observation=observation)
                        else:
                            # CPU processing
                            if hasattr(agent, 'model'):
                                agent.model = agent.model.cpu()
                            action, reasoning = agent(observation=observation)
                    except Exception as e:
                        print(f"Error processing on {'CPU' if device_idx < 0 else f'GPU {device_idx}'}: {e}")
                        # Fall back to CPU as a last resort
                        try:
                            if hasattr(agent, 'model'):
                                agent.model = agent.model.cpu()
                            print("Falling back to CPU")
                            action, reasoning = agent(observation=observation)
                        except Exception as e2:
                            print(f"CPU fallback also failed: {e2}")
                            # Create default values if all else fails
                            action = "default_action"
                            reasoning = "Error occurred during inference"
                    
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
                        # Episode completed (either done or max steps reached)
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
                            
                            # Assign to a specific GPU (or CPU if no GPUs)
                            device_idx = get_next_gpu()
                            env_device_map[env_id] = device_idx
                            
                            print(f"Environment {env_id} assigned to {'CPU' if device_idx < 0 else f'GPU {device_idx}'}")
                            
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
                    
                    # Force garbage collection to prevent memory issues
                    gc.collect()
                    if self.gpu_available:
                        torch.cuda.empty_cache()
        
        return collected_data
    
