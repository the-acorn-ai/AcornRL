import textarena as ta 
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import torch
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from acornrl.collectors.base import Collector
from acornrl.agents.base import Agent

class MultiGPUTextArenaCollector(Collector):
    """
    A collector that runs episodes in TextArena environments in parallel,
    utilizing multiple GPUs for inference by using a work-stealing approach.
    
    Each GPU has its own independent worker queue to process batches,
    avoiding cross-device tensor operations.
    """
    
    def __init__(
        self,
        env_ids: List[str],
        batch_size: int = 1,
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
        self.per_gpu_batch_size = max(1, batch_size // max(1, self.num_gpus))
        print(f"Using batch size {self.per_gpu_batch_size} per device")
    
    def _setup_gpu_worker(self, agent: Agent, gpu_idx: int):
        """
        Setup a worker for a specific GPU.
        
        Args:
            agent: The base agent to copy
            gpu_idx: GPU index to use
            
        Returns:
            Tuple of (worker_id, worker_fn) where worker_fn processes observations
        """
        # Create a deep copy of the agent
        import copy
        agent_copy = copy.deepcopy(agent)
        
        # Move the entire model to the specified GPU device
        if hasattr(agent_copy, 'model'):
            # Use to() with explicit parameter to ensure everything moves together
            device = torch.device(f"cuda:{gpu_idx}")
            agent_copy.model = agent_copy.model.to(device)
            
            # Set a device attribute that can be checked in __call__
            if not hasattr(agent_copy, '_device'):
                agent_copy._device = device
            
            print(f"Created worker for GPU {gpu_idx}")
        
        # Function to process a batch with this worker
        def worker_fn(observations):
            # Ensure all operations happen on the correct device context
            with torch.cuda.device(gpu_idx):
                # Process each observation individually to keep tensors on same device
                results = []
                for obs in observations:
                    try:
                        result = agent_copy(observation=obs)
                        results.append(result)
                    except Exception as e:
                        print(f"Error in GPU {gpu_idx} worker: {e}")
                        # Fall back to CPU if needed
                        if hasattr(agent_copy, 'model'):
                            cpu_model = agent_copy.model.cpu()
                            agent_copy.model = cpu_model
                            result = agent_copy(observation=obs)
                            # Move back to GPU
                            agent_copy.model = agent_copy.model.to(device)
                            results.append(result)
                return results
        
        return (gpu_idx, worker_fn)
        
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
        
        # Set up worker queues for each GPU (or CPU if no GPUs)
        if self.gpu_available and self.num_gpus > 0:
            workers = []
            for gpu_idx in self.usable_gpus:
                workers.append(self._setup_gpu_worker(agent, gpu_idx))
        else:
            # CPU-only mode
            workers = [(0, lambda obs: [agent(observation=o) for o in obs])]
        
        # Create worker thread pool
        executor = ThreadPoolExecutor(max_workers=len(workers))
        
        # Set up a queue for environments requiring model prediction
        env_queue = deque()
        
        # Track episode data for each environment
        all_episode_data = {}  # maps env_id to its episode data
        
        # Initialize environments in parallel
        env_to_id = {}  # maps env object to a unique ID
        
        # Set default batch size based on available workers
        batch_size = min(self.original_batch_size, 
                        self.per_gpu_batch_size * len(workers))
        
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
                env_to_id[env] = env_id
                
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
                    # Process environments in worker-sized batches
                    per_worker_batch = max(1, min(self.per_gpu_batch_size, 
                                                len(env_queue) // len(workers)))
                    
                    # Group environments for each worker
                    worker_batches = []
                    worker_env_data = []
                    
                    # Take environments from the queue
                    environments_to_process = min(len(env_queue), 
                                                per_worker_batch * len(workers))
                    
                    for _ in range(environments_to_process):
                        env_data = env_queue.popleft()
                        worker_env_data.append(env_data)
                    
                    # Group by worker batches
                    for i in range(0, len(worker_env_data), per_worker_batch):
                        batch = worker_env_data[i:i+per_worker_batch]
                        worker_batches.append([env_data["current_observation"] for env_data in batch])
                    
                    # Submit work to each worker
                    futures = []
                    for i, (worker_id, worker_fn) in enumerate(workers):
                        if i < len(worker_batches):
                            future = executor.submit(worker_fn, worker_batches[i])
                            futures.append((future, i, len(worker_batches[i])))
                    
                    # Process results as they complete
                    all_results = []
                    for future, worker_idx, batch_size in futures:
                        try:
                            results = future.result()
                            all_results.extend(results)
                        except Exception as e:
                            print(f"Error in worker {worker_idx}: {e}")
                            # Fall back to CPU processing if a GPU worker fails
                            start_idx = sum(len(worker_batches[j]) for j in range(worker_idx))
                            end_idx = start_idx + batch_size
                            batch = worker_env_data[start_idx:end_idx]
                            results = [agent(observation=env_data["current_observation"]) 
                                      for env_data in batch]
                            all_results.extend(results)
                    
                    batch_actions = [action for action, _ in all_results]
                    batch_reasoning = [reasoning for _, reasoning in all_results]
                    
                    # Process each environment with its predicted action
                    for i, env_data in enumerate(worker_env_data):
                        env = env_data["env"]
                        env_id = env_data["env_id"]
                        player_id = env_data["current_player_id"]
                        observation = env_data["current_observation"]
                        step = env_data["step"]
                        
                        try:
                            action = batch_actions[i]
                            reasoning = batch_reasoning[i]
                        except IndexError:
                            # In case of any problems with results, fall back to single item processing
                            action, reasoning = agent(observation=observation)
                        
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
                                env_to_id[env] = env_id
                                
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
        
        # Clean up executor
        executor.shutdown()
        
        return collected_data
    