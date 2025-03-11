import textarena as ta 
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import torch
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from acornrl.collectors.base import Collector
from acornrl.agents.base import Agent

class ParallelTextArenaCollectorDistributed(Collector):
    """
    A collector that runs episodes in TextArena environments in parallel,
    utilizing multiple GPUs if available.
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
        
        # Set up GPU device tracking
        self.num_gpus = torch.cuda.device_count()
        self.gpu_available = self.num_gpus > 0
        
        if self.gpu_available:
            print(f"Found {self.num_gpus} GPU(s)")
            # Check if all GPUs are usable
            for i in range(self.num_gpus):
                try:
                    with torch.cuda.device(i):
                        torch.tensor([1.0], device=f"cuda:{i}")
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                except RuntimeError as e:
                    print(f"  GPU {i}: Not usable - {e}")
                    self.num_gpus -= 1
        else:
            print("No GPUs detected, using CPU only")
        
        # Calculate per-GPU batch size (minimum 1)
        self.per_gpu_batch_size = max(1, batch_size // max(1, self.num_gpus))
        print(f"Using batch size {self.per_gpu_batch_size} per device")
    
    def _create_agent_copies(self, agent: Agent) -> List[Agent]:
        """Create copies of the agent for each available GPU."""
        if not self.gpu_available or self.num_gpus <= 1:
            return [agent]
        
        agent_copies = []
        # Determine if agent has a model attribute that can be moved to different devices
        has_model = hasattr(agent, 'model')
        
        for i in range(self.num_gpus):
            if has_model:
                # Create a deep copy of the agent for each GPU
                import copy
                agent_copy = copy.deepcopy(agent)
                
                # Move model to the specific GPU
                device = torch.device(f"cuda:{i}")
                agent_copy.model.to(device)
                
                # Set index to help with debugging
                agent_copy._gpu_idx = i
                agent_copies.append(agent_copy)
            else:
                # If we can't identify a model to move between devices,
                # just use the original agent for all operations
                if i == 0:
                    agent_copy = agent
                    agent_copy._gpu_idx = 0
                    agent_copies.append(agent_copy)
                else:
                    print(f"Warning: Agent doesn't have a 'model' attribute. Using the same agent for all GPUs.")
                    break
        
        return agent_copies
    
    def _process_batch_with_agent(self, agent: Agent, observations: List[str]) -> List[Tuple[str, str]]:
        """Process a batch of observations with a specific agent."""
        return [agent(observation=obs) for obs in observations]
    
    def _process_multi_gpu(self, agent_copies: List[Agent], all_observations: List[str]) -> List[Tuple[str, str]]:
        """Process observations in parallel across multiple GPUs."""
        # Split observations into chunks for each GPU
        batches = []
        
        for i in range(0, len(all_observations), self.per_gpu_batch_size):
            batch = all_observations[i:i+self.per_gpu_batch_size]
            batches.append(batch)
        
        # Ensure we don't try to use more GPUs than we have batches
        num_batches = len(batches)
        active_agents = agent_copies[:min(self.num_gpus, num_batches)]
        
        # Process batches in parallel using thread pool
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=len(active_agents)) as executor:
            futures = []
            
            for i, (agent, batch) in enumerate(zip(active_agents, batches[:len(active_agents)])):
                future = executor.submit(self._process_batch_with_agent, agent, batch)
                futures.append(future)
            
            # If we have more batches than GPUs, process them sequentially with round-robin GPU assignment
            if num_batches > len(active_agents):
                for i, batch in enumerate(batches[len(active_agents):]):
                    agent_idx = i % len(active_agents)
                    future = executor.submit(self._process_batch_with_agent, active_agents[agent_idx], batch)
                    futures.append(future)
            
            # Collect results in order
            for future in as_completed(futures):
                results.extend(future.result())
        
        return results

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
        
        # Create agent copies for each GPU if multiple GPUs are available
        agent_copies = self._create_agent_copies(agent)
        
        # Set up a queue for the environments requiring model prediction
        env_queue = deque()
        
        # Track episode data for each environment
        all_episode_data = {}  # maps env_id to its episode data
        
        # Initialize environments in parallel
        env_to_id = {}  # maps env object to a unique ID
        
        with tqdm(total=num_episodes, desc="Collecting episodes") as episode_pbar:
            # Initialize environments
            for env_idx in range(min(self.batch_size, num_episodes)):
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
                    # Get all environments that need processing
                    batch_size = min(self.batch_size, len(env_queue))
                    batch_envs = [env_queue.popleft() for _ in range(batch_size)]
                    
                    # Prepare batch of observations for the model
                    batch_observations = [env_data["current_observation"] for env_data in batch_envs]
                    
                    # Get actions from agent(s) depending on GPU availability
                    if self.num_gpus <= 1:
                        # Single GPU or CPU mode - process sequentially
                        batch_actions_with_reasoning = self._process_batch_with_agent(agent_copies[0], batch_observations)
                    else:
                        # Multi-GPU mode - distribute across GPUs
                        batch_actions_with_reasoning = self._process_multi_gpu(agent_copies, batch_observations)
                    
                    batch_actions = [action for action, _ in batch_actions_with_reasoning]
                    batch_reasoning = [reasoning for _, reasoning in batch_actions_with_reasoning]
                    
                    # Process each environment with its predicted action
                    for i, env_data in enumerate(batch_envs):
                        env = env_data["env"]
                        env_id = env_data["env_id"]
                        player_id = env_data["current_player_id"]
                        observation = env_data["current_observation"]
                        step = env_data["step"]
                        action = batch_actions[i]
                        reasoning = batch_reasoning[i]
                        
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
        
        return collected_data