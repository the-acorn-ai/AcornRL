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

class MultiGPUTextArenaCollector(Collector):
    """
    A collector that runs episodes in TextArena environments in parallel,
    utilizing multiple GPUs to accelerate inference.
    
    The collector creates separate model instances for each GPU and distributes
    batches across available devices for maximum throughput.
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
            usable_gpus = []
            for i in range(self.num_gpus):
                # try:
                with torch.cuda.device(i):
                    torch.tensor([1.0], device=f"cuda:{i}")
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                usable_gpus.append(i)
                # except RuntimeError as e:
                #     print(f"  GPU {i}: Not usable - {e}")
            
            self.usable_gpus = usable_gpus
            self.num_gpus = len(usable_gpus)
        else:
            print("No GPUs detected, using CPU only")
            self.usable_gpus = []
        
        # Calculate per-GPU batch size (minimum 1)
        self.per_gpu_batch_size = max(1, batch_size // max(1, self.num_gpus))
        print(f"Using batch size {self.per_gpu_batch_size} per device")
    
    def _setup_model_on_device(self, model, device_idx):
        """Move model to specified device with proper initialization."""
        device = f"cuda:{device_idx}" if device_idx >= 0 else "cpu"
        return model.to(device)
    
    def _create_agent_replicas(self, agent: Agent) -> List[Tuple[Agent, int]]:
        """
        Create separate agent replicas for each GPU.
        
        Returns:
            List of tuples (agent_replica, gpu_idx)
        """
        # Check if agent has the expected model structure
        has_usable_model = hasattr(agent, 'model') and hasattr(agent.model, 'to')
        
        if not has_usable_model:
            print("Agent doesn't have a standard model attribute. Using original agent.")
            return [(agent, -1)]  # Use CPU
        
        if not self.gpu_available or self.num_gpus == 0:
            return [(agent, -1)]  # Use CPU
            
        # For single GPU, just move the original agent to that GPU
        if self.num_gpus == 1:
            gpu_idx = self.usable_gpus[0]
            agent.model = self._setup_model_on_device(agent.model, gpu_idx)
            return [(agent, gpu_idx)]
            
        # Create properly initialized model instances for each GPU
        import copy
        agent_replicas = []
        
        print(f"Creating agent replicas for {self.num_gpus} GPUs")
        for gpu_idx in self.usable_gpus:
            # Deep copy the agent
            agent_copy = copy.deepcopy(agent)
            
            # Create model on specific GPU
            agent_copy.model = self._setup_model_on_device(agent_copy.model, gpu_idx)
            
            # Store device information for debugging
            agent_copy._device_idx = gpu_idx
            
            agent_replicas.append((agent_copy, gpu_idx))
            
        return agent_replicas
    
    def _process_batch(self, agent_with_device, batch_observations):
        """Process a batch of observations with agent on its assigned device."""
        agent, device_idx = agent_with_device
        
        if device_idx >= 0:
            # Run in specific CUDA context to keep tensors on correct device
            with torch.cuda.device(device_idx):
                return [agent(observation=obs) for obs in batch_observations]
        else:
            # CPU processing
            return [agent(observation=obs) for obs in batch_observations]
    
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
        
        # Create agent replicas for each available GPU
        agent_replicas = self._create_agent_replicas(agent)
        
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
                    num_to_process = min(self.batch_size, len(env_queue))
                    batch_envs = [env_queue.popleft() for _ in range(num_to_process)]
                    
                    # Prepare batch of observations for the model
                    batch_observations = [env_data["current_observation"] for env_data in batch_envs]
                    
                    if len(agent_replicas) == 1:
                        # Single device mode (CPU or single GPU)
                        batch_actions_with_reasoning = self._process_batch(
                            agent_replicas[0], batch_observations
                        )
                    else:
                        # Multi-GPU mode - distribute processing across GPUs
                        batch_results = []
                        
                        # Split observations into chunks for each GPU
                        chunks = []
                        chunk_size = max(1, len(batch_observations) // len(agent_replicas))
                        
                        for i in range(0, len(batch_observations), chunk_size):
                            chunks.append(batch_observations[i:i+chunk_size])
                        
                        # Launch parallel processing with a thread for each GPU
                        with ThreadPoolExecutor(max_workers=len(agent_replicas)) as executor:
                            futures = []
                            
                            # Process each chunk on a different GPU
                            for i, (chunk, agent_replica) in enumerate(zip(chunks, agent_replicas)):
                                if i >= len(chunks):
                                    break
                                future = executor.submit(self._process_batch, agent_replica, chunk)
                                futures.append(future)
                            
                            # If we have more chunks than GPUs, use round-robin assignment
                            if len(chunks) > len(agent_replicas):
                                for i, chunk in enumerate(chunks[len(agent_replicas):]):
                                    agent_idx = i % len(agent_replicas)
                                    future = executor.submit(self._process_batch, agent_replicas[agent_idx], chunk)
                                    futures.append(future)
                            
                            # Gather results from all futures
                            for future in as_completed(futures):
                                batch_results.extend(future.result())
                        
                        batch_actions_with_reasoning = batch_results
                    
                    # Extract actions and reasoning
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
    
    def _run_episode(
        self, 
        agent1: Agent,
        agent2: Agent
    ) -> List[Dict[str, Any]]:
        """
        Run a single episode in an environment.
        
        Args:
            agent1: First agent to use
            agent2: Second agent to use
            
        Returns:
            List of (observation, action, reward) samples from the episode
        """
        print("\n\nRUN EPISODE\n\n")
        # randomly set agents
        if np.random.uniform() < 0.5:
            agents = {0: agent1, 1: agent2}
        else:
            agents = {0: agent2, 1: agent1}
        
        episode_data = []

        # create the environment
        env = ta.make(self.env_ids if not isinstance(self.env_ids, list) else np.random.choice(self.env_ids))

        # wrap the env
        env = ta.wrappers.LLMObservationWrapper(env=env)
        
        # Reset the environment
        env.reset(num_players=len(agents))
        done = False
        step = 0
        
        # Run episode until done or max steps reached
        with tqdm(total=self.max_steps_per_episode, desc="Running episode", leave=False) as pbar:
            while not done and step < self.max_steps_per_episode:
                # get observation and current player 
                player_id, observation = env.get_observation()
                
                # get player action
                action, reasoning = agents[player_id](observation=observation)
                
                # execute step in environment
                done, info = env.step(action=action)
                
                # track everything
                episode_data.append({
                    "player_id": player_id,
                    "observation": observation,
                    "reasoning": reasoning,
                    "action": action,
                    "step": step,
                })
                step += 1
                pbar.update(1)  # Increment progress bar
        
        # get game rewards
        rewards = env.close()
        
        # add rewards
        for i in range(len(episode_data)):
            # get the player's reward
            episode_data[i]["final_reward"] = rewards[episode_data[i]["player_id"]]
            episode_data[i]["full_length"] = len(episode_data)
        
        return episode_data