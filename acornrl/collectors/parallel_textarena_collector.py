import textarena as ta 
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
from acornrl.collectors.base import Collector
from acornrl.agents.base import Agent

class ParallelTextArenaCollector(Collector):
    """
    A collector that runs episodes in TextArena environments in parallel.
    """
    
    def __init__(
        self,
        env_ids: List[str],
        batch_size: int = 16,
        max_steps_per_episode: int = 100,
        **kwargs
    ):
        """
        Initialize a TextArena collector.
        
        Args:
            env_ids: List of TextArena environment IDs to collect from
            batch_size: Number of episodes to collect per batch
            max_steps_per_episode: Maximum number of steps per episode
            **kwargs: Additional parameters
        """
        super().__init__(batch_size=batch_size)
        self.env_ids = env_ids
        self.max_steps_per_episode = max_steps_per_episode
        
    def collect(self, agent: Agent, num_episodes: int) -> List[Dict[str, Any]]:
        collected_data = []
        active_episodes = 0
        
        # set up a queue for the environments requiring model prediction
        env_queue = deque()
        
        # track episode data for each environment
        all_episode_data = {}  # maps env_id to its episode data
        
        # initialize environments in parallel
        envs = []
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
                    # Process environments in batches
                    batch_size = min(self.batch_size, len(env_queue))
                    batch_envs = [env_queue.popleft() for _ in range(batch_size)]
                    
                    # Prepare batch of observations for the model
                    batch_observations = [env_data["current_observation"] for env_data in batch_envs]
                    
                    # Get actions from the agent (self-play, so same agent for both players)
                    batch_actions_with_reasoning = [agent(observation=obs) for obs in batch_observations]
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
                        
                        if not done:
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
    