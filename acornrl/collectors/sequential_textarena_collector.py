# # import textarena as ta 
# # import numpy as np
# # from typing import List, Dict, Any, Optional, Union, Tuple
# # from acornrl.collectors.base import Collector
# # from acornrl.agents.base import Agent

# # class SequentialTextArenaCollector(Collector):
# #     """
# #     A collector that runs episodes in TextArena environments sequentially.
# #     """
    
# #     def __init__(
# #         self,
# #         env_ids: List[str],
# #         batch_size: int = 16,
# #         **kwargs
# #     ):
# #         """
# #         Initialize a TextArena collector.
        
# #         Args:
# #             env_ids: List of TextArena environment IDs to collect from
# #             batch_size: Number of episodes to collect per batch
# #             **kwargs: Additional parameters
# #         """
# #         super().__init__(batch_size=batch_size)
# #         self.env_ids = env_ids
        
# #     def collect(
# #         self, 
# #         agent1: Agent,
# #         agent2: Agent, 
# #         num_episodes: int
# #     ) -> List[Dict[str, Any]]:
# #         """
# #         Collect data by running episodes with the agent.
        
# #         Args:
# #             agent: Agent to use for data collection
# #             num_episodes: Number of episodes to collect
            
# #         Returns:
# #             List of collected data samples
# #         """
# #         collected_data = []

# #         for _ in range(num_episodes):
# #             episode_data = self._run_episode(agent1, agent2)
# #             collected_data.extend(episode_data)
        
# #         return collected_data
    
# #     def _run_episode(
# #         self, 
# #         agent1: Agent,
# #         agent2: Agent
# #     ) -> List[Dict[str, Any]]:
# #         """
# #         Run a single episode in an environment.
        
# #         Args:
# #             agent: Agent to use
            
# #         Returns:
# #             List of (observation, action, reward) samples from the episode
# #         """
# #         # randomly set agents
# #         if np.random.uniform() < 0.5:
# #             agents = {0: agent1, 1: agent2}
# #         else:
# #             agents = {0: agent2, 1: agent1}


# #         episode_data = []

# #         # create the environment
# #         env = ta.make(self.env_ids)
        
# #         # Reset the environment
# #         env.reset(num_players=len(agents))
# #         done = False
# #         step = 0

# #         # Run episode until done or max steps reached
# #         while not done:
# #             # get observation and current player 
# #             player_id, observation = env.get_observation()

# #             # get player action
# #             action, reasoning = agents[player_id](observation=observation)

# #             # execute step in environment
# #             done, info = env.step(action=action)

# #             # track everything
# #             episode_data.append({
# #                 "player_id": player_id,
# #                 "observation": observation,
# #                 "reasoning": reasoning,
# #                 "action": action,
# #                 "step": step,
# #                 # "info": info
# #             })
# #             step += 1
        
# #         # get game rewards
# #         rewards = env.close()

# #         # add rewards
# #         for i in range(len(episode_data)):
# #             # get the players reward
# #             episode_data[i]["final_reward"] = rewards[episode_data[i]["player_id"]]

# #         # return 
# #         return episode_data

# import textarena as ta 
# import numpy as np
# from tqdm import tqdm
# from typing import List, Dict, Any
# from acornrl.collectors.base import Collector
# from acornrl.agents.base import Agent

# class SequentialTextArenaCollector(Collector):
#     """
#     A collector that runs episodes in TextArena environments sequentially.
#     """
    
#     def __init__(
#         self,
#         env_ids: List[str],
#         batch_size: int = 16,
#         **kwargs
#     ):
#         """
#         Initialize a TextArena collector.
        
#         Args:
#             env_ids: List of TextArena environment IDs to collect from
#             batch_size: Number of episodes to collect per batch
#             **kwargs: Additional parameters
#         """
#         super().__init__(batch_size=batch_size)
#         self.env_ids = env_ids
        
#     def collect(
#         self, 
#         agent1: Agent,
#         agent2: Agent, 
#         num_episodes: int
#     ) -> List[Dict[str, Any]]:
#         """
#         Collect data by running episodes with the agent.
        
#         Args:
#             agent1: First agent to use for data collection
#             agent2: Second agent to use for data collection
#             num_episodes: Number of episodes to collect
            
#         Returns:
#             List of collected data samples
#         """
#         collected_data = []

#         for _ in tqdm(range(num_episodes), desc="Collecting episodes"):
#             episode_data = self._run_episode(agent1, agent2)
#             collected_data.extend(episode_data)
        
#         return collected_data
    
#     def _run_episode(
#         self, 
#         agent1: Agent,
#         agent2: Agent
#     ) -> List[Dict[str, Any]]:
#         """
#         Run a single episode in an environment.
        
#         Args:
#             agent1: First agent to use
#             agent2: Second agent to use
            
#         Returns:
#             List of (observation, action, reward) samples from the episode
#         """
#         print("\n\nRUN EPISODE\n\n")
#         # randomly set agents
#         if np.random.uniform() < 0.5:
#             agents = {0: agent1, 1: agent2}
#         else:
#             agents = {0: agent2, 1: agent1}
        
#         episode_data = []

#         # create the environment
#         env = ta.make(self.env_ids)

#         # wrap the env
#         env = ta.wrappers.LLMObservationWrapper(env=env)
        
#         # Reset the environment
#         env.reset(num_players=len(agents))
#         done = False
#         step = 0
        
#         # Run episode until done or max steps reached
#         with tqdm(total=100, desc="Running episode", leave=False) as pbar:
#             while not done:
#                 # get observation and current player 
#                 player_id, observation = env.get_observation()
                
#                 # get player action
#                 action, reasoning = agents[player_id](observation=observation)
                
#                 # execute step in environment
#                 done, info = env.step(action=action)
                
#                 # track everything
#                 episode_data.append({
#                     "player_id": player_id,
#                     "observation": observation,
#                     "reasoning": reasoning,
#                     "action": action,
#                     "step": step,
#                 })
#                 step += 1
#                 pbar.update(1)  # Increment progress bar
        
#         # get game rewards
#         rewards = env.close()
        
#         # add rewards
#         for i in range(len(episode_data)):
#             # get the player's reward
#             episode_data[i]["final_reward"] = rewards[episode_data[i]["player_id"]]
#             episode_data[i]["full_length"] = len(episode_data)
        
#         return episode_data


import textarena as ta 
import numpy as np
import time
import uuid
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union, Tuple
from acornrl.collectors.base import Collector
from acornrl.agents.base import Agent

class SequentialTextArenaCollector(Collector):
    """
    A collector that runs episodes in TextArena environments and captures structured LLM outputs.
    """
    
    def __init__(
        self,
        env_ids: List[str],
        batch_size: int = 16,
        max_episode_steps: int = 50,
        standard_prompt: str = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format.",
        **kwargs
    ):
        """
        Initialize a TextArena collector.
        
        Args:
            env_ids: List of TextArena environment IDs to collect from
            batch_size: Number of episodes to collect per batch
            max_episode_steps: Maximum number of steps per episode
            standard_prompt: Standard prompt to include with each observation
            **kwargs: Additional parameters
        """
        super().__init__(batch_size=batch_size)
        self.env_ids = env_ids if isinstance(env_ids, list) else [env_ids]
        self.max_episode_steps = max_episode_steps
        self.standard_prompt = standard_prompt
        
    def collect(
        self, 
        agent1: Agent,
        agent2: Agent, 
        num_episodes: int
    ) -> List[Dict[str, Any]]:
        """
        Collect data by running episodes with the agents.
        
        Args:
            agent1: First agent to use for data collection
            agent2: Second agent to use for data collection
            num_episodes: Number of episodes to collect
            
        Returns:
            List of collected data samples with structured reasoning and actions
        """
        all_episode_data = []

        for episode_idx in tqdm(range(num_episodes), desc="Collecting episodes"):
            try:
                # Generate a unique episode ID
                episode_id = str(uuid.uuid4())
                
                # Run the episode and get the data
                episode_data = self._run_episode(agent1, agent2, episode_id)
                
                # Add all samples from this episode to our collection
                if episode_data:
                    all_episode_data.extend(episode_data)
                else:
                    print(f"Warning: Episode {episode_idx} produced no data")
                
            except Exception as e:
                print(f"Error in episode {episode_idx}: {e}")
                # Continue with the next episode if one fails
                continue
        
        return all_episode_data
    
    def format_observation(self, observation: str) -> str:
        """
        Format the observation with structured tags.
        
        Args:
            observation: Raw observation text
            
        Returns:
            Formatted observation
        """
        # Check if already formatted
        if "<｜begin▁of▁sentence｜>" in observation:
            return observation
            
        # Add structured format
        return (
            f"<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within [ ].\n"
            f"{self.standard_prompt}"
            f"\n<｜User｜>{observation}"
            f"\n<｜Assistant｜><think>\n"
        )
    
    def _run_episode(
        self, 
        agent1: Agent,
        agent2: Agent,
        episode_id: str
    ) -> List[Dict[str, Any]]:
        """
        Run a single episode in an environment with structured outputs.
        
        Args:
            agent1: First agent to use
            agent2: Second agent to use
            episode_id: Unique identifier for this episode
            
        Returns:
            List of experience dictionaries from the episode
        """
        # Randomly assign agents to player IDs
        if np.random.uniform() < 0.5:
            agents = {0: agent1, 1: agent2}
        else:
            agents = {0: agent2, 1: agent1}
        
        episode_data = []

        # Select environment from available IDs
        env_id = np.random.choice(self.env_ids)
        
        try:
            # Create the environment
            env = ta.make(env_id)
            
            # Add LLM observation wrapper if available
            try:
                env = ta.wrappers.LLMObservationWrapper(env=env)
            except Exception as e:
                # If wrapper not available, continue with raw env
                print(f"Warning: Could not apply LLMObservationWrapper: {e}")
            
            # Reset the environment
            env.reset(num_players=len(agents))
            done = False
            step = 0
            
            # Run episode until done or max steps reached
            while not done and step < self.max_episode_steps:
                # Get observation and current player
                try:
                    player_id, observation = env.get_observation()
                except Exception as e:
                    print(f"Error getting observation: {e}")
                    break
                
                # Safety check - ensure observation is not None
                if observation is None:
                    print("Warning: Received None observation, skipping step")
                    step += 1
                    continue
                
                # Format observation with structure if needed
                formatted_observation = self.format_observation(observation)
                
                # Get player action with reasoning
                try:
                    action, reasoning = agents[player_id](formatted_observation)
                except Exception as e:
                    print(f"Error getting action: {e}")
                    # Use a default action if agent fails
                    action = "pass"
                    reasoning = "Failed to generate reasoning."
                
                # Safety check - ensure action is not None
                if action is None:
                    print("Warning: Agent returned None action, using 'pass'")
                    action = "pass"
                
                # Execute step in environment
                try:
                    done, info = env.step(action=action)
                except Exception as e:
                    print(f"Error executing step in environment: {e}")
                    break
                
                # Track this step's data with structured format
                step_data = {
                    "episode_id": episode_id,
                    "player_id": player_id,
                    "observation": formatted_observation,  # Store formatted observation
                    "reasoning": reasoning,              # Store reasoning separately
                    "action": action,                    # Store action
                    "raw_observation": observation,      # Store original observation too
                    "step": step,
                    "info": info if info is not None else {}
                }
                
                episode_data.append(step_data)
                step += 1
            
            # Get game rewards
            try:
                rewards = env.close()
            except Exception as e:
                print(f"Error getting rewards: {e}")
                # Fallback to zero rewards
                rewards = {player_id: 0.0 for player_id in agents.keys()}
            
            # Add rewards to all samples from this episode
            for i in range(len(episode_data)):
                player_id = episode_data[i]["player_id"]
                episode_data[i]["final_reward"] = rewards.get(player_id, 0.0)
                episode_data[i]["full_length"] = len(episode_data)
            
            return episode_data
            
        except Exception as e:
            print(f"Error running episode with environment {env_id}: {e}")
            return []
    
    def prepare_for_training(
        self, 
        collected_data: List[Dict[str, Any]]
    ) -> Dict[str, List]:
        """
        Prepare collected data for training with structured format.
        
        Args:
            collected_data: Raw collected data
            
        Returns:
            Dict containing processed training samples
        """
        training_samples = {
            "observations": [],
            "reasoning": [],
            "actions": [],
            "rewards": [],
            "player_ids": [],
            "episode_ids": []
        }
        
        for sample in collected_data:
            training_samples["observations"].append(sample["observation"])
            training_samples["reasoning"].append(sample["reasoning"])
            training_samples["actions"].append(sample["action"])
            
            # Calculate discounted reward
            step = sample["step"]
            full_length = sample["full_length"]
            reward = sample["final_reward"]
            discounted_reward = reward * (0.95 ** (full_length - step - 1))
            
            training_samples["rewards"].append(discounted_reward)
            training_samples["player_ids"].append(sample["player_id"])
            training_samples["episode_ids"].append(sample["episode_id"])
        
        return training_samples