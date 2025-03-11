# import textarena as ta 
# import numpy as np
# from typing import List, Dict, Any, Optional, Union, Tuple
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
#             agent: Agent to use for data collection
#             num_episodes: Number of episodes to collect
            
#         Returns:
#             List of collected data samples
#         """
#         collected_data = []

#         for _ in range(num_episodes):
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
#             agent: Agent to use
            
#         Returns:
#             List of (observation, action, reward) samples from the episode
#         """
#         # randomly set agents
#         if np.random.uniform() < 0.5:
#             agents = {0: agent1, 1: agent2}
#         else:
#             agents = {0: agent2, 1: agent1}


#         episode_data = []

#         # create the environment
#         env = ta.make(self.env_ids)
        
#         # Reset the environment
#         env.reset(num_players=len(agents))
#         done = False
#         step = 0

#         # Run episode until done or max steps reached
#         while not done:
#             # get observation and current player 
#             player_id, observation = env.get_observation()

#             # get player action
#             action, reasoning = agents[player_id](observation=observation)

#             # execute step in environment
#             done, info = env.step(action=action)

#             # track everything
#             episode_data.append({
#                 "player_id": player_id,
#                 "observation": observation,
#                 "reasoning": reasoning,
#                 "action": action,
#                 "step": step,
#                 # "info": info
#             })
#             step += 1
        
#         # get game rewards
#         rewards = env.close()

#         # add rewards
#         for i in range(len(episode_data)):
#             # get the players reward
#             episode_data[i]["final_reward"] = rewards[episode_data[i]["player_id"]]

#         # return 
#         return episode_data

import textarena as ta 
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from acornrl.collectors.base import Collector
from acornrl.agents.base import Agent

class SequentialTextArenaCollector(Collector):
    """
    A collector that runs episodes in TextArena environments sequentially.
    """
    
    def __init__(
        self,
        env_ids: List[str],
        batch_size: int = 16,
        **kwargs
    ):
        """
        Initialize a TextArena collector.
        
        Args:
            env_ids: List of TextArena environment IDs to collect from
            batch_size: Number of episodes to collect per batch
            **kwargs: Additional parameters
        """
        super().__init__(batch_size=batch_size)
        self.env_ids = env_ids
        
    def collect(
        self, 
        agent1: Agent,
        agent2: Agent, 
        num_episodes: int
    ) -> List[Dict[str, Any]]:
        """
        Collect data by running episodes with the agent.
        
        Args:
            agent1: First agent to use for data collection
            agent2: Second agent to use for data collection
            num_episodes: Number of episodes to collect
            
        Returns:
            List of collected data samples
        """
        collected_data = []

        for _ in tqdm(range(num_episodes), desc="Collecting episodes"):
            episode_data = self._run_episode(agent1, agent2)
            collected_data.extend(episode_data)
        
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
        env = ta.make(self.env_ids)

        # wrap the env
        env = ta.wrappers.LLMObservationWrapper(env=env)
        
        # Reset the environment
        env.reset(num_players=len(agents))
        done = False
        step = 0
        
        # Run episode until done or max steps reached
        with tqdm(total=100, desc="Running episode", leave=False) as pbar:
            while not done:
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
