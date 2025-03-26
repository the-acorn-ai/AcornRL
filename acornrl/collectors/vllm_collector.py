import os, csv, time, torch
import statistics, logging, threading 
import concurrent.futures 
import numpy as np 
from tqdm import tqdm 
from typing import Optional, List, Dict, Any


import textarena as ta
from acornrl.inference import VLLMServerManager, VLLMInferenceClient
from acornrl.utils import CSVManagerLogging, CSVManagerData


# class VLLMCollector(Collector):
class VLLMCollector:
    def __init__(
        self, env_ids: List[str], checkpoint_path: str, output_dir: str, max_new_tokens: int, 
        max_workers: int, temperature: float = 0.7, top_p: float = 0.9, 
        tensor_parallel_size: int = 1, gpus: Optional[List[int]] = None, base_port: int = 8000,
    ):
        self.env_ids = env_ids
        self.output_dir = output_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.base_port = base_port
        self.max_workers = max_workers

        self.checkpoint_path = checkpoint_path

        # generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.gpus = list(range(torch.cuda.device_count())) if gpus is None else gpus
            
        self.logger = logging.getLogger(__name__)
        self.server_manager = None 
        self.vllm_clients = []

        self.csv_manager_logging = CSVManagerLogging(output_dir=output_dir)

    def __enter__(self):
        """Context manager entry: Setup vLLM clients."""
        self.server_manager = VLLMServerManager(
            model_path=self.checkpoint_path, max_seq_len=self.max_new_tokens, gpus=self.gpus,
            tensor_parallel_size=self.tensor_parallel_size, base_port=self.base_port,
        )
        self.server_manager.start_servers()
        
        # Initialize clients
        self.vllm_clients = []
        for i in range(self.server_manager.num_servers):
            client = self.server_manager.get_client(i)
            self.vllm_clients.append(client)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.server_manager:
            self.server_manager.stop_servers()
            self.vllm_clients = []

    def collect(self, num_episodes: int, iteration: int) -> None:
        # Start CSVManagerData as a context manager
        with CSVManagerData(self.output_dir, iteration, episode_type="train") as csv_manager_data:

            def run_episode(episode_id: int) -> None:
                try:
                    client_idx = episode_id % len(self.vllm_clients)
                    vllm_client = self.vllm_clients[client_idx]

                    # Create & wrap environment 
                    env = ta.make(self.env_ids)
                    env_id = env.env_id
                    env = ta.wrappers.FirstLastObservationWrapper(env=env)
                    env.reset(num_players=2)
                    env.state.error_allowance = 0

                    episode_data = []
                    step_count = 0
                    done = False 

                    while not done:
                        player_id, observation = env.get_observation()

                        # Use vLLM for inference 
                        formatted_observation, reasoning, action = vllm_client.generate_text(
                            prompt=observation, max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature, top_p=self.top_p
                        )

                        episode_data.append({
                            "episode_id": episode_id, "env_id": env_id, "model_name": self.checkpoint_path, 
                            "player_id": player_id, "observation": observation, "formatted_observation": formatted_observation,
                            "reasoning": reasoning, "action": action, "step": step_count
                        })

                        done, info = env.step(action=action)
                        step_count += 1

                    # Episode done
                    rewards = env.close()

                    # Add rewards to episode data and send to CSV manager
                    for step_data in episode_data:
                        csv_manager_data.add_episode([
                            step_data["episode_id"], step_data["env_id"], step_data["model_name"], step_data["player_id"],
                            step_data["observation"], step_data["formatted_observation"], step_data["reasoning"],
                            step_data["action"], step_data["step"], len(episode_data), rewards[step_data["player_id"]] 
                        ])
                except Exception as e:
                    print(f"Episode collection failed with exception {e}")


            # Parallel collection
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(run_episode, i) for i in range(num_episodes)]
                progress_bar = tqdm(total=num_episodes, desc="Collecting episodes")
                for future in concurrent.futures.as_completed(futures):
                    future.result()  # wait for the result, but nothing to do with it here
                    progress_bar.update(1)
                progress_bar.close()



    def evaluate(self, num_episodes: int, iteration: int) -> None:
        opponent_name = "google/gemini-2.0-flash-lite-001"
        opponent_name = "openai/gpt-4o"
        # Start CSVManagerData as a context manager
        with CSVManagerData(self.output_dir, iteration, episode_type="eval") as csv_manager_data:

            def run_episode(episode_id: int) -> None:
                
                client_idx = episode_id % len(self.vllm_clients)
                vllm_client = self.vllm_clients[client_idx]

                # assign player roles
                agent_idx = int(np.random.uniform() < 0.5)
                agents = {agent_idx: vllm_client, 1-agent_idx: ta.agents.OpenRouterAgent(model_name=opponent_name)}

                # Create & wrap environment 
                env = ta.make(self.env_ids)
                env_id = env.env_id
                env = ta.wrappers.FirstLastObservationWrapper(env=env)
                env.reset(num_players=2)

                episode_data = []
                step_count = 0
                done = False 

                while not done:
                    player_id, observation = env.get_observation()

                    # route to correct model 
                    model = agents[player_id]
                    if isinstance(model, VLLMInferenceClient):
                        # Use vLLM for inference 
                        formatted_observation, reasoning, action = vllm_client.generate_text(
                            prompt=observation, max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature, top_p=self.top_p
                        )
                        model_name = self.checkpoint_path
                    else:
                        model_name = opponent_name
                        formatted_observation, reasoning, action = None, None, model(observation)

                    episode_data.append({
                        "episode_id": episode_id, "env_id": env_id, "model_name": model_name, "player_id": player_id,
                        "observation": observation, "formatted_observation": formatted_observation,
                        "reasoning": reasoning, "action": action, "step": step_count
                    })

                    done, info = env.step(action=action)
                    step_count += 1

                # Episode done
                rewards = env.close()

                # Add rewards to episode data and send to CSV manager
                for step_data in episode_data:
                    csv_manager_data.add_episode([
                        step_data["episode_id"], step_data["env_id"], step_data["model_name"], step_data["player_id"],
                        step_data["observation"], step_data["formatted_observation"], step_data["reasoning"],
                        step_data["action"], step_data["step"], len(episode_data), rewards[step_data["player_id"]] 
                    ])


            # Parallel collection
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(run_episode, i) for i in range(num_episodes)]
                progress_bar = tqdm(total=num_episodes, desc="Collecting episodes")
                for future in concurrent.futures.as_completed(futures):
                    future.result()  # wait for the result, but nothing to do with it here
                    progress_bar.update(1)
                progress_bar.close()

    def run_evaluation(self, env_ids: Optional[List[str]] = None, num_episodes: int = 10):
        opponent_name = "google/gemini-2.0-flash-lite-001"

        eval_tracker_subdict = {
            "num_episodes": 0, "model_rewards": [], 
            "model_outcomes": [], "opponent_rewards": [], "episode_lengths": []
            }
        eval_tracker = {"train": eval_tracker_subdict.copy()}
        if env_ids is not None and len(env_ids) > 0:
            eval_tracker["eval"] = eval_tracker_subdict.copy()

        def run_episode(env_id: str, episode_id: int):
            
            client_idx = episode_id % len(self.vllm_clients)
            vllm_client = self.vllm_clients[client_idx]

            # assign player roles
            agent_idx = int(np.random.uniform() < 0.5)
            agents = {
                agent_idx: vllm_client,
                1-agent_idx: ta.agents.OpenRouterAgent(model_name=opponent_name)
            }

            # Create & wrap environment
            env = ta.make(env_id)
            env.state.error_allowance = 0
            env = ta.wrappers.LLMObservationWrapper(env=env)

            env.reset(num_players=2)
            step_count = 0
            done = False 

            while not done:
                player_id, observation = env.get_observation()

                # route to correct model
                model = agents[player_id]
                if isinstance(model, VLLMInferenceClient):
                    result = model.generate_text(
                        prompt=observation,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p
                    )
                    action = result["answer"]
                else:
                    action = model(observation)

                # step
                done, info = env.step(action=action)
                step_count += 1

            # Episode is done
            rewards = env.close()

            eval_episode_info = {
                "episode_id": episode_id,
                "env_id": env_id,
                "opponent": opponent_name,
                "model_player_id": agent_idx,
                "model_reward": rewards[agent_idx],
                "opponent_reward": rewards[1-agent_idx],
                "episode_length": step_count+1,
                "info": info,
                "set": "train" if env_id in self.env_ids else "eval"
            }

            # Incrementally write eval info to CSV file
            self._write_eval_episode_to_detailed_eval_csv(self.iteration, eval_episode_info)

            return eval_episode_info

        wins, count = 0, 0
        rewards = []

        # Parallel collection
        all_env_ids = self.env_ids if env_ids is None else self.env_ids+env_ids
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            # iterate over eval env ids and train env_ids
            for env_id in all_env_ids:
                for i in range(num_episodes):
                    futures.append(executor.submit(run_episode, env_id, i))

            progress_bar = tqdm(
                total=len(futures),
                desc="Evaluating Model",
                postfix={"win_rate": "0%", "avg_reward": "0"}
            )

            for future in concurrent.futures.as_completed(futures):
                # Lock the eval tracker dict
                eval_info = future.result()

                with self.token_metrics_lock:
                    wins += int(eval_info["model_reward"] > eval_info["opponent_reward"])
                    count += 1
                    rewards.append(eval_info["model_reward"])

                    # update progress bar
                    progress_bar.set_postfix(win_rate=f"{wins/count:.2f}%", avg_reward=f"{np.mean(rewards):.2f}")
                    progress_bar.update(1)

                    # update eval tracker dict
                    eval_tracker[eval_info["set"]]["num_episodes"] += 1
                    eval_tracker[eval_info["set"]]["model_rewards"].append(eval_info["model_reward"])
                    eval_tracker[eval_info["set"]]["model_outcomes"].append(int(eval_info["model_reward"] > eval_info["opponent_reward"]))
                    eval_tracker[eval_info["set"]]["opponent_rewards"].append(eval_info["opponent_reward"])
                    eval_tracker[eval_info["set"]]["episode_lengths"].append(eval_info["episode_length"])

            progress_bar.close()

        # log accumulated eval results
        accumulated_eval_dict = {}
        for set_name in eval_tracker.keys():
            if set_name not in accumulated_eval_dict:
                accumulated_eval_dict[set_name] = {}
            set_env_ids = self.env_ids if set_name == "train" else env_ids
            accumulated_eval_dict[set_name]["env_ids"] = ",".join(set_env_ids)
            accumulated_eval_dict[set_name]["num_episodes"] = eval_tracker[set_name]["num_episodes"]
            accumulated_eval_dict[set_name]["avg_model_reward"] = np.mean(eval_tracker[set_name]["model_rewards"])
            
            wins = np.sum(eval_tracker[set_name]["model_outcomes"]) 
            games = len(eval_tracker[set_name]["model_outcomes"])
            accumulated_eval_dict[set_name]["avg_model_win_rate"] = wins/games if games!=0 else 0
            accumulated_eval_dict[set_name]["opponent"] = opponent_name
            accumulated_eval_dict[set_name]["avg_opponent_reward"] = np.mean(eval_tracker[set_name]["opponent_rewards"])
            accumulated_eval_dict[set_name]["avg_episode_length"] = np.mean(eval_tracker[set_name]["episode_lengths"])
        
        # Write the summary of this evaluation run
        self._write_eval_summary_to_eval_csv(iteration=self.iteration, eval_summary_dict=accumulated_eval_dict)

        # Return the summary metrics
        return accumulated_eval_dict