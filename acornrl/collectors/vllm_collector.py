import os, csv, time, torch
import statistics, logging, threading 
import concurrent.futures 
import numpy as np 
from tqdm import tqdm 
from typing import Optional, List, Dict, Any


import textarena as ta
from acornrl.inference import VLLMServerManager, VLLMInferenceClient



# class VLLMCollector(Collector):
class VLLMCollector:
    def __init__(
        self,
        env_ids: List[str],
        logging_dir: str,
        iteration: int,
        checkpoint_paths: List[str],
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_workers: Optional[int] = None,
        tensor_parallel_size: int = 1,
        gpus: Optional[List[int]] = None,
        base_port: int = 8000,
        **kwargs
    ):
        self.env_ids = env_ids
        self.logging_dir = logging_dir
        self.iteration = iteration
        self.tensor_parallel_size = tensor_parallel_size
        self.base_port = base_port

        self.checkpoint_paths = checkpoint_paths

        # generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Use all GPUs if none are specified
        if gpus is None:
            self.gpus = list(range(torch.cuda.device_count()))
        else:
            self.gpus = gpus
            
        # Determine number of workers
        if max_workers is None:
            self.max_workers = len(self.gpus) // tensor_parallel_size
        else:
            self.max_workers = max_workers

        self.logger = logging.getLogger(__name__)
        self.server_manager = None 
        self.vllm_clients = []

        # Metrics tracking
        self.token_metrics_lock = threading.Lock()
        self.total_tokens = 0
        self.total_time = 0
        self.total_requests = 0
        self.completed_episodes = 0
        self.episode_tps_list = []

        # Prepare CSV file paths
        os.makedirs(self.logging_dir, exist_ok=True)
        self.details_csv_path = os.path.join(self.logging_dir, "details.csv")
        self.summary_csv_path = os.path.join(self.logging_dir, "summary.csv")

        self.eval_details_csv_path = os.path.join(self.logging_dir, "eval_details.csv")
        self.eval_summary_csv_path = os.path.join(self.logging_dir, "eval_summary.csv")
        
        # Create CSV headers if files are new
        self._create_csv_if_needed()


    def _create_csv_if_needed(self):
        """Create the CSV files if they don't exist, adding headers."""
        # 1) details.csv: logs data about each episode (or even each step)
        if not os.path.exists(self.details_csv_path):
            with open(self.details_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "iteration", "episode_id", "env_id", "checkpoint_path", "player_id", "step", 
                    "tokens", "generation_time", "avg_tokens_per_second", "final_reward", 
                    "episode_length", "episode_tokens", "episode_time", "tokens_per_second_episode"
                ])

        # 2) summary.csv: logs one line **per iteration** of data collection
        if not os.path.exists(self.summary_csv_path):
            with open(self.summary_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "iteration", "checkpoint_path", "num_episodes", "collection_time_sec", "episodes_per_second",
                    "total_tokens", "total_generation_time_sec", "global_tokens_per_second", "tokens_per_second_median",
                    "tokens_per_second_min", "tokens_per_second_max", "tokens_per_second_stddev", "total_requests", "timestamp_utc"
                ])


        # 3) eval_details.csv: logs data about each eval episode
        if not os.path.exists(self.eval_details_csv_path):
            with open(self.eval_details_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "iteration", "set", "env_id", "episode_id", "checkpoint_path", "model_player_id", 
                    "model_reward", "opponent", "opponent_reward", "episode_length", "info" 
                ])


        # 4) eval_summary.csv: logs eval summary by set (train vs eval)
        if not os.path.exists(self.eval_summary_csv_path):
            with open(self.eval_summary_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "iteration", "set", "set_env_ids", "num_episodes", "checkpoint_path", "avg_model_reward",
                    "avg_model_win_rate", "opponent", "avg_opponent_reward", "avg_episode_length"
                ])

                    

    def collect(self, num_episodes: int) -> Dict:
        # Reset metrics
        with self.token_metrics_lock:
            self.total_tokens = 0
            self.total_time = 0
            self.total_requests = 0
            self.completed_episodes = 0
            self.episode_tps_list = []

        start_time = time.time()

        # Prepare for parallel collection
        token_tracker = threading.local()
        collected_data = []


        def run_episode(episode_id: int) -> List[Dict[str, Any]]:
            # Thread-local metrics
            token_tracker.episode_tokens = 0
            token_tracker.episode_time = 0
            token_tracker.episode_requests = 0
            
            client_idx = episode_id % len(self.vllm_clients)
            vllm_client = self.vllm_clients[client_idx]
            checkpoint_path = self.checkpoint_paths[client_idx % len(self.checkpoint_paths)]

            # Create & wrap environment 
            env = ta.make(self.env_ids)
            env_id = env.env_id
            env = ta.wrappers.LLMObservationWrapper(env=env)

            env.reset(num_players=2)

            episode_data = []
            step_count = 0
            done = False 

            while not done:
                player_id, observation = env.get_observation()

                # Use vLLM for inference 
                result = vllm_client.generate_text(
                    prompt=observation,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                
                # Extract relevant info
                action = result["answer"]
                reasoning = result["reasoning"]
                completion_info = result.get("metrics", {})
                
                tokens_generated = completion_info.get("completion_tokens", 0)
                generation_time = completion_info.get("completion_time", 0)
                
                # Update thread-local counters
                token_tracker.episode_tokens += tokens_generated
                token_tracker.episode_time += generation_time
                token_tracker.episode_requests += 1

                # Record step data
                episode_data.append({
                    "env_id": env_id,
                    "checkpoint_path": checkpoint_path,
                    "player_id": player_id,
                    "observation": observation,
                    "reasoning": reasoning,
                    "action": action,
                    "step": step_count,
                    "tokens": tokens_generated,
                    "generation_time": generation_time,
                    "avg_tokens_per_second": tokens_generated / generation_time if generation_time > 0 else 0
                })

                done, info = env.step(action=action)
                step_count += 1

            # Episode done
            rewards = env.close()

            # Add rewards to episode data
            for step_data in episode_data:
                step_data["final_reward"] = rewards[step_data["player_id"]]
                step_data["full_length"] = len(episode_data)

            # Episode-level metrics
            episode_tps = token_tracker.episode_tokens / token_tracker.episode_time if token_tracker.episode_time > 0 else 0
            episode_metrics = {
                "episode_id": episode_id,
                "episode_tokens": token_tracker.episode_tokens,
                "episode_time": token_tracker.episode_time,
                "episode_requests": token_tracker.episode_requests,
                "tokens_per_second": episode_tps,
            }

            # Attach them to each step
            for step_data in episode_data:
                step_data["episode_metrics"] = episode_metrics

            # Update global metrics
            with self.token_metrics_lock:
                self.total_tokens += token_tracker.episode_tokens
                self.total_time += token_tracker.episode_time
                self.total_requests += token_tracker.episode_requests
                self.completed_episodes += 1
                self.episode_tps_list.append(episode_tps)

            # Write this episode to details.csv
            self._write_episode_to_details_csv(self.iteration, episode_id, episode_data)


            return episode_data

        # Parallel collection
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(run_episode, i) for i in range(num_episodes)]
            
            progress_bar = tqdm(
                total=num_episodes, 
                desc="Collecting episodes",
                postfix={"tokens_per_sec": "0.00", "episodes": f"0/{num_episodes}"}
            )
            
            for future in concurrent.futures.as_completed(futures):
                episode_data = future.result()
                collected_data.extend(episode_data)
                
                with self.token_metrics_lock:
                    current_tps = self.total_tokens / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                    progress_bar.set_postfix(
                        tokens_per_sec=f"{current_tps:.2f}", 
                        episodes=f"{self.completed_episodes}/{num_episodes}",
                    )
                progress_bar.update(1)
            progress_bar.close()

        collection_time = time.time() - start_time
        episodes_per_second = num_episodes / collection_time if collection_time > 0 else 0
        
        with self.token_metrics_lock:
            tokens_per_second = self.total_tokens / self.total_time if self.total_time > 0 else 0

            tps_median = statistics.median(self.episode_tps_list) if self.episode_tps_list else 0
            tps_min = min(self.episode_tps_list) if self.episode_tps_list else 0
            tps_max = max(self.episode_tps_list) if self.episode_tps_list else 0
            tps_stddev = statistics.stdev(self.episode_tps_list) if len(self.episode_tps_list) > 1 else 0

            # Log summary
            self.logger.info(f"== Token Generation Statistics ==")
            self.logger.info(f"  Total tokens: {self.total_tokens:,}")
            self.logger.info(f"  Total generation time: {self.total_time:.2f}s")
            self.logger.info(f"  Average tokens per second: {tokens_per_second:.2f}")
            self.logger.info(f"  Token throughput range: {tps_min:.2f} - {tps_max:.2f} tokens/s")
            self.logger.info(f"  Token throughput median: {tps_median:.2f} tokens/s")
            self.logger.info(f"  Total requests: {self.total_requests}")
        
        self.logger.info(f"== Collection Performance ==")
        self.logger.info(f"  Collected {num_episodes} episodes in {collection_time:.2f}s")
        self.logger.info(f"  Collection speed: {episodes_per_second:.2f} episodes/second")

        # **Write a summary row** for this iteration
        self._write_summary_csv(
            iter_id=self.iteration,
            checkpoint_path=self.checkpoint_paths[0] if self.checkpoint_paths else "unknown",
            num_episodes=num_episodes,
            collection_time=collection_time,
            episodes_per_second=episodes_per_second,
            tokens_per_second=tokens_per_second,
            tps_median=tps_median,
            tps_min=tps_min,
            tps_max=tps_max,
            tps_stddev=tps_stddev
        )

        # Prepare return dictionary
        metrics = {
            "token_metrics": {
                "total_tokens": self.total_tokens,
                "total_generation_time": self.total_time,
                "tokens_per_second": tokens_per_second,
                "tokens_per_second_median": tps_median,
                "tokens_per_second_min": tps_min,
                "tokens_per_second_max": tps_max,
                "tokens_per_second_stddev": tps_stddev,
                "total_requests": self.total_requests,
            },
            "collection_metrics": {
                "iteration": self.iteration,
                "checkpoint_paths": self.checkpoint_paths,
                "total_episodes": num_episodes,
                "completed_episodes": self.completed_episodes,
                "collection_time": collection_time,
                "episodes_per_second": episodes_per_second,
            },
            "data": collected_data
        }

        return metrics


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

            # write eval info
            self._write_eval_episode_to_detailed_eval_csv(self.iteration, eval_episode_info)


            # Write this episode to evals.csv
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
                # TODO lock the eval tracker dict
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
        

        # submit for writing
        self._write_eval_summary_to_eval_csv(iteration=self.iteration, eval_summary_dict=accumulated_eval_dict)




    def _setup_vllm_clients(self):
        # for now only handle single model case 
        if len(self.checkpoint_paths) != 1:
            raise NotImplementedError

        from acornrl.inference.vllm import VLLMServerManager
        
        self.server_manager = VLLMServerManager(
            model_path=self.checkpoint_paths[0],
            max_seq_len=self.max_new_tokens,
            gpus=self.gpus,
            tensor_parallel_size=self.tensor_parallel_size,
            base_port=self.base_port,
        )

        # Start vLLM servers
        self.logger.info("Starting vLLM servers for fast inference")
        self.server_manager.start_servers()
        
        # Initialize clients
        self.vllm_clients = []
        for i in range(self.server_manager.num_servers):
            client = self.server_manager.get_client(i)
            self.vllm_clients.append(client)
            
        self.logger.info(f"Successfully initialized {len(self.vllm_clients)} vLLM clients")

    def get_token_metrics(self) -> Dict[str, float]:
        """Get the current token generation metrics."""
        with self.token_metrics_lock:
            tokens_per_second = self.total_tokens / self.total_time if self.total_time > 0 else 0
            tps_median = statistics.median(self.episode_tps_list) if self.episode_tps_list else 0
            
            return {
                "total_tokens": self.total_tokens,
                "total_time": self.total_time,
                "tokens_per_second": tokens_per_second,
                "tokens_per_second_median": tps_median,
                "total_requests": self.total_requests,
                "completed_episodes": self.completed_episodes
            }
    
    def shutdown(self):
        """Shut down vLLM servers."""
        if self.server_manager:
            self.server_manager.stop_servers()
            self.vllm_clients = []


    def __enter__(self):
        """Context manager entry: Setup vLLM clients."""
        self._setup_vllm_clients()
        return self  # Return the instance for use in the `with` block

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit: Shut down vLLM clients."""
        self.shutdown()   


    def _write_episode_to_details_csv(self, iteration: int, episode_id: int, episode_data: List[Dict[str, Any]]):
        """Append detailed step-level records to details.csv."""
        if not episode_data:
            return

        with open(self.details_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for step_data in episode_data:
                checkpoint_path = step_data["checkpoint_path"]
                env_id = step_data["env_id"]
                player_id = step_data["player_id"]
                step = step_data["step"]
                tokens = step_data["tokens"]
                generation_time = step_data["generation_time"]
                avg_tps_step = step_data["avg_tokens_per_second"]
                final_reward = step_data["final_reward"]
                episode_length = step_data["full_length"]

                # Episode-level metrics stored in step_data["episode_metrics"]
                ep_tokens = step_data["episode_metrics"]["episode_tokens"]
                ep_time = step_data["episode_metrics"]["episode_time"]
                ep_tps = step_data["episode_metrics"]["tokens_per_second"]

                writer.writerow([
                    iteration,
                    episode_id,
                    env_id,
                    checkpoint_path,
                    player_id,
                    step,
                    tokens,
                    generation_time,
                    f"{avg_tps_step:.2f}",
                    final_reward,
                    episode_length,
                    ep_tokens,
                    f"{ep_time:.2f}",
                    f"{ep_tps:.2f}"
                ])


    def _write_episode_csv(
        self, iteration: int, episode_id: int, env_id: str, checkpoint_path: str, episode_tokens: float,
        episode_time: float, episode_tps: float, episode_length: int, final_rewards: Any
    ):
        """Write a single line for this entire episode."""
        # final_rewards could be multiple players; store as string if needed
        # e.g. final_rewards = [reward_0, reward_1] => convert to JSON or str
        import json
        rewards_str = json.dumps(final_rewards)

        with open(self.episodes_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                episode_id,
                env_id,
                checkpoint_path,
                f"{episode_tokens:.2f}",
                f"{episode_time:.2f}",
                f"{episode_tps:.2f}",
                episode_length,
                rewards_str
            ])

    def _write_eval_episode_to_detailed_eval_csv(self, iteration: int, eval_info: Dict[str, Any]):
        if not eval_info:
            return 

        with open(self.eval_details_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                eval_info.get("set"),
                eval_info.get("episode_id"),
                self.checkpoint_paths[0],
                eval_info.get("model_player_id"),
                eval_info.get("model_reward"),
                eval_info.get("opponent"),
                eval_info.get("opponent_reward"),
                eval_info.get("episode_length"),
                eval_info.get("info")
            ])

    def _write_eval_summary_to_eval_csv(self, iteration: int, eval_summary_dict: Dict[str, Dict[str, Any]]):
        if not eval_summary_dict:
            return 

        with open(self.eval_summary_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for eval_set in eval_summary_dict.keys():
                writer.writerow([
                    iteration,
                    eval_set,
                    eval_summary_dict[eval_set].get("env_ids"),
                    eval_summary_dict[eval_set].get("num_episodes"),
                    self.checkpoint_paths[0],
                    eval_summary_dict[eval_set].get("avg_model_reward"),
                    eval_summary_dict[eval_set].get("avg_model_win_rate"),
                    eval_summary_dict[eval_set].get("opponent"),
                    eval_summary_dict[eval_set].get("avg_opponent_reward"),
                    eval_summary_dict[eval_set].get("avg_episode_length")
                ])


    def _write_summary_csv(
        self,
        iter_id: int,
        checkpoint_path: str,
        num_episodes: int,
        collection_time: float,
        episodes_per_second: float,
        tokens_per_second: float,
        tps_median: float,
        tps_min: float,
        tps_max: float,
        tps_stddev: float
    ):
        """Append one line of summary stats to summary.csv."""
        timestamp_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        with open(self.summary_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                iter_id,
                checkpoint_path,
                num_episodes,
                f"{collection_time:.2f}",
                f"{episodes_per_second:.2f}",
                self.total_tokens,
                f"{self.total_time:.2f}",
                f"{tokens_per_second:.2f}",
                f"{tps_median:.2f}",
                f"{tps_min:.2f}",
                f"{tps_max:.2f}",
                f"{tps_stddev:.2f}",
                self.total_requests,
                timestamp_utc
            ])
