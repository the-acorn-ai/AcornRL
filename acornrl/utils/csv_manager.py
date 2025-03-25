""" The csv manager should take care of all necessary storing and loading of data """
import os, csv, time
import threading, queue

from typing import List, Dict, Tuple, Any 

class CSVManagerLogging:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.logging_dir = os.path.join(output_dir, "logging")


        # determine individual file paths
        self.details_csv_path = os.path.join(self.logging_dir, "details.csv")
        self.summary_csv_path = os.path.join(self.logging_dir, "summary.csv")

        self.eval_details_csv_path = os.path.join(self.logging_dir, "eval_details.csv")
        self.eval_summary_csv_path = os.path.join(self.logging_dir, "eval_summary.csv")


        self._initialize_csv_files()


    def _initialize_csv_files(self):
        # 1) details.csv: Logs data about each episode
        if not os.path.exists(self.details_csv_path):
            with open(self.details_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "iteration", "episode_id", "env_id", "checkpoint_path", "player_id", 
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

class CSVManagerData:
    def __init__(self, output_dir: str, iteration: int, episode_type: str):
        # Construct the CSV file name
        self.file_name = os.path.join(output_dir, "data", f"{episode_type}_{iteration}.csv")

        # Create file if necessary & write header if it doesn't exist
        if not os.path.exists(self.file_name):
            with open(self.file_name, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode_id", "env_id", "model_name", "player_id", "observation", 
                    "formatted_observation", "reasoning", "action", "step", "full_length", 
                    "final_reward"
                ])


        # This queue holds items that need to be written to the CSV
        self.file_queue = queue.Queue()

        # Stop event to tell the writer thread to stop after finishing the queue
        self._stop_event = threading.Event()
        self._writer_thread = None

    def add_episode(self, episode_data):
        """
        Add a list/row (episode_data) to the queue for writing to CSV.
        'episode_data' should be a sequence matching the CSV columns.
        """
        self.file_queue.put(episode_data)

    def __enter__(self):
        """
        Start the CSV writer thread upon entering the context.
        """
        self._writer_thread = threading.Thread(
            target=self._write_to_csv,
            daemon=True
        )
        self._writer_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Signal the writer thread to stop and wait for it to finish.
        """
        self._stop_event.set()
        self._writer_thread.join()

    def _write_to_csv(self):
        """
        Continuously pop items from the queue and write them to the CSV file 
        until the queue is empty and we have been signaled to stop.
        """
        with open(self.file_name, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Keep writing while not (stop_event is set AND queue is empty)
            while not (self._stop_event.is_set() and self.file_queue.empty()):
                try:
                    # Block for a short time, so we don't busy-wait
                    data = self.file_queue.get(timeout=0.1)
                    writer.writerow(data)
                except queue.Empty:
                    # If we time out, just check the loop condition again
                    pass
