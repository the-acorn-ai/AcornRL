""" The csv manager should take care of all necessary storing and loading of data """
import os, csv, time
import threading, queue

from typing import List, Dict, Tuple, Any 


class CSVManagerData:
    def __init__(self, output_dir: str, iteration: int, episode_type: str):
        # Construct the CSV file name
        self.file_name = os.path.join(output_dir, "data", f"{episode_type}_{iteration}.csv")
        self.file_name_info = os.path.join(output_dir, "logging", f"{episode_type}_{iteration}_info.csv")
        self.episode_type = episode_type 

        # This queue holds items that need to be written to the CSV
        self.file_queue = queue.Queue()
        self.episode_info_file_queue = queue.Queue()

        # Stop event to tell the writer thread to stop after finishing the queue
        self._stop_event = threading.Event()
        self._writer_thread = None
        self._writer_thread_info = None

        self._create_csv_files()


    def _create_csv_files(self):
        # Create file if necessary & write header if it doesn't exist
        if not os.path.exists(self.file_name):
            with open(self.file_name, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode_id", "env_id", "model_name", "player_id", "observation", 
                    "formatted_observation", "reasoning", "action", "step", "full_length", 
                    "final_reward"
                ])

        if not os.path.exists(self.file_name_info):
            with open(self.file_name_info, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if self.episode_type == "train":
                    writer.writerow([
                        "episode_id", "env_id", "model_name", "t0", "t1", "t_delta", "num_char_formatted_obs",
                        "num_char_reasoning", "num_char_answer", "num_turns", "completion_status", "draw"
                    ])
                elif self.episode_type == "eval":
                    writer.writerow([
                        "env_type", "episode_id", "env_id", "model_name", "opponent_name", "t0", "t1", "t_delta",
                        "num_turns","model_reward", "opponent_reward", "completion_status", "model_outcome"
                    ])


    def add_episode(self, episode_data):
        """
        Add a list/row (episode_data) to the queue for writing to CSV.
        'episode_data' should be a sequence matching the CSV columns.
        """
        self.file_queue.put(episode_data)

    def __enter__(self):
        """ Start the CSV writer thread upon entering the context """
        # start the main write thread
        self._writer_thread = threading.Thread(target=self._write_to_csv, daemon=True)
        self._writer_thread.start()

        # start the info write thread
        self._writer_thread_info = threading.Thread(target=self._write_to_csv_info, daemon=True)
        self._writer_thread_info.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Signal the writer thread to stop and wait for it to finish """
        self._stop_event.set()
        self._writer_thread.join()
        self._writer_thread_info.join()

    def _write_to_csv(self):
        """
        Continuously pop items from the queue and write them to the CSV file 
        until the queue is empty and we have been signaled to stop.
        """
        # Keep writing while not (stop_event is set AND queue is empty)
        while not (self._stop_event.is_set() and self.file_queue.empty()):
            try:
                # Block for a short time, so we don't busy-wait
                data = self.file_queue.get(timeout=1)
                with open(self.file_name, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
            except queue.Empty:
                # If we time out, just check the loop condition again
                pass


    def add_episode_information(self, eps_info):
        self.episode_info_file_queue.put(eps_info)

    def _write_to_csv_info(self):
        # Keep writing while not (stop_event is set AND queue is empty)
        while not (self._stop_event.is_set() and self.episode_info_file_queue.empty()):
            try:
                # Block for a short time, so we don't busy-wait
                data = self.episode_info_file_queue.get(timeout=1)
                with open(self.file_name_info, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
            except queue.Empty:
                # If we time out, just check the loop condition again
                pass

    

