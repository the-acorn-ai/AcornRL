import os, json, argparse 
from acornrl.collectors import VLLMCollector


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Collect RL data using a model")
    parser.add_argument("--checkpoint", type=str, required=True, help="The model used for data collection")
    parser.add_argument("--episodes", type=int, required=True, help="The number episodes to be collected")
    parser.add_argument("--max-seq-len", type=int, required=True, help="Maximum sequence length")
    parser.add_argument("--env-ids", nargs="+", required=True, help="List of environment IDs")
    parser.add_argument("--output-dir", type=str, required=True, help="Base directory for storing data")
    parser.add_argument("--iter", type=int, required=True, help="Iteration number")
    parser.add_argument("--temperature", type=float, required=False, default=0.7, help="Generation temperature")
    parser.add_argument("--top-p", type=float, required=False, default=0.9, help="Generation top-p")


    args = parser.parse_args()

    # Define storage paths
    data_folder = os.path.join(args.output_dir, "data")
    os.makedirs(data_folder, exist_ok=True)

    logging_folder = os.path.join(args.output_dir, "logging")
    os.makedirs(logging_folder, exist_ok=True)


    # Output file for this iteration
    data_file = os.path.join(data_folder, f"iter_{args.iter}.json")
    print(f"[Data Collection] Running iteration {args.iter} on environments: {args.env_ids}")

    # Initialize the collector
    collector = VLLMCollector(
        env_ids=args.env_ids, 
        tensor_parallel_size=1, 
        logging_dir=logging_folder,
        iteration=args.iter
    )

    print(f"[Data Collection] loading model {args.checkpoint}")
    # Collect data
    data_dict = collector.collect(
        checkpoint_paths=[args.checkpoint],
        num_episodes=args.episodes,
        max_new_tokens=args.max_seq_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    new_data = data_dict["data"]
    print(f"[Data Collection] Collected {len(new_data)} new transitions.")

    # Save collected data as a new JSON file
    with open(data_file, "w") as f:
        json.dump({"data": new_data}, f, indent=2)

    print(f"[Data Collection] Saved data to {data_file}")

if __name__ == "__main__":
    main()