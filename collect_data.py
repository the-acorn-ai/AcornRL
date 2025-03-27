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
    parser.add_argument("--max-workers", type=int, required=False, default=512, help="The number of envs running in parallel")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU IDs to use")
    # run eval with already loaded vllm model
    # parser.add_argument("--run-eval", type=bool, required=False, default=False, help="Whether to run evals after collecting data.")
    parser.add_argument("--run-eval", action="store_true", help="Run evaluations after collecting data.")
    parser.add_argument("--eval-env-ids", nargs="+", required=False, default=["SpellingBee-v0", "ConnectFour-v0"], help="List of environment IDs for evaluation")
    parser.add_argument("--eval-episodes", type=int, required=False, default=10, help="Number of evaluation episodes")


    args = parser.parse_args()

    if args.gpus is not None:
        gpus = [int(gpu) for gpu in args.gpus.split(",")]
    else:
        gpus = None

    # Define storage paths
    data_folder = os.path.join(args.output_dir, "data")
    os.makedirs(data_folder, exist_ok=True)

    logging_folder = os.path.join(args.output_dir, "logging")
    os.makedirs(logging_folder, exist_ok=True)


    # Output file for this iteration
    # data_file = os.path.join(data_folder, f"iter_{args.iter}.json")
    print(f"[Data Collection] Running iteration {args.iter} on environments: {args.env_ids}")

    # Run the data collection and optionally evaluation 
    with VLLMCollector(
        env_ids=args.env_ids, checkpoint_path=args.checkpoint, output_dir=args.output_dir, 
        max_new_tokens=args.max_seq_len, max_workers=args.max_workers, gpus=gpus   
    ) as collector:
        # collect training data
        print(f"[Data Collection] loading model {args.checkpoint}")
        data_dict = collector.collect(num_episodes=args.episodes, iteration=args.iter)

        # # check if eval
        if args.run_eval:
            print(f"[Evaluation] running {args.eval_episodes} of evaluation on {args.eval_env_ids}")
            collector.evaluate(num_episodes=args.eval_episodes, iteration=args.iter)


if __name__ == "__main__":
    main()