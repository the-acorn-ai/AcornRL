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
    parser.add_argument("--vllm-max-num-seq", type=int, required=False, default=64, help="The number of seq being generated in parallel on vllm")
   
   
    # run eval with already loaded vllm model
    # parser.add_argument("--run-eval", type=bool, required=False, default=False, help="Whether to run evals after collecting data.")
    parser.add_argument("--run-eval", action="store_true", help="Run evaluations after collecting data.")
    parser.add_argument("--eval-env-ids", nargs="+", required=False, default=["SpellingBee-v0", "ConnectFour-v0"], help="List of environment IDs for evaluation")
    parser.add_argument("--eval-episodes", type=int, required=False, default=10, help="Number of evaluation episodes")
    parser.add_argument("--opponent-name", type=str, required=False, default="google/gemini-2.0-flash-001", help="The eval opponent used.")


    args = parser.parse_args()

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
        max_new_tokens=args.max_seq_len, max_workers=args.max_workers, vllm_max_num_seq=args.vllm_max_num_seq
    ) as collector:        # check if eval
        if args.run_eval:
            print(f"[Evaluation] running {args.eval_episodes} of evaluation on {args.eval_env_ids}")
            collector.evaluate(env_ids=args.eval_env_ids, iteration=args.iter, num_episodes=args.eval_episodes, opponent_name=args.opponent_name)

        # collect training data
        print(f"[Data Collection] loading model {args.checkpoint}")
        collector.collect(num_episodes=args.episodes, iteration=args.iter)



if __name__ == "__main__":
    main()