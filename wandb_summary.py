import os, wandb, argparse
import pandas as pd
from tabulate import tabulate


def load_csv(filepath):
    """Loads a CSV file and returns a pandas DataFrame or None if not found."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"[Warning] File not found: {filepath}")
        return None


def calculate_training_stats(df_train):
    """Calculates training metrics from the CSV DataFrame."""
    if df_train is None or df_train.empty:
        return {}

    metrics = {
        "train-overall/num_episodes": len(df_train),
        "train-overall/avg_eps_duration": df_train["t_delta"].mean(),
        "train-overall/avg_num_turns": df_train["num_turns"].mean(),
        "train-overall/avg_len_obs": (
            (df_train["num_char_formatted_obs"] * df_train["num_turns"]).sum()
            / df_train["num_turns"].sum()
        ),
        "train-overall/avg_len_reasoning": df_train["num_char_reasoning"].mean(),
        "train-overall/avg_len_answer": df_train["num_char_answer"].mean(),
        "train-overall/invalid_move_rate": (
            df_train["completion_status"].eq("invalid").sum() / len(df_train)
        ),
        "train-overall/draw_rate": df_train["draw"].mean(),
    }

    for env_id in df_train["env_id"].unique():
        env_df = df_train[df_train["env_id"] == env_id]
        prefix = f"train-{env_id}"
        metrics.update({
            f"{prefix}/num_episodes": len(env_df),
            f"{prefix}/avg_eps_duration": env_df["t_delta"].mean(),
            f"{prefix}/avg_num_turns": env_df["num_turns"].mean(),
            f"{prefix}/avg_len_obs": (
                (env_df["num_char_formatted_obs"] * env_df["num_turns"]).sum()
                / env_df["num_turns"].sum()
            ),
            f"{prefix}/avg_len_reasoning": env_df["num_char_reasoning"].mean(),
            f"{prefix}/avg_len_answer": env_df["num_char_answer"].mean(),
            f"{prefix}/invalid_move_rate": (
                env_df["completion_status"].eq("invalid").sum() / len(env_df)
            ),
            f"{prefix}/draw_rate": env_df["draw"].mean(),
        })

    return metrics


def calculate_evaluation_stats(df_eval):
    """Calculates evaluation metrics from the CSV DataFrame."""
    if df_eval is None or df_eval.empty:
        return {}

    def eval_metrics_for(df, prefix):
        outcomes = df["model_outcome"].value_counts(normalize=True).to_dict()
        return {
            f"{prefix}/num_episodes": len(df),
            f"{prefix}/avg_eps_duration": df["t_delta"].mean(),
            f"{prefix}/avg_num_turns": df["num_turns"].mean(),
            f"{prefix}/avg_model_reward": df["model_reward"].mean(),
            f"{prefix}/avg_opponent_reward": df["opponent_reward"].mean(),
            f"{prefix}/win_rate": outcomes.get("win", 0),
            f"{prefix}/draw_rate": outcomes.get("draw", 0),
            f"{prefix}/loss_rate": outcomes.get("loss", 0),
        }

    metrics = {}
    metrics.update(eval_metrics_for(df_eval, "eval/overall"))

    for env_type in ["train", "eval"]:
        type_df = df_eval[df_eval["env_type"] == env_type]
        if not type_df.empty:
            metrics.update(eval_metrics_for(type_df, f"eval-{env_type}"))

    for env_id in df_eval["env_id"].unique():
        env_df = df_eval[df_eval["env_id"] == env_id]
        metrics.update(eval_metrics_for(env_df, f"eval-{env_id}"))

    return metrics


def pretty_print_table(metrics_dict, title):
    if not metrics_dict:
        print(f"\n{title} - No Data Available")
        return

    table_data = [(key, f"{value:.4f}" if isinstance(value, float) else value)
                  for key, value in metrics_dict.items()]
    table = tabulate(table_data, headers=["Metric", "Value"], tablefmt="fancy_grid")
    print(f"\n{title}\n{table}")


def main():
    parser = argparse.ArgumentParser(description="W&B Summary Logger")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--wandb-project", type=str, default="SuperHumandSofty")
    parser.add_argument("--wandb-run-name", type=str, required=True)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_run_name,
               id=os.getenv("WANDB_RUN_ID"), resume="allow")

    df_train = load_csv(os.path.join(args.data_folder, "logging", f"train_{args.iteration}_info.csv"))
    df_eval = load_csv(os.path.join(args.data_folder, "logging", f"eval_{args.iteration}_info.csv"))

    train_metrics = calculate_training_stats(df_train)
    eval_metrics = calculate_evaluation_stats(df_eval)

    pretty_print_table(train_metrics, f"Training Metrics - Iteration {args.iteration}")
    pretty_print_table(eval_metrics, f"Evaluation Metrics - Iteration {args.iteration}")

    wandb.log({**train_metrics, **eval_metrics}, step=args.iteration)
    wandb.finish()


if __name__ == "__main__":
    main()
