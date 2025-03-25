import os, json, time, argparse, torch
import numpy as np
import pandas as pd 
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

import torch.nn.functional as F
from transformers import Trainer
# local imports 
from acornrl.trainers import ReinforceTrainer
from acornrl.reward_shaping import reshape_rewards

import gc

def tokenize_and_mask(batch, tokenizer, gamma, max_seq_len=1024):
    # prompt_texts = batch["observation"]
    prompt_texts = batch["formatted_observation"]
    completion_texts = [
        f"{r}</think><answer>{a}</answer>"
        for r, a in zip(batch["reasoning"], batch["action"])
    ]
    full_texts = [p + "\n" + c for p, c in zip(prompt_texts, completion_texts)]

    tokenized_full = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=False
    )

    input_ids = tokenized_full["input_ids"]
    attention_masks = tokenized_full["attention_mask"]
    labels = []

    for idx, (prompt, completion) in enumerate(zip(prompt_texts, completion_texts)):
        # Tokenize prompt alone to find its length
        prompt_ids = tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        seq_len = sum(attention_masks[idx])
        label_seq = [-100] * len(input_ids[idx])
        for cpos in range(prompt_len, seq_len):
            label_seq[cpos] = input_ids[idx][cpos]
        labels.append(label_seq)

    tokenized_full["labels"] = labels

    # discounted reward: final_reward * (gamma^(full_length - step))
    # advantage = avg_reward + final_reward
    tokenized_full["reward"] = np.array(batch["final_reward"]) * gamma ** (
        np.array(batch["full_length"]) - np.array(batch["step"])
    )

    return tokenized_full

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Full fine-tuning on collected traces.")
    
    parser.add_argument("--gamma", type=float, required=False, default=0.99, help="Gamma time discounting of rewards")
    parser.add_argument("--max-seq-len", type=int, required=True, help="Maximum sequence length")

    parser.add_argument("--output-dir", type=str, required=True, help="Base directory")
    parser.add_argument("--iter", type=int, required=True, help="Iteration number")
    
    parser.add_argument("--local_rank", type=int, default=0, help="(For deepspeed) local rank.")

    parser.add_argument("--normalize-rewards", action="store_true", help="Whether the reward should be normalized")
    parser.add_argument("--reward-transformations", nargs="+", required=False, default=None, help="List of reward transformations")

    args = parser.parse_args()

    # Set memory efficiency options
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Define and build paths
    model_output_path = os.path.join(args.output_dir, "checkpoints", f"{args.iter}", "model")
    os.makedirs(model_output_path, exist_ok=True)


    # Determine whether to load from previous checkpoint
    first_training = args.iter == 1
    prev_model_path = os.path.join(args.output_dir, "checkpoints", f"{args.iter-1}", "model")
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" if first_training else prev_model_path


    # load the data
    training_data_path = os.path.join(args.output_dir, "data", f"train_{args.iter}.csv")
    df = pd.read_csv(training_data_path)
    print(len(df))
    # exit()

    episodes_with_reward1 = df.loc[df["final_reward"] == 1, "episode_id"].unique()
    df_filtered = df[df["episode_id"].isin(episodes_with_reward1)]

    # 2) Also keep only the rows where reasoning is not empty and final_reward != 0
    df = df_filtered[(df_filtered["reasoning"] != "") & (df_filtered["final_reward"] != 0)]

    # 3) Check if anything is left
    if len(df) == 0:
        print("[Training] 0 data points found. Nothing to train on.")
        return
    print(f"[Training] Found {len(df)} data points.")

    dataset = Dataset.from_pandas(df)


    # Set up the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _map_fn(batch):
        # Reduce sequence length to save memory
        return tokenize_and_mask(batch, tokenizer, gamma=args.gamma, max_seq_len=args.max_seq_len)

    dataset = dataset.map(_map_fn, batched=True)
    keep_cols = ["input_ids", "attention_mask", "labels", "reward"]
    remove_cols = set(dataset.column_names) - set(keep_cols)
    dataset = dataset.remove_columns(remove_cols)


    print(dataset)
    print(len(dataset))
    # exit()


    # Load the model
    print(f"[Training] Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_output_path,
        save_strategy="no",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # Increased for full fine-tuning
        learning_rate=1e-5,             # Lower learning rate for full fine-tuning
        num_train_epochs=3,
        fp16=False,
        bf16=True,
        logging_dir="./logs",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",            # Using PyTorch's implementation to avoid deprecation warning
        deepspeed="./acornrl/deepspeed_configs/ds_config_fp16.json",
        # Memory optimizations
        per_device_eval_batch_size=1,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        logging_steps=10,               # More frequent logging
        # save_strategy="epoch"
    )

    trainer = ReinforceTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    # Train
    trainer.train()
    model = trainer.model 
    model.save_pretrained(model_output_path, safe_serialization=True)
    tokenizer.save_pretrained(model_output_path)
   
    # Clean up
    model = None
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[Trained Model] Model saved at: {model_output_path}")

if __name__ == "__main__":
    main()