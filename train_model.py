import os, json, time, argparse, torch
import numpy as np
import pandas as pd 
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

import torch.nn.functional as F
from transformers import Trainer
# local imports 
from acornrl.trainers import ReinforceTrainer, SPAGTrainer, SPAGTrainingArguments, SFTTrainer
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
    tokenized_full["reward"] = batch["final_reward"] 
    # np.array(batch["final_reward"]) * gamma ** (
    #     np.array(batch["full_length"]) - np.array(batch["step"])
    # )

    return tokenized_full

def spag_collator(batch, tokenizer, gamma, max_seq_len=1024):
    results = tokenize_and_mask(batch, tokenizer, gamma, max_seq_len)
    sft_mask = [0. for item in results['input_ids']]
    weights = [1. for item in results['input_ids']]
    results['sft_mask'] = torch.Tensor(sft_mask).float()
    results['weights'] = torch.Tensor(weights).float()
    return results

def filter_by_length(batch, tokenizer, max_seq_len=1024):
    # prompt_texts = batch["observation"]
    prompt_texts = batch["formatted_observation"]
    completion_texts = [
        f"{r}</think><answer>{a}</answer>"
        for r, a in zip(batch["reasoning"], batch["action"])
    ]
    full_texts = [p + "\n" + c for p, c in zip(prompt_texts, completion_texts)]

    tokenized_lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in full_texts]
    return [True if length <= max_seq_len else False for length in tokenized_lengths]

def get_trainer_and_args(train_method, model, base_args, dataset, tokenizer, **kwargs):
    """Factory function to create the appropriate trainer and args based on method."""
    if train_method == "reinforce":
        train_args = TrainingArguments(**base_args)
        return ReinforceTrainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            tokenizer=tokenizer
        ), train_args
    elif train_method == "sft":
        sft_args = SPAGTrainingArguments(
            lm_sft_coeff=kwargs.get('lm_sft_coeff', 0.1),
            lm_kl_coeff=kwargs.get('lm_kl_coeff', 0.01),
            clip_range=kwargs.get('clip_range', 0.2),
            **base_args
        )
        return SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=dataset,
            tokenizer=tokenizer
        ), sft_args
    elif train_method == "spag":
        spag_args = SPAGTrainingArguments(
            lm_sft_coeff=kwargs.get('lm_sft_coeff', 0.1),
            lm_kl_coeff=kwargs.get('lm_kl_coeff', 0.01),
            clip_range=kwargs.get('clip_range', 0.2),
            **base_args
        )
        return SPAGTrainer(
            model=model,
            args=spag_args,
            train_dataset=dataset,
            tokenizer=tokenizer
        ), spag_args
    else:
        raise ValueError(f"Unknown training method: {train_method}")

def get_data_processor(train_method, tokenizer, gamma, max_seq_len):
    """Returns the appropriate data processing function based on method."""
    if train_method in ["spag", "sft"]:
        return lambda batch: spag_collator(batch, tokenizer, gamma, max_seq_len)
    else:
        return lambda batch: tokenize_and_mask(batch, tokenizer, gamma, max_seq_len)

def get_keep_columns(train_method):
    """Returns the columns to keep based on training method."""
    if train_method in ["spag", "sft"]:
        return ["input_ids", "attention_mask", "labels", "reward", "sft_mask", "weights"]
    else:
        return ["input_ids", "attention_mask", "labels", "reward"]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Full fine-tuning on collected traces.")
    
    parser.add_argument("--model-path", type=str, help="Model path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--gamma", type=float, required=False, default=0.99, help="Gamma time discounting of rewards")
    parser.add_argument("--max-seq-len", type=int, required=True, help="Maximum sequence length")

    parser.add_argument("--output-dir", type=str, required=True, help="Base directory")
    parser.add_argument("--iter", type=int, required=True, help="Iteration number")
    
    parser.add_argument("--local_rank", type=int, default=0, help="(For deepspeed) local rank.")

    parser.add_argument("--normalize-rewards", action="store_true", help="Whether the reward should be normalized")
    parser.add_argument("--reward-transformations", nargs="+", required=False, default=None, help="List of reward transformations")
    parser.add_argument("--train-method", type=str, default="reinforce", help="Training method", choices=["reinforce", "spag", "sft"])
    
    parser.add_argument("--lm-sft-coeff", type=float, default=0.1, help="Weight for supervised fine-tuning loss")
    parser.add_argument("--lm-kl-coeff", type=float, default=0.01, help="Weight for KL divergence penalty")
    parser.add_argument("--clip-range", type=float, default=0.2, help="Clip range for importance ratio")
    
    # Add wandb logging arguments
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="full-training", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name (defaults to timestamp if not provided)")

    args = parser.parse_args()

    # Set memory efficiency options
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Define and build paths
    model_output_path = os.path.join(args.output_dir, "checkpoints", f"{args.iter}", "model")
    os.makedirs(model_output_path, exist_ok=True)

    # Determine whether to load from previous checkpoint
    first_training = args.iter == 1
    prev_model_path = os.path.join(args.output_dir, "checkpoints", f"{args.iter-1}", "model")
    model_path = args.model_path if first_training else prev_model_path

    # load the data
    training_data_path = os.path.join(args.output_dir, "data", f"train_{args.iter}.csv")
    df = pd.read_csv(training_data_path)
    print(len(df))

    episodes_with_reward1 = df.loc[df["final_reward"] == 1, "episode_id"].unique()
    df_filtered = df[df["episode_id"].isin(episodes_with_reward1)]

    # 2) Also keep only the rows where reasoning is not empty and final_reward != 0
    df = df_filtered[(df_filtered["reasoning"] != "") & (df_filtered["final_reward"] != 0)]

    # 3) Check if anything is left
    if len(df) == 0:
        print("[Training] 0 data points found. Nothing to train on.")
        return
    print(f"[Training] Found {len(df)} data points.")

    # Apply reward transformations if specified
    if args.reward_transformations or args.normalize_rewards:
        data_list = df.to_dict('records')
        data_list = reshape_rewards(
            data_list=data_list, 
            transformations=args.reward_transformations,
            normalize=args.normalize_rewards
        )
        df = pd.DataFrame(data_list)

    dataset = Dataset.from_pandas(df)

    # Set up the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Filter dataset by length using the filter_by_length function
    print(f"[Training] Initial dataset size: {len(dataset)}")
    dataset = dataset.filter(lambda batch: filter_by_length(batch, tokenizer, args.max_seq_len), batched=True)
    print(f"[Training] After length filtering: {len(dataset)} examples remaining")
    
    # Get the appropriate data processor based on method
    data_processor = get_data_processor(args.train_method, tokenizer, args.gamma, args.max_seq_len)
    dataset = dataset.map(data_processor, batched=True)
    
    # Get columns to keep based on method
    keep_cols = get_keep_columns(args.train_method)
    remove_cols = set(dataset.column_names) - set(keep_cols)
    dataset = dataset.remove_columns(remove_cols)

    print(dataset)
    print(len(dataset))

    # Load the model
    print(f"[Training] Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # 6.1) Add reference model if SPAG
    if args.train_method == "spag":
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        if hasattr(ref_model, "ref_model"):
            del ref_model.ref_model
        for param in ref_model.parameters():
            param.requires_grad = False

        model.ref_model = ref_model

    # Configure wandb if enabled
    if args.use_wandb:
        import wandb
        
        # Set up wandb run name if not provided
        run_name = args.wandb_name if args.wandb_name else f"full-training-iter-{args.iter}-{time.strftime('%Y%m%d-%H%M%S')}"
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            name=run_name
        )

    # Training arguments
    training_args_dict = {
        "output_dir": model_output_path,
        "save_strategy": "no",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,  # Increased for full fine-tuning
        "learning_rate": 1e-5,             # Lower learning rate for full fine-tuning
        "num_train_epochs": 3,
        "fp16": False,
        "bf16": True,
        "logging_dir": "./logs",
        "report_to": "wandb" if args.use_wandb else "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",            # Using PyTorch's implementation to avoid deprecation warning
        "deepspeed": "./acornrl/deepspeed_configs/ds_config_fp16.json",
        # Memory optimizations
        "per_device_eval_batch_size": 1,
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 0,
        "logging_steps": 10,               # More frequent logging
    }

    # Get the appropriate trainer and args
    trainer, _ = get_trainer_and_args(
        args.train_method, 
        model, 
        training_args_dict, 
        dataset, 
        tokenizer,
        lm_sft_coeff=args.lm_sft_coeff,
        lm_kl_coeff=args.lm_kl_coeff,
        clip_range=args.clip_range
    )

    # Train
    trainer.train()
    model = trainer.model
    # Clean up reference model if it exists
    if hasattr(model, "ref_model") and model.ref_model is not None:
        model.ref_model = None
        torch.cuda.empty_cache()
        gc.collect()

    model.save_pretrained(model_output_path, safe_serialization=True)
    tokenizer.save_pretrained(model_output_path)
   
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()

    # Clean up
    model = None
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[Trained Model] Model saved at: {model_output_path}")

if __name__ == "__main__":
    main()