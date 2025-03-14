import os, json, argparse, torch
import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, TaskType, PeftConfig, PeftModel

import torch.nn.functional as F

# local imports 
from acornrl.trainers import ReinforceTrainer



def tokenize_and_mask(batch, tokenizer, gamma, max_seq_len=1024):
    prompt_texts = batch["observation"]
    completion_texts = [
        f"{r}</think><answer>{a}</answer>"
        for r, a in zip(batch["reasoning"], batch["action"])
    ]
    full_texts = [p + "\n" + c for p, c in zip(prompt_texts, completion_texts)]

    tokenized_full = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len
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
    tokenized_full["reward"] = np.array(batch["final_reward"]) * gamma ** (
        np.array(batch["full_length"]) - np.array(batch["step"])
    )

    return tokenized_full





def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train LoRa model on colelcted traces.")
    
    # parser.add_argument("--checkpoint", type=str, required=True, help="The model used for data collection")
    parser.add_argument("--gamma", type=float, required=False, default=0.99, help="Gamma time discounting of rewards")
    parser.add_argument("--max-seq-len", type=int, required=True, help="Maximum sequence length")

    parser.add_argument("--output-dir", type=str, required=True, help="Base directory")
    parser.add_argument("--iter", type=int, required=True, help="Iteration number")
    
    parser.add_argument("--local_rank", type=int, default=0, help="(For deepspeed) local rank.")
    # parser.add_argument("--data-file", type=str, required=True, help=".json file of training data")

    args = parser.parse_args()


    # define and build paths
    lora_adapter_path = os.path.join(args.output_dir, "checkpoints", f"{args.iter}", "lora_adapter")
    os.makedirs(lora_adapter_path, exist_ok=True)

    final_model_path = os.path.join(args.output_dir, "checkpoints", f"{args.iter}", "model")
    os.makedirs(final_model_path, exist_ok=True)

    training_data_path = os.path.join(args.output_dir, "data", f"iter_{args.iter}.json")


    # check whether lora adapter already exists
    first_training = not (os.path.exists(lora_adapter_path) and len(os.listdir(lora_adapter_path))>0) 

    # 2) Check if there is data 
    if not os.path.exists(training_data_path):
        print("[Training] No data file found. Please run data_collection.py first.")
        return

    with open(training_data_path, "r") as f:
        data_json = json.load(f)


    data_list = data_json.get("data", [])
    if len(data_list) == 0:
        print("[Training] 0 data points found. Nothing to train on.")
        return


    print(f"[Training] Found {len(data_list)} data points.")


    # 3) Convert to Dataset
    dataset = Dataset.from_list(data_list)

    # 4) Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _map_fn(batch):
        # Reduce sequence length to save memory
        return tokenize_and_mask(batch, tokenizer, gamma=args.gamma, max_seq_len=args.max_seq_len)

    dataset = dataset.map(_map_fn, batched=True)
    keep_cols = ["input_ids", "attention_mask", "labels", "reward"]
    remove_cols = set(dataset.column_names) - set(keep_cols)
    dataset = dataset.remove_columns(remove_cols)

    # 5) Load the base model (without device_map='auto' for distributed training)
    base_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        torch_dtype=torch.float16,  # Force FP16 to save memory
    )

    # Enable gradient checkpointing for memory efficiency
    base_model.gradient_checkpointing_enable()

    # 6) Create and apply LoRA configuration
    print(f"[Training] Configuring LoRA adapter")
    lora_config = LoraConfig(
        r=16,                     # Rank dimension
        lora_alpha=32,            # Alpha parameter for LoRA scaling
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Prepare model fro training
    model = get_peft_model(base_model, lora_config)
    if not first_training:
        model.load_adapter(lora_adapter_path, adapter_name="default")
        model.set_adapter("default")


    # 7) Training arguments optimized for LoRA
    training_args = TrainingArguments(
        output_dir=lora_adapter_path,
        save_strategy="no",
        save_steps=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,        # Higher learning rate for LoRA fine-tuning
        num_train_epochs=5,
        fp16=False,
        bf16=True,
        logging_dir="./logs",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_hf", #"adamw_torch",
        deepspeed="./acornrl/deepspeed_configs/ds_config_fp16.json",
        # Memory optimizations
        per_device_eval_batch_size=1,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        logging_steps=50
    )

    trainer = ReinforceTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )


    # 8) Train
    trainer.train()

    # 9) Save the LoRA adapter model
    model.save_pretrained(lora_adapter_path)
    print(f"[Training] LoRA adapter saved to {lora_adapter_path}")

    # merge lora into main model 

    # 1) Read the LoRA config
    peft_config = PeftConfig.from_pretrained(lora_adapter_path)

    # 2) Load the base model 
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, torch_dtype=torch.float16, device_map="auto")

    # 3) Load the LoRA adapter
    model = PeftModel.from_pretrained(model, lora_adapter_path, device_map="auto")

    # 4) Merge the LoRA weights into the main model
    model = model.merge_and_unload()

    # 5. Save the merged model
    model.save_pretrained(final_model_path)

    # 6. Also save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"[Trained Model] Merged model saved at: {final_model_path}")

if __name__ == "__main__":
    main()