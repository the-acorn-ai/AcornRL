import torch
from typing import Dict, Any, List, Union, Tuple, Optional
import numpy as np
from acornrl.agents.base import Agent
from acornrl.models.huggingface import HFModel

class ActorAgent(Agent):
    """
    An agent that uses a structured language model to generate actions with reasoning.
    """
    
    def __init__(
        self,
        model: HFModel,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        gamma: float = 0.95,
        **kwargs
    ):
        """
        Initialize an ActorAgent.
        
        Args:
            model: HFModel instance
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
            top_p: Top-p sampling parameter
            gamma: Discount factor for rewards
            **kwargs: Additional parameters
        """
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gamma = gamma
        super().__init__(**kwargs)
    
    def __call__(self, observation: Union[str, Dict[str, Any]]) -> Tuple[str, Optional[str]]:
        """
        Process observation and return an action with reasoning.
        
        Args:
            observation: The environment observation (string or dict)
            
        Returns:
            Tuple[str, str]: Action and reasoning
        """
        # Handle dict observations by converting to string if needed
        if isinstance(observation, dict):
            if "text" in observation:
                observation = observation["text"]
            else:
                # If there's no clear text field, convert the dict to a formatted string
                observation = str(observation)
        
        # Generate action and reasoning using the model
        reasoning, action = self.model.generate_text(
            prompt=observation,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        
        return action, reasoning
    
    def compute_loss(
        self,
        batch: List[Dict[str, Any]],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute REINFORCE loss with structured outputs.
        
        Args:
            batch: List of experience dictionaries
            **kwargs: Additional parameters
            
        Returns:
            torch.Tensor: Loss value
        """
        # Ensure the model is in training mode
        self.model.train()
        
        # Process all samples to get log probs and rewards
        log_probs = []
        rewards = []
        
        # Group experiences by episodes to apply discounting correctly
        episodes = {}
        for sample in batch:
            episode_id = sample.get("episode_id", 0)  # Default to 0 if not specified
            if episode_id not in episodes:
                episodes[episode_id] = []
            episodes[episode_id].append(sample)
        
        # Process each episode
        for episode_id, samples in episodes.items():
            # Sort samples by step within each episode
            samples.sort(key=lambda x: x["step"])
            
            # Apply discount to rewards from final reward
            episode_length = samples[-1]["full_length"]
            final_reward = samples[0]["final_reward"]  # Assume same reward for each player
            
            for sample in samples:
                # Skip samples with empty observations or actions
                if not sample["observation"] or not sample["action"]:
                    continue
                    
                # Create structured output with reasoning if available
                reasoning = sample.get("reasoning", "")
                
                # Compute log probabilities for the action with reasoning
                action_log_probs = self.model.compute_logprobs(
                    sample["observation"], 
                    f"{reasoning}</think>\n<answer>{sample['action']}</answer>"
                )
                
                # Sum log probs to get action probability
                action_log_prob = action_log_probs.sum()
                
                # Apply discounting based on step in episode
                discounted_reward = final_reward * (self.gamma ** (episode_length - sample["step"]))
                
                log_probs.append(action_log_prob)
                rewards.append(discounted_reward)
        
        # Handle empty case
        if not log_probs:
            print("Warning: No valid samples for loss computation")
            # Return a zero tensor with grad attached
            return torch.tensor(0.0, requires_grad=True, device=self.model.device)
            
        # Convert to tensors
        log_probs_tensor = torch.stack(log_probs)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=self.model.device)
        
        # Normalize rewards for training stability
        if len(rewards) > 1:
            rewards_mean = rewards_tensor.mean()
            rewards_std = rewards_tensor.std() + 1e-8  # Avoid division by zero
            rewards_tensor = (rewards_tensor - rewards_mean) / rewards_std
        
        # Compute REINFORCE loss: -log(p(a|s)) * reward
        loss = -(log_probs_tensor * rewards_tensor).mean()
        
        # Check for NaN or Inf values and handle them
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is {loss}. Using fallback loss.")
            return torch.tensor(0.1, requires_grad=True, device=self.model.device)
            
        return loss
    
    def prepare_batch_for_training(
        self,
        batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch of experiences for training.
        
        Args:
            batch: List of experience dictionaries
            
        Returns:
            Dict with input_ids, attention_mask, labels, and rewards
        """
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        rewards_list = []
        
        for sample in batch:
            observation = sample["observation"]
            reasoning = sample.get("reasoning", "")
            action = sample["action"]
            reward = sample["final_reward"] * (self.gamma ** (sample["full_length"] - sample["step"]))
            
            # Prepare data in structured format
            training_data = self.model.prepare_for_training(observation, reasoning, action)
            
            input_ids_list.append(training_data["input_ids"])
            attention_mask_list.append(training_data["attention_mask"])
            labels_list.append(training_data["labels"])
            rewards_list.append(torch.tensor([reward], device=self.model.device))
        
        # Combine into batch
        return {
            "input_ids": torch.cat(input_ids_list, dim=0),
            "attention_mask": torch.cat(attention_mask_list, dim=0),
            "labels": torch.cat(labels_list, dim=0),
            "rewards": torch.cat(rewards_list, dim=0)
        }