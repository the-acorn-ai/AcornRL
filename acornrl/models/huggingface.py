# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from typing import Dict, Any, Optional, Tuple, List, Union

# class HFModel(nn.Module):
#     """
#     Wrapper for Hugging Face models to be used in RL training.
#     """
#     def __init__(
#         self,
#         model_name: str,
#         device: Optional[str] = None,
#         max_length: int = 2048,
#         **kwargs
#     ):
#         """
#         Initialize a Hugging Face model.
        
#         Args:
#             model_name: Name of the Hugging Face model to load
#             device: Device to load the model on (default: use CUDA if available)
#             max_length: Maximum sequence length for generation
#             **kwargs: Additional parameters to pass to the model
#         """
#         super().__init__()
#         self.model_name = model_name
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.max_length = max_length
        
#         # Load model and tokenizer
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name, 
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             **kwargs
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
#         # Set padding token if not set
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         # Move model to device
#         self.model.to(self.device)
        
#     def forward(
#         self, 
#         input_ids: torch.Tensor, 
#         attention_mask: Optional[torch.Tensor] = None,
#         **kwargs
#     ) -> torch.Tensor:
#         """
#         Forward pass through the model.
        
#         Args:
#             input_ids: Input token IDs
#             attention_mask: Attention mask
#             **kwargs: Additional parameters to pass to the model
            
#         Returns:
#             Model outputs
#         """
#         return self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             **kwargs
#         )
    
#     def generate_text(
#         self, 
#         prompt: str, 
#         max_new_tokens: int = 256,
#         temperature: float = 0.7,  # Lower temperature for more stable outputs
#         top_p: float = 0.9,
#         do_sample: bool = True,
#         **kwargs
#     ) -> str:
#         """
#         Generate text using the model.
        
#         Args:
#             prompt: Input prompt
#             max_new_tokens: Maximum number of tokens to generate
#             temperature: Sampling temperature
#             top_p: Top-p sampling parameter
#             do_sample: Whether to use sampling
#             **kwargs: Additional parameters to pass to the model's generate method
            
#         Returns:
#             Generated text
#         """
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
#         try:
#             # Generate output with error handling
#             with torch.no_grad():  # Add no_grad here for inference
#                 outputs = self.model.generate(
#                     inputs.input_ids,
#                     attention_mask=inputs.attention_mask,
#                     max_new_tokens=max_new_tokens,
#                     temperature=temperature,
#                     top_p=top_p,
#                     do_sample=do_sample,
#                     pad_token_id=self.tokenizer.pad_token_id,
#                     **kwargs
#                 )
            
#             # Decode the generated output
#             generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
#             # Return only the newly generated text (not the prompt)
#             prompt_decoded = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
#             if generated_text.startswith(prompt_decoded):
#                 return generated_text[len(prompt_decoded):]
#             else:
#                 # Fallback if substring matching fails
#                 return generated_text
                
#         except Exception as e:
#             print(f"Error in text generation: {e}")
#             # Return a safe fallback if generation fails
#             return "I cannot generate a response at this time."
    
#     def compute_logprobs(
#         self, 
#         input_text: str, 
#         output_text: str
#     ) -> torch.Tensor:
#         """
#         Compute log probabilities of output_text given input_text.
        
#         Args:
#             input_text: Context/input text
#             output_text: Text to compute log probabilities for
            
#         Returns:
#             Log probabilities of the output tokens
#         """
#         # Tokenize the combined input and output
#         full_text = input_text + output_text
#         tokens = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
#         # Get the number of tokens in the input
#         input_tokens = self.tokenizer(input_text, return_tensors="pt")
#         input_len = input_tokens.input_ids.shape[1]
        
#         # Forward pass through the model
#         # with torch.no_grad():  # Use no_grad for inference
#         outputs = self.model(
#             input_ids=tokens.input_ids,
#             attention_mask=tokens.attention_mask
#         )
            
#         # Get the logits
#         logits = outputs.logits
        
#         # Compute log probabilities for the output tokens
#         log_probs = []
        
#         # Ensure we don't go out of bounds
#         max_idx = min(len(tokens.input_ids[0]) - 1, logits.shape[1] - 1)
        
#         for i in range(input_len - 1, max_idx):
#             next_token_id = tokens.input_ids[0, i + 1].item()
#             token_logits = logits[0, i, :]
            
#             # Use log_softmax for numerical stability
#             log_probs_all = torch.log_softmax(token_logits, dim=0)
#             token_log_prob = log_probs_all[next_token_id]
            
#             log_probs.append(token_log_prob)
        
#         # Handle empty log_probs case
#         if not log_probs:
#             # Return a small constant tensor if no tokens were processed
#             return torch.tensor([-1.0], device=self.device)
            
#         return torch.stack(log_probs)
    
#     def parameters(self):
#         """
#         Return the parameters of the model.
#         """
#         return self.model.parameters()

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, Tuple, List, Union
import re

class HFModel(nn.Module):
    """
    Wrapper for Hugging Face models to be used in RL training with structured generation.
    """
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 2048,
        standard_prompt: str = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format.",
        **kwargs
    ):
        """
        Initialize a Hugging Face model.
        
        Args:
            model_name: Name of the Hugging Face model to load
            device: Device to load the model on (default: use CUDA if available)
            max_length: Maximum sequence length for generation
            standard_prompt: Standard prompt to include in all interactions
            **kwargs: Additional parameters to pass to the model
        """
        super().__init__()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.standard_prompt = standard_prompt
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.model.to(self.device)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Model outputs
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def format_observation(self, observation: str) -> str:
        """
        Format the observation following the structured format.
        
        Args:
            observation: Raw observation from the environment
            
        Returns:
            Formatted observation string
        """
        # Standardize begin token
        begin_token = "<｜begin▁of▁sentence｜>"
        if not begin_token in observation:
            formatted_input = f"{begin_token}Please reason step by step, and put your final answer within [ ].\n{self.standard_prompt}\n<｜User｜>{observation}\n<｜Assistant｜><think>\n"
        else:
            # Observation already has formatting
            formatted_input = observation
            
        return formatted_input
    
    def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Generate text using the model with reasoning and answer separation.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional parameters to pass to the model's generate method
            
        Returns:
            Tuple[str, str]: reasoning and answer
        """
        # Format the input with structure
        formatted_input = self.format_observation(prompt)
        
        # Tokenize the input
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)
        
        try:
            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )
            
            # Decode the generated output
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the newly generated text
            prompt_decoded = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
            if full_output.startswith(prompt_decoded):
                generated_text = full_output[len(prompt_decoded):]
            else:
                generated_text = full_output
            
            # Extract reasoning and answer
            # If the model completed both reasoning and answer tags
            if "</think>" in generated_text and "<answer>" in generated_text and "</answer>" in generated_text:
                # Extract reasoning (between <think> and </think>)
                reasoning_match = re.search(r'(.*?)</think>', generated_text, re.DOTALL)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                
                # Extract answer (between <answer> and </answer>)
                answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
                answer = answer_match.group(1).strip() if answer_match else ""
            
            # If the model only completed the reasoning part
            elif "</think>" in generated_text:
                # Extract reasoning (between start and </think>)
                reasoning_match = re.search(r'(.*?)</think>', generated_text, re.DOTALL)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                
                # Look for any content after </think> that might be part of the answer
                after_think = generated_text.split("</think>")[-1].strip()
                if after_think.startswith("<answer>"):
                    answer_match = re.search(r'<answer>(.*)', after_think, re.DOTALL)
                    answer = answer_match.group(1).strip() if answer_match else ""
                else:
                    answer = after_think
            
            # Basic fallback - try to find bracketed content as the answer
            elif "[" in generated_text and "]" in generated_text:
                # Try to extract bracketed content as the answer
                answer_match = re.search(r'\[(.*?)\]', generated_text, re.DOTALL)
                answer = answer_match.group(1).strip() if answer_match else ""
                
                # Everything else is reasoning
                reasoning = generated_text.replace(f"[{answer}]", "").strip()
            
            # If we can't find a clear separation
            else:
                # Just split at the midpoint as a fallback
                mid_point = len(generated_text) // 2
                reasoning = generated_text[:mid_point].strip()
                answer = generated_text[mid_point:].strip()
            
            return reasoning, answer
            
        except Exception as e:
            print(f"Error in text generation: {e}")
            # Return a safe fallback
            return "I cannot reason properly at this time.", "I cannot generate a valid response."
    
    def prepare_for_training(
        self,
        observation: str,
        reasoning: str,
        action: str
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for training with structured format.
        
        Args:
            observation: Observation from environment
            reasoning: Reasoning process
            action: Final action
            
        Returns:
            Dict with input_ids, attention_mask, and labels
        """
        # Format the input
        formatted_input = self.format_observation(observation)
        
        # Format the full output with reasoning and action
        full_text = formatted_input + f"{reasoning}</think>\n<answer>{action}</answer>"
        
        # Tokenize the combined text
        tokenized = self.tokenizer(full_text, padding=False, truncation=True, max_length=self.max_length, return_tensors="pt")
        
        # Tokenize just the observation to determine its length
        tokenized_obs = self.tokenizer(formatted_input, padding=False, truncation=True, max_length=self.max_length, return_tensors="pt")
        
        # Create labels: -100 for observation tokens, actual IDs for prediction tokens
        labels = tokenized.input_ids.clone()
        obs_len = tokenized_obs.input_ids.shape[1]
        labels[:, :obs_len] = -100
        
        # Replace padding token IDs with -100
        padding_mask = (labels == self.tokenizer.pad_token_id)
        labels[padding_mask] = -100
        
        return {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "labels": labels
        }
    
    def compute_logprobs(
        self, 
        input_text: str, 
        output_text: str
    ) -> torch.Tensor:
        """
        Compute log probabilities of output_text given input_text.
        Uses structured format with reasoning and answer.
        
        Args:
            input_text: Context/input text
            output_text: Text to compute log probabilities for (should include reasoning and answer)
            
        Returns:
            Log probabilities of the output tokens
        """
        # Extract reasoning and action if output_text contains both
        reasoning = ""
        action = ""
        
        # Check if output_text has structured format
        if "</think>" in output_text and "<answer>" in output_text:
            # Extract reasoning and action from structured format
            reasoning_match = re.search(r'(.*?)</think>', output_text, re.DOTALL)
            reasoning = reasoning_match.group(1) if reasoning_match else ""
            
            action_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL) 
            action = action_match.group(1) if action_match else ""
        else:
            # If not structured, use the whole output as action
            action = output_text
        
        # Format input with structure
        formatted_input = self.format_observation(input_text)
        
        # Create full structured text
        full_text = formatted_input + f"{reasoning}</think>\n<answer>{action}</answer>"
        
        # Tokenize just the observation to find its length
        input_tokens = self.tokenizer(formatted_input, return_tensors="pt")
        input_len = input_tokens.input_ids.shape[1]
        
        # Tokenize the full text
        tokens = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        # Forward pass through the model
        outputs = self.model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask
        )
            
        # Get the logits
        logits = outputs.logits
        
        # Compute log probabilities for the output tokens (skip the input tokens)
        log_probs = []
        
        # Ensure we don't go out of bounds
        max_idx = min(len(tokens.input_ids[0]) - 1, logits.shape[1] - 1)
        
        for i in range(input_len - 1, max_idx):
            next_token_id = tokens.input_ids[0, i + 1].item()
            token_logits = logits[0, i, :]
            
            # Use log_softmax for numerical stability
            log_probs_all = torch.log_softmax(token_logits, dim=0)
            token_log_prob = log_probs_all[next_token_id]
            
            log_probs.append(token_log_prob)
        
        # Handle empty log_probs case
        if not log_probs:
            # Return a small constant tensor if no tokens were processed
            return torch.tensor([-1.0], device=self.device)
            
        return torch.stack(log_probs)
    
    def parameters(self):
        """
        Return the parameters of the model.
        """
        return self.model.parameters()