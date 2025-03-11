import torch
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm
import logging
import time

from acornrl.agents.base import Agent
from acornrl.collectors.base import Collector

class StandardTrainer:
    """
    Standard trainer for reinforcement learning with language models.
    """
    
    def __init__(
        self,
        agent: Agent,
        collector: Collector,
        optimizer: torch.optim.Optimizer,
        log_interval: int = 1,
        checkpoint_interval: int = 5,
        eval_interval: int = 1,
        checkpoint_path: Optional[str] = None,
        device: str = None
    ):
        """
        Initialize the trainer.
        
        Args:
            agent: Agent to train
            collector: Data collector
            optimizer: Optimizer for agent's model
            log_interval: How often to log training stats (in epochs)
            checkpoint_interval: How often to save model checkpoints (in epochs)
            eval_interval: How often to run evaluation (in epochs)
            checkpoint_path: Where to save checkpoints
            device: Device to train on
        """
        self.agent = agent
        self.collector = collector
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Training metrics
        self.metrics = {
            "train_loss": [],
            "eval_reward": [],
            "epoch_time": []
        }
    
    def _run_training_step(self, episodes_per_epoch):
        """
        Run a single training step.
        
        Args:
            episodes_per_epoch: Number of episodes to collect per epoch
            
        Returns:
            float: Average loss for this step
        """
        # Collect the data
        self.logger.info(f"Collecting {episodes_per_epoch} episodes")
        data = self.collector.collect(agent=self.agent, num_episodes=episodes_per_epoch)
        
        # Track total loss
        total_loss = 0.0
        num_batches = 0
        
        # Process collected data
        # for batch in data:
        # for sample in data:


        # Zero gradients
        self.optimizer.zero_grad()
        
        # Calculate loss
        loss = self.agent.compute_loss(data)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients (optional, uncomment if needed)
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), max_norm=1.0)
        
        # Update parameters
        # self.optimizer.step()
        
        # # Track loss
        # total_loss += loss.item()
        # num_batches += 1
        
        # # Calculate average loss
        # avg_loss = total_loss / max(num_batches, 1)
        return loss.item()
    
    def train(self, epochs=1, episodes_per_epoch=1):
        """
        Train the agent for a specified number of epochs.
        
        Args:
            epochs: Number of training epochs
            episodes_per_epoch: Number of episodes to collect per epoch
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Run the training step
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            avg_loss = self._run_training_step(episodes_per_epoch=episodes_per_epoch)
            
            # Log training metrics
            epoch_time = time.time() - epoch_start_time
            self.metrics["train_loss"].append(avg_loss)
            self.metrics["epoch_time"].append(epoch_time)
            
            # if (epoch + 1) % self.log_interval == 0:
            self.logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")
            
            # Run evaluation
            # if (epoch + 1) % self.eval_interval == 0:
            #     eval_reward = self._run_eval_step()
            #     self.metrics["eval_reward"].append(eval_reward)
            #     self.logger.info(f"Evaluation reward: {eval_reward:.4f}")
            
            # Save checkpoint
            # if self.checkpoint_path and (epoch + 1) % self.checkpoint_interval == 0:
            #     self._save_checkpoint(epoch + 1)
                
        self.logger.info("Training completed")
    
    def _run_eval_step(self):
        """
        Run evaluation.
        
        Returns:
            float: Average evaluation reward
        """
        pass
        # # Simple evaluation: run a few episodes and compute average reward
        # # In a real implementation, you might want a separate evaluation collector
        # eval_episodes = max(1, self.collector.batch_size // 4)
        # eval_data = self.collector.collect(agent=self.agent, num_episodes=eval_episodes)
        
        # # Compute average reward
        # total_reward = 0.0
        # count = 0
        # for batch in eval_data:
        #     total_reward += sum(batch["rewards"])
        #     count += len(batch["rewards"])
        
        # avg_reward = total_reward / max(count, 1)
        # return avg_reward
    
    def _save_checkpoint(self, epoch):
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch number
        """
        if not self.checkpoint_path:
            return
            
        try:
            # Create checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.agent.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": self.metrics
            }
            
            # Save checkpoint
            checkpoint_file = f"{self.checkpoint_path}/checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_file)
            self.logger.info(f"Saved checkpoint to {checkpoint_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")