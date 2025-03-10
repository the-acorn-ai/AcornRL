"""Base classes for trainers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from acornrl.agents.base import Agent
from acornrl.collectors.base import Collector


class Trainer(ABC):
    """Base class for trainers."""

    def __init__(self,
                agent: Agent,
                collector: Collector,
                **kwargs):
        """Initialize the trainer with an agent and data collector.

        Args:
            agent: The agent to train
            collector: The data collector to gather training samples
            **kwargs: Additional trainer-specific parameters
        """
        self.agent = agent
        self.collector = collector
        self._setup(**kwargs)

    def _setup(self, **kwargs):
        """Setup trainer-specific components.

        Override this method to initialize trainer-specific components.
        """
        pass

    def train(self,
            num_epochs: int,
            samples_per_epoch: Optional[int] = None,
            **kwargs):
        """Train the agent for a specified number of epochs.

        Args:
            num_epochs: Number of training epochs
            samples_per_epoch: Number of samples to collect per epoch (if None, uses collector's batch_size)
            **kwargs: Additional training parameters
        """
        if samples_per_epoch is None:
            samples_per_epoch = self.collector.batch_size

        for epoch in range(num_epochs):
            # Collect training data
            training_data = self.collector.collect(samples_per_epoch)

            # Train the agent on collected data
            loss = self._train_epoch(training_data, epoch=epoch, **kwargs)

            # Optional hook for epoch end processing
            self._on_epoch_end(epoch, loss)

    def _train_epoch(self, training_data: List[Dict[str, Any]], epoch: int, **kwargs) -> float:
        """Train for a single epoch on the provided data.

        Args:
            training_data: List of training samples
            epoch: Current epoch number
            **kwargs: Additional parameters

        Returns:
            float: Training loss for this epoch
        """
        # Default implementation - subclasses should override
        total_loss = 0.0

        # Process each training sample
        for sample in training_data:
            # Use the agent to compute loss on this sample
            loss = self.agent.compute_loss(**sample)
            total_loss += loss

        return total_loss / len(training_data) if training_data else 0.0

    def _on_epoch_end(self, epoch: int, loss: float):
        """Hook called at the end of each training epoch.

        Args:
            epoch: The completed epoch number
            loss: The average loss for the epoch
        """
        pass
