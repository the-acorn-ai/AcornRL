from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


class Agent(ABC):
    """
    Base class for all agents.
    Each implementation defines its own config class as needed.
    """
    def __init__(self,
                 **kwargs):
        """
        Initialize an agent.

        Args:
            name: Name identifier for the agent
            config: Agent configuration
            **kwargs: Additional parameters
        """
        # Initialize the agent
        self._initialize_agent(**kwargs)

    def _initialize_agent(self, **kwargs):
        """Initialize agent-specific components."""
        pass

    @abstractmethod
    def __call__(self, observation: Union[str, Dict[str, Any]]) -> str:
        """
        Process observation and return action.

        Args:
            observation: The environment observation

        Returns:
            str: Action to take
        """
        pass

    def compute_loss(self,
                     **kwargs) -> float:
        """
        Compute the loss for the agent.
        """
        pass

    def cleanup(self) -> None:
        """
        Cleanup any resources.
        Override this method to handle resource cleanup.
        """
        pass
