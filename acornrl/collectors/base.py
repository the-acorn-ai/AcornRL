"""Base classes for data collection workers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from acornrl.agents.base import Agent

class Collector(ABC):

    def __init__(
        self,
        batch_size: int,
    ):
        self.batch_size = batch_size


    @abstractmethod
    def collect(self, agent: Agent, num_samples: int) -> List[Dict[str, Any]]:
        """Collect data samples."""
        pass
