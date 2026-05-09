from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class AbstractModelAdapter(ABC):
    """
    Abstract base class for all simulation model adapters.
    Ensures a consistent interface for the simulation orchestrator.
    """

    @abstractmethod
    def prepare_infiles(self, sim_state: Dict[str, Any], output_path: str) -> None:
        """
        Translates standardized sim_state (CID) into model-specific input text files.
        """
        pass

    @abstractmethod
    def run_simulation(self, reach_dir: str, infile_name: str) -> bool:
        """
        Executes the binary and manages binary-level execution.
        """
        pass

    @abstractmethod
    def parse_outputs(self, reach_dir: str) -> None:
        """
        Translates model-specific output back into a standardized CID dictionary.
        """
        pass

    @abstractmethod
    def save_master_parquet(self, reach_dir: str, profile_key: str, data: List[Dict]):
        """
        Aggregates and saves storm results into a master Parquet file.
        """
        pass
