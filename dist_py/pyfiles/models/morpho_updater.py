from abc import ABC, abstractmethod
import numpy as np

class MorphologyUpdater(ABC):
    @abstractmethod
    def update(self, model_outputs: dict) -> np.ndarray:
        """
        Updates bathymetry based on model outputs.
        """
        pass

class MockMorphoUpdater(MorphologyUpdater):
    def update(self, model_outputs: dict) -> np.ndarray:
        # Standard implementation: extract final profile from model outputs
        # Accessing the specific requested data structure
        return model_outputs["final_profile_zb"]
