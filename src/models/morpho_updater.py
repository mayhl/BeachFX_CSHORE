from abc import ABC, abstractmethod
import numpy as np

class MorphologyUpdater(ABC):
    @abstractmethod
    def update(self, model_outputs: dict) -> dict:
        """
        Returns updated profile {"x": ..., "z": ...} in meters for the next run.
        """
        pass

class MockMorphoUpdater(MorphologyUpdater):
    def update(self, model_outputs: dict) -> dict:
        return {
            "x": model_outputs["final_profile_x"],
            "z": model_outputs["final_profile_zb"],
        }
