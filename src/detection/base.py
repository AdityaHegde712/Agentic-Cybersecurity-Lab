from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def update(self, y):
        """Update the detector with new observation y."""
        pass
        
    @abstractmethod
    def reset(self):
        """Reset the detector state."""
        pass