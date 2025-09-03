from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def evaluate(self, df):
        """
        Evaluate the latest candle and return a trade signal.
        Must return dict like:
        {
            "signal": "long" or "short" or None,
            "entry": float,
            "sl": float,
            "tp": float,
        }
        """
        pass

