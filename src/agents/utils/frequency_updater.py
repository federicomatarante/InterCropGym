from abc import ABC, abstractmethod

import numpy as np


class FrequencyUpdater(ABC):

    def __init__(self):
        self.updates = 0
        self._steps = 0

    @abstractmethod
    def _get_frequency(self) -> int:
        pass

    def update(self) -> bool:
        if self._steps >= self._get_frequency():
            self._steps = 0
            self.updates += 1
            return True
        return False

    def step(self):
        self._steps += 1


class FixedUpdater(FrequencyUpdater):
    def __init__(self, frequency: int):
        super().__init__()
        self.frequency = frequency

    def _get_frequency(self) -> int:
        return self.frequency


class DecayingUpdater(FrequencyUpdater):
    def __init__(self, start_freq=10, min_freq=1, decay_rate=0.995):
        super().__init__()
        self.current_freq = start_freq
        self.min_freq = min_freq
        self.decay_rate = decay_rate
        self.steps = 0

    def _get_frequency(self) -> int:
        self.current_freq *= self.decay_rate
        self.current_freq = max(self.current_freq, self.min_freq)
        return int(np.ceil(self.current_freq))
