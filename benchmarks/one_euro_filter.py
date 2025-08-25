# one_euro_filter_np.py
import numpy as np
import math

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.s = None

    def __call__(self, x, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        if self.y is None:
            self.s = x
        else:
            self.s = self.alpha * x + (1.0 - self.alpha) * self.s
        self.y = x
        return self.s

class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter(self._alpha(self.min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(self.d_cutoff))
        self.last_time = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        if self.last_time is not None and timestamp is not None:
            self.freq = 1.0 / (timestamp - self.last_time)
        self.last_time = timestamp

        prev_x = self.x_filter.y
        dx = np.zeros_like(x) if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter(dx, alpha=self._alpha(self.d_cutoff))
        
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        
        return self.x_filter(x, alpha=self._alpha(cutoff))