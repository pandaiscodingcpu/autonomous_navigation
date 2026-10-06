"""Small shared helpers."""
import numpy as np


def wrap_pi(a):
    """Wrap an angle (scalar or array) to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi
