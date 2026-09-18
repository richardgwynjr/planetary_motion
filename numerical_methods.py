import numpy as np


def forward_difference(values, step_size):
    """Calculate the periodic forward-difference derivative."""
    return (np.roll(values, -1) - values) / step_size


def backward_difference(values, step_size):
    """Calculate the periodic backward-difference derivative."""
    return (values - np.roll(values, 1)) / step_size


def central_difference(values, step_size):
    """Calculate the periodic central-difference derivative."""
    next_values = np.roll(values, -1)
    previous_values = np.roll(values, 1)

    return (next_values - previous_values) / (2 * step_size)


def second_central_difference(values, step_size):
    """Calculate the periodic second derivative."""
    next_values = np.roll(values, -1)
    previous_values = np.roll(values, 1)

    return (
        next_values
        - 2 * values
        + previous_values
    ) / (step_size**2)
