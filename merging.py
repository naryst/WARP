import torch
from copy import deepcopy


def scale_model_weights(model, scaling_factor):
    scaled_model = deepcopy(model)
    with torch.no_grad():
        for param in scaled_model.parameters():
            param *= scaling_factor
    return scaled_model


def sum_model_weights(model1, model2):
    scaled_model = deepcopy(model1)
    with torch.no_grad():
        for param1, param2 in zip(scaled_model.parameters(), model2.parameters()):
            param1 += param2
    return scaled_model


def models_interpolate(model, additional_model, mu):
    scaled_model = scale_model_weights(model, mu)
    scaled_model_ema = scale_model_weights(additional_model, (1 - mu))
    return sum_model_weights(scaled_model, scaled_model_ema)


def lerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    return (1 - t) * v0 + t * v1


def normalize(v: torch.Tensor, eps: float):
    norm_v = torch.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v


def slerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
):
    """
    Spherical linear interpolation

    From: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    Args:
        t: Float value between 0.0 and 1.0
        v0: Starting layer weights
        v1: Final layer weights
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        v2: Interpolation vector between v0 and v1
    """
    # Copy the vectors to reuse them later
    v0_copy = v0.detach().clone()
    v1_copy = v1.detach().clone()

    # Normalize the vectors to get the directions and angles
    v0 = normalize(v0, eps)
    v1 = normalize(v1, eps)

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = torch.sum(v0 * v1)

    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    if torch.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        return res

    # Calculate initial angle between v0 and v1
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return res


def slerp_models(model1, model2, Lambda=0.5):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        param1 = slerp(Lambda, param1, param2)
    return model1
