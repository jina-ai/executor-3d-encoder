"""
ref: https://github.com/hancyran/RepSurf/blob/44b8da1a40/modules/polar_utils.py
"""

import numpy as np
import torch


def xyz2sphere(xyz, normalize=True):
    """
    Convert XYZ to Spherical Coordinate
    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    :param xyz: [B, N, 3] / [B, N, G, 3]
    :return: (rho, theta, phi) [B, N, 3] / [B, N, G, 3]
    """
    rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, min=0)  # range: [0, inf]
    theta = torch.acos(xyz[..., 2, None] / rho)  # range: [0, pi]
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # check nan
    idx = rho == 0
    theta[idx] = 0

    if normalize:
        theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + 0.5  # [0, 1]
    out = torch.cat([rho, theta, phi], dim=-1)
    return out


def xyz2cylind(xyz, normalize=True):
    """
    Convert XYZ to Cylindrical Coordinate
    reference: https://en.wikipedia.org/wiki/Cylindrical_coordinate_system
    :param normalize: Normalize phi & z
    :param xyz: [B, N, 3] / [B, N, G, 3]
    :return: (rho, phi, z) [B, N, 3]
    """
    rho = torch.sqrt(torch.sum(torch.pow(xyz[..., :2], 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, 0, 1)  # range: [0, 1]
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    z = xyz[..., 2, None]
    z = torch.clamp(z, -1, 1)  # range: [-1, 1]

    if normalize:
        phi = phi / (2 * np.pi) + 0.5
        z = (z + 1.0) / 2.0
    out = torch.cat([rho, phi, z], dim=-1)
    return
