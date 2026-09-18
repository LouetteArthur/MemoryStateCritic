# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

# Rate profiles implementations


def betaflight_rate_profile(
    rc_input,  # shape: [N, 3]
    rc_rate=torch.tensor([1.55, 1.55, 1.50]),
    super_rate=torch.tensor([0.73, 0.73, 0.73]),
    rc_expo=torch.tensor([0.30, 0.30, 0.30]),
    super_expo_active=True,
    limit=torch.tensor([2000.0, 2000.0, 2000.0]),
):
    """
    Fully vectorized Betaflight rate profile over [N, 3] RC input.
    Each row of rc_input is a 3D command (roll, pitch, yaw).
    output in rad/s
    """
    rc_rate = rc_rate.view(1, 3).to(rc_input.device)
    super_rate = super_rate.view(1, 3).to(rc_input.device)
    rc_expo = rc_expo.view(1, 3).to(rc_input.device)
    limit = limit.view(1, 3).to(rc_input.device)

    # RC Rate > 2 shaping
    rc_rate = torch.where(rc_rate > 2, rc_rate + (rc_rate - 2) * 14.54, rc_rate)

    # Expo shaping
    expo_power = 3
    rc_input_shaped = rc_input * (rc_input.abs() ** expo_power) * rc_expo + rc_input * (1 - rc_expo)

    # Super Expo shaping
    if super_expo_active:
        rc_factor = 1.0 / torch.clamp(1.0 - rc_input_shaped.abs() * super_rate, 0.01, 1.0)
        angular_vel = 200 * rc_rate * rc_input_shaped * rc_factor
    else:
        angular_vel = (((rc_rate * 100) + 27) * rc_input_shaped / 16.0) / 4.1

    angular_vel = torch.clamp(angular_vel, -limit, limit)
    return angular_vel  # [N, 3]


def raceflight_rate_profile(
    rc_input,  # shape: [N, 3]
    rc_rate=torch.tensor([1.0, 1.0, 1.0]),
    expo=torch.tensor([0.4, 0.4, 0.4]),
    rate=torch.tensor([0.75, 0.75, 0.75]),
    limit=torch.tensor([2000.0, 2000.0, 2000.0]),
):
    """
    Fully vectorized RaceFlight (FlightOne) rate profile over [N, 3] RC input.
    https://github.com/Marc-Anderson/multirotor-rate-converter/blob/master/app/api/ratefitter.py
    """
    rc_input = torch.as_tensor(rc_input, dtype=torch.float32)  # [N, 3]
    rc_rate = rc_rate.view(1, 3).to(rc_input.device)
    rate = rate.view(1, 3).to(rc_input.device)
    expo = expo.view(1, 3).to(rc_input.device)
    limit = limit.view(1, 3).to(rc_input.device)

    rc_input_shaped = rc_input * (rc_input.abs() ** 3) * expo + rc_input * (1 - expo)
    angular_vel = rc_input_shaped * rate * rc_rate * 667.0

    angular_vel = torch.clamp(angular_vel, -limit, limit)

    return angular_vel  # [N, 3]


def actual_rate_profile(
    rc_input,  # shape: [N, 3]
    center_sensitivity=torch.tensor([1.0, 1.0, 1.0]),
    max_vel=torch.tensor([1100.0, 1100.0, 1100.0]),
    expo=torch.tensor([0.3, 0.3, 0.3]),
    acro_rate=torch.tensor([1.0, 1.0, 1.0]),
    limit=torch.tensor([2000.0, 2000.0, 2000.0]),
):
    """
    Fully vectorized Actual rate profile over [N, 3] RC input.
    """
    rc_input = torch.as_tensor(rc_input, dtype=torch.float32)
    expo = expo.view(1, 3).to(rc_input.device)
    center_sensitivity = center_sensitivity.view(1, 3).to(rc_input.device)
    max_vel = max_vel.view(1, 3).to(rc_input.device)
    acro_rate = acro_rate.view(1, 3).to(rc_input.device)
    limit = limit.view(1, 3).to(rc_input.device)

    stick = rc_input.abs()
    expo_curve = stick * stick * stick * expo + stick * (1 - expo)
    angular_vel = (
        torch.sign(rc_input) * (center_sensitivity + (1 - center_sensitivity) * expo_curve) * max_vel * acro_rate
    )

    angular_vel = torch.clamp(angular_vel, -limit, limit)
    return angular_vel  # [N, 3]


def kiss_rate_profile(
    rc_input,  # shape: [N, 3]
    rate=torch.tensor([1.5, 1.5, 1.5]),
    rc_curve=torch.tensor([0.3, 0.3, 0.3]),
    limit=torch.tensor([2000.0, 2000.0, 2000.0]),
):
    """
    Fully vectorized KISS rate profile over [N, 3] RC input.
    https://github.com/Marc-Anderson/multirotor-rate-converter/blob/master/app/api/ratefitter.py
    """
    rc_input = torch.as_tensor(rc_input, dtype=torch.float32)
    rate = rate.view(1, 3).to(rc_input.device)
    rc_curve = rc_curve.view(1, 3).to(rc_input.device)
    limit = limit.view(1, 3).to(rc_input.device)

    expo_input = rc_input * (rc_input.abs() ** 3) * rc_curve + rc_input * (1 - rc_curve)
    angular_vel = expo_input * rate * 1000.0

    angular_vel = torch.clamp(angular_vel, -limit, limit)
    return angular_vel  # [N, 3]


# ------------------------------
# Mixer with your geometry
# ------------------------------
class QuadMixer:
    r"""
    Maps (T, Mx, My, Mz) to rotor angular speeds omega [rad/s] using the
    Crazyflie brushless allocation matrix.
    """

    def __init__(self, num_envs, drone_cfg, device="cuda"):
        self.device = device
        self.k_eta = drone_cfg.k_eta.to(device)
        _, self.TM_to_f = drone_cfg.get_mixer()
        self.motor_speed_min = float(getattr(drone_cfg, "motor_speed_min", 0.0))
        self.motor_speed_max = float(getattr(drone_cfg, "motor_speed_max", float("inf")))

    def __call__(self, thrust: torch.Tensor, moments: torch.Tensor) -> torch.Tensor:
        """Return rotor speeds for the desired wrench."""
        wrench = torch.cat((thrust, moments), dim=-1)
        motor_forces = torch.matmul(wrench, self.TM_to_f.t()).clamp_min(0.0)
        omega = torch.sqrt(motor_forces / self.k_eta)
        return omega.clamp(self.motor_speed_min, self.motor_speed_max)


class PDRateController:
    """Lee geometric controller (attitude error law from integrated rate commands)."""

    def __init__(self, num_envs, drone_cfg, device="cuda", dt=0.01):
        self.inertia_matrix = torch.diag_embed(drone_cfg.inertia)
        self.gain_body_rate = torch.tensor([0.25, 0.25, 0.15], device=device) @ self.inertia_matrix.inverse()
        print(
            f"[INFO]: Controller gains: roll {self.gain_body_rate[0]:.0f}, pitch {self.gain_body_rate[1]:.0f}, yaw"
            f" {self.gain_body_rate[2]:.0f}"
        )

    def __call__(self, body_rate: torch.Tensor, body_rate_des: torch.Tensor):
        body_rate_err = torch.clamp(body_rate - body_rate_des, -1, 1)
        # Compute gyroscopic term ω × (J ω)
        J_omega = body_rate @ self.inertia_matrix.T  # (N,3)
        coriolis = torch.cross(body_rate, J_omega, dim=1)  # (N,3)
        # PD torque law
        M = -self.gain_body_rate * body_rate_err + coriolis
        return M


# ------------------------------
# Wrapper
# ------------------------------
class BodyRateToOmega:
    """Controller + mixer wrapper: (body_rate, body_rate_des, thrust) -> omega (rotor speeds)."""

    def __init__(self, controller, mixer):
        super().__init__()
        self.controller = controller
        self.mixer = mixer

    def __call__(self, body_rate, body_rate_des, thrust):
        M = self.controller(body_rate, body_rate_des)
        omega = self.mixer(thrust, M)
        return omega
