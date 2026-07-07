from __future__ import annotations

import math
from typing import Optional, Tuple


def _max_time_for_range(range_m: float, muzzle_speed_mps: float, theta_rad: float) -> float:
    if range_m <= 0.0 or muzzle_speed_mps <= 0.0:
        return 0.0
    cos_theta = abs(math.cos(theta_rad))
    cos_theta = max(0.05, cos_theta)
    base_time = range_m / max(1e-6, muzzle_speed_mps * cos_theta)
    return max(0.5, base_time * 5.0)


def simulate_y_at_range(
    range_m: float,
    muzzle_speed_mps: float,
    theta_rad: float,
    drag_k: float,
    dt: float,
    g: float = 9.81,
    max_time_s: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    if range_m <= 0.0 or muzzle_speed_mps <= 0.0 or dt <= 0.0:
        return None
    cos_theta = math.cos(theta_rad)
    if cos_theta <= 1e-6:
        return None
    vx = muzzle_speed_mps * cos_theta
    vy = muzzle_speed_mps * math.sin(theta_rad)
    x = 0.0
    y = 0.0
    t = 0.0
    prev_x = x
    prev_y = y
    prev_t = t
    max_time = max_time_s if max_time_s is not None else _max_time_for_range(
        range_m, muzzle_speed_mps, theta_rad
    )
    max_steps = int(max_time / dt) + 1
    for _ in range(max_steps):
        v = math.hypot(vx, vy)
        ax = -drag_k * v * vx
        ay = -g - drag_k * v * vy
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        if x >= range_m:
            if x == prev_x:
                return y, t
            ratio = (range_m - prev_x) / (x - prev_x)
            y_at = prev_y + ratio * (y - prev_y)
            t_at = prev_t + ratio * (t - prev_t)
            return y_at, t_at
        prev_x = x
        prev_y = y
        prev_t = t
        if vx <= 0.0:
            break
    return None


def solve_pitch_for_target(
    range_m: float,
    height_m: float,
    muzzle_speed_mps: float,
    drag_k: float,
    pitch_min_rad: float,
    pitch_max_rad: float,
    dt: float,
    g: float = 9.81,
    max_iter: int = 24,
) -> Optional[Tuple[float, float]]:
    if range_m <= 0.0 or muzzle_speed_mps <= 0.0 or dt <= 0.0:
        return None
    low = float(pitch_min_rad)
    high = float(pitch_max_rad)
    if low > high:
        low, high = high, low
    low_result = simulate_y_at_range(
        range_m, muzzle_speed_mps, low, drag_k, dt, g=g
    )
    high_result = simulate_y_at_range(
        range_m, muzzle_speed_mps, high, drag_k, dt, g=g
    )
    if low_result is None or high_result is None:
        return None
    low_y, low_t = low_result
    high_y, high_t = high_result
    f_low = low_y - height_m
    f_high = high_y - height_m
    if abs(f_low) <= 1e-4:
        return low, low_t
    if abs(f_high) <= 1e-4:
        return high, high_t
    if f_low * f_high > 0.0:
        return None

    mid = 0.5 * (low + high)
    mid_t = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_result = simulate_y_at_range(
            range_m, muzzle_speed_mps, mid, drag_k, dt, g=g
        )
        if mid_result is None:
            return None
        mid_y, mid_t = mid_result
        f_mid = mid_y - height_m
        if abs(f_mid) <= 1e-4:
            return mid, mid_t
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return mid, mid_t
