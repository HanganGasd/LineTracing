"""Compare baseline PPO, P correction, and backward-difference PD correction."""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from driving.simulation import CameraObservationWrapper
from lane_keeping.environment import LaneKeepingEnv
from rl_core.ppo import CameraActorCritic

SEEDS = [1, 2, 3, 4, 5]
MAX_STEPS = 1000
KP = 0.35
KD = 0.06
MAX_CORRECTION = 0.35
MODES = ["PPO", "P", "PD"]


def signed_error(env):
    """Signed perpendicular distance to the nearest track segment.

    Positive means right of the ordered track direction (screen y points down).
    """
    px, py = env.car_x, env.car_y
    points = env.track_points
    best_d2 = float("inf")
    best_e = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        vx, vy = x2 - x1, y2 - y1
        length2 = vx * vx + vy * vy
        if length2 < 1e-9:
            continue
        wx, wy = px - x1, py - y1
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / length2))
        dx, dy = wx - t * vx, wy - t * vy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            cross = vx * wy - vy * wx
            best_e = math.copysign(math.sqrt(d2), cross) if cross else 0.0
    return best_e


def make_env():
    # Fixed track and no camera randomization for an initial controlled comparison.
    return CameraObservationWrapper(
        LaneKeepingEnv(
            render_mode=False,
            procedural_tracks=False,
            domain_randomization=False,
        )
    )


def main():
    device = torch.device("cpu")
    env = make_env()
    model = CameraActorCritic(
        env.image_shape, env.auxiliary_size, 3, env.action_space.shape[0]
    ).to(device)
    checkpoint = Path(__file__).resolve().parent / "lane_keeping" / "camera_ppo_lane_keeping_v2.pt"
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Trained PPO checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    summaries = []
    records = []
    try:
        for mode in MODES:
            for seed in SEEDS:
                observation, _ = env.reset(seed=seed)
                base = env.unwrapped
                prev_error = signed_error(base) / max(base.road_width / 2, 1e-6)
                abs_errors = [abs(signed_error(base))]
                total_reward = 0.0
                total_steer_change = 0.0
                prev_command = None
                reason = "max_test_steps"
                steps = 0

                for step in range(1, MAX_STEPS + 1):
                    obs = torch.as_tensor(
                        observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.inference_mode():
                        action = torch.tanh(model.actor(obs))[0].cpu().numpy()

                    error = signed_error(base) / max(base.road_width / 2, 1e-6)
                    derivative = (error - prev_error) / base.dt
                    correction = 0.0
                    if mode in ("P", "PD"):
                        correction = -KP * error
                    if mode == "PD":
                        correction -= KD * derivative
                    correction = float(np.clip(correction, -MAX_CORRECTION, MAX_CORRECTION))
                    action[0] = np.clip(float(action[0]) + correction, -1.0, 1.0)
                    if prev_command is not None:
                        total_steer_change += abs(float(action[0]) - prev_command)
                    prev_command = float(action[0])
                    prev_error = error

                    observation, reward, terminated, truncated, info = env.step(action)
                    steps = step
                    total_reward += float(reward)
                    abs_errors.append(abs(float(info["dist_to_center"])))
                    records.append({
                        "mode": mode,
                        "seed": seed,
                        "time": step * base.dt,
                        "distance_from_center": float(info["dist_to_center"]),
                        "steering_command": float(action[0]),
                        "reward": float(reward),
                    })
                    if terminated or truncated:
                        reason = info["done_reason"]
                        break

                times = np.arange(steps + 1) * base.dt
                integral = float(np.trapezoid(abs_errors, times))
                summary = {
                    "mode": mode,
                    "seed": seed,
                    "steps": steps,
                    "duration_s": steps * base.dt,
                    "mean_error_px": integral / (steps * base.dt),
                    "max_error_px": max(abs_errors),
                    "integrated_error_px_s": integral,
                    "distance_px": float(info["total_distance"]),
                    "progress_px": float(info["lap_progress"]),
                    "mean_steer_change": total_steer_change / max(steps - 1, 1),
                    "total_reward": total_reward,
                    "done_reason": reason,
                }
                summaries.append(summary)
                print(
                    f"{mode:3s} seed={seed} steps={steps:3d} "
                    f"mean_error={summary['mean_error_px']:.2f}px "
                    f"progress={summary['progress_px']:.1f}px "
                    f"end={reason}"
                )
    finally:
        env.close()

    for filename, rows in [("pd_summary.csv", summaries), ("pd_steps.csv", records)]:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {filename}")

    means = [
        np.mean([r["mean_error_px"] for r in summaries if r["mode"] == mode])
        for mode in MODES
    ]
    plt.figure(figsize=(7, 4))
    plt.bar(MODES, means)
    plt.ylabel("Time-averaged center error (px)")
    plt.title("PPO vs P vs PD (backward difference)")
    plt.tight_layout()
    plt.savefig("pd_comparison.png", dpi=160)
    plt.close()
    print("Saved pd_comparison.png")


if __name__ == "__main__":
    main()
