import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_scalars(log_dir, tag):
    steps = []
    values = []

    for file in os.listdir(log_dir):
        if "events" not in file:
            continue

        ea = event_accumulator.EventAccumulator(os.path.join(log_dir, file))
        ea.Reload()

        if tag not in ea.Tags()["scalars"]:
            print(f"Tag '{tag}' not found in {file}")
            continue

        scalars = ea.Scalars(tag)
        steps.extend([s.step for s in scalars])
        values.extend([s.value for s in scalars])

    return np.array(steps), np.array(values)

def smooth(values, weight=0.99):
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


log_dir = "runs/Assault/"
tag = "Rewards/rew_avg"
runs = ["autoencoder_transfer", 'control', 'encoder_transfer']
ylim_low = np.inf
ylim_high = -np.inf

plt.figure(figsize=(8, 5))
for i, run_name in enumerate(runs):
    color = f'C{i}'
    steps, values = load_tensorboard_scalars(log_dir + run_name, tag)
    smooth_values = smooth(values, weight=0.9)

    plt.plot(steps, values, color, alpha=0.4, label=f"raw {run_name}")
    plt.plot(steps, smooth_values, color, linewidth=2, label=f"smoothed {run_name}")
    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.legend()
    ylim_low = min(np.min(smooth_values), ylim_low)
    ylim_high = max(np.max(smooth_values) + 200, ylim_high)
    
plt.ylim(ylim_low, ylim_high)
plt.title("All")
plt.savefig("runs.png")
plt.ylim(0, 500)
plt.xlim(0, 500_000)
plt.savefig("runs_zoomed.png")

for i, run_name in enumerate(runs):
    steps, values = load_tensorboard_scalars(log_dir + run_name, tag)
    smooth_values = smooth(values, weight=0.9)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, alpha=0.4, label=f"raw {run_name}")
    plt.plot(steps, smooth_values, linewidth=2, label=f"smoothed {run_name}")
    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.title(f"{run_name}")
    plt.legend()
    plt.ylim(np.min(smooth_values), np.max(smooth_values) + 200)
    
    plt.savefig(f"{run_name}.png")


tag = "Loss/reward_loss"
plt.figure(figsize=(8, 5))
for i, run_name in enumerate(runs):
    color = f'C{i}'
    steps, values = load_tensorboard_scalars(log_dir + run_name, tag)
    smooth_values = smooth(values, weight=0.9)

    plt.plot(steps, values, color, alpha=0.4, label=f"raw {run_name}")
    plt.plot(steps, smooth_values, color, linewidth=2, label=f"smoothed {run_name}")
    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.legend()

plt.title("All")
plt.savefig("reward_loss.png")

log_dir = "runs/SpaceInvaders/"
tag = "Rewards/rew_avg"
runs = ['control', 'transfer_0', 'transfer_1', 'transfer_2']
ylim_low = np.inf
ylim_high = -np.inf

plt.figure(figsize=(8, 5))
for i, run_name in enumerate(runs):
    color = f'C{i}'
    steps, values = load_tensorboard_scalars(log_dir + run_name, tag)
    inds = steps.argsort()
    steps = steps[inds]
    values = values[inds]
    smooth_values = smooth(values, weight=0.9)

    plt.plot(steps, smooth_values, color, linewidth=2, label=f"smoothed {run_name}")
    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.legend()
    ylim_low = min(np.min(smooth_values), ylim_low)
    ylim_high = max(np.max(smooth_values) + 200, ylim_high)
    
plt.title("World Model Transfer")
plt.ylim(ylim_low, ylim_high)
plt.xlim(0, 500_000)
plt.savefig("world_model_transfer.png")
