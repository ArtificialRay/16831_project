import os
import numpy as np
import matplotlib.pyplot as plt


def read(csv_path):
    """
    Reads CSV with format:
    env_steps,mean_return
    1000,-12.3
    2000,-11.8
    ...
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    xs = data[:, 0]
    ys = data[:, 1]
    return xs, ys


# --- load data ---
csv_path = "results/random_agent_return_curve.csv"
xs, ys = read(csv_path)

# --- plot ---
plt.figure(figsize=(6, 4))
plt.plot(xs, ys, linewidth=2)

plt.xlabel("Environment steps")
plt.ylabel("Mean episodic return (recent episodes)")
plt.title("Random Agent Performance")

# adjust this range however you like
plt.ylim(-40, 0)

plt.grid(True)

# --- save ---
os.makedirs("results", exist_ok=True)
out_png = "results/random_agent_return_curve.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")

print(f"Saved plot to {out_png}")
