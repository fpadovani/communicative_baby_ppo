import pandas as pd, matplotlib.pyplot as plt, pathlib

# Load the reward log
file_path = "./txt_files/reward_tracking_log_blue_1_best.csv"
df = pd.read_csv(file_path)

# Sort by batch number just in case
df = df.sort_values(by="batch")

# Calculate moving average with window size 10
window_size = 100
df["moving_avg_reward"] = df["avg_reward"].rolling(window=window_size).mean()

# Plotting
plt.figure(figsize=(10, 6))
#plt.ylim(0, 4)  # Set y-axis limits to match the reward scale (1-5)
plt.plot(df["batch"], df["moving_avg_reward"], label=f"{window_size}-Batch Moving Average", color='blue')
plt.xlabel("Batch")
plt.ylabel("Average Reward")
plt.title(f"Reward Trend Over Time (Window = {window_size})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("./plots/reward_trend_plot_blue_best.png")  # Save the plot as an image file



# Load the reward log
file_path = "./txt_files/reward_tracking_log_sem_1_best.csv"
df = pd.read_csv(file_path)

# Sort by batch number just in case
df = df.sort_values(by="batch")

# Calculate moving average with window size 10
window_size = 200
df["moving_avg_reward"] = df["avg_reward"].rolling(window=window_size).mean()

# Plotting
plt.figure(figsize=(10, 6))
#plt.ylim(0, 4)  # Set y-axis limits to match the reward scale (1-5)
plt.plot(df["batch"], df["moving_avg_reward"], label=f"{window_size}-Batch Moving Average", color='blue')
plt.xlabel("Batch")
plt.ylabel("Average Reward")
plt.title(f"Reward Trend Over Time (Window = {window_size})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("./plots/reward_trend_plot_sem_1_best.png")  # Save the plot as an image file


# Load the reward log
file_path = "./csv_logs/reward_tracking_conf.csv"
df = pd.read_csv(file_path)

# Sort by batch number just in case
df = df.sort_values(by="batch")

# Calculate moving average with window size 10
window_size = 200
df["moving_avg_reward"] = df["avg_reward"].rolling(window=window_size).mean()

# Plotting
plt.figure(figsize=(10, 6))
#plt.ylim(0, 4)  # Set y-axis limits to match the reward scale (1-5)
plt.plot(df["batch"], df["moving_avg_reward"], label=f"{window_size}-Batch Moving Average", color='blue')
plt.xlabel("Batch")
plt.ylabel("Average Reward")
plt.title(f"Reward Trend Over Time (Window = {window_size})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("reward_trend_plot_conf.png")  # Save the plot as an image file



file_path = pathlib.Path("./csv_logs/reward_tracking_conf.csv")
window = 200

df = pd.read_csv(file_path)

# single step axis
max_batch = df["batch"].max()
df["step"] = (df["epoch"]-1) * max_batch + df["batch"]
df = df.sort_values("step")

df["moving"] = df["avg_reward"].rolling(window).mean()

plt.figure(figsize=(10,5))
plt.plot(df["step"], df["moving"], label=f"{window}-batch moving avg")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("global batch #"); plt.ylabel("avg_reward")
plt.title("Reward trend")
plt.grid(True); plt.legend()

plt.savefig("reward_trend_plot_conf2.png")
# plt.show()  # only if you’re in an interactive session
print("Saved → ./plots/reward_trend_plot_conf2.png")



file_path = pathlib.Path("./csv_logs/reward_tracking_conf.csv")
if not file_path.exists():
    raise FileNotFoundError(file_path)

df = pd.read_csv(file_path)

# Build a monotonic axis
max_batch = df["batch"].max()
df["step"] = (df["epoch"]-1)*max_batch + df["batch"]
df = df.sort_values("step")

window = 50          # batches
df["moving"] = df["avg_reward"].rolling(window).mean()

plt.figure(figsize=(10,5))
plt.plot(df["step"], df["moving"], label=f"rolling {window}", color="blue")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("global batch #")
plt.ylabel("avg_reward")
plt.title("Baby PPO reward")
plt.grid(True); plt.legend()
plt.savefig("reward_trend_plot_conf3.png")
plt.close()
print("Plot saved to plots/reward_trend_plot_conf3.png")