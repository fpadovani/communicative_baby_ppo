import pandas as pd
import matplotlib.pyplot as plt

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