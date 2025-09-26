import pandas as pd, matplotlib.pyplot as plt, pathlib

window = 200

##### BLEU #####
file_path = pathlib.Path("./txt_files/rewards_files/reward_tracking_blue.csv")
df = pd.read_csv(file_path)
max_batch = df["batch"].max()
df["step"] = (df["epoch"] - 1) * max_batch + df["batch"]
df = df.sort_values("step")
df["moving"] = df["avg_reward"].rolling(window).mean()

plt.figure(figsize=(8, 5))
plt.plot(df["step"], df["moving"], label=f"{window}-batch moving avg", linewidth=1.2)  # default color
plt.axhline(0, color="grey", lw=0.5)
plt.axvline(x=13750, color="red", linestyle="--", linewidth=0.8)
plt.text(13750 + 200, 0.05, "epoch 1", color="black", fontsize=12)
plt.ylim(0, 1)
plt.xlabel("Global batch", fontsize=20, labelpad=12)
plt.ylabel("Reward", fontsize=20, labelpad=12)
plt.xticks(fontsize=16, rotation=20)
plt.yticks(fontsize=20)
plt.title("1-gram BLEU", fontsize=25, pad=20, loc='left', x=0.32)
plt.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
for spine in plt.gca().spines.values(): spine.set_edgecolor('lightgrey')
plt.legend(fontsize=22)
plt.tight_layout()
plt.savefig("./final_plots/bleu.pdf")
plt.close()


##### SEM_SIM #####
file_path = pathlib.Path("./txt_files/rewards_files/reward_tracking_semsim.csv")
df = pd.read_csv(file_path)
max_batch = df["batch"].max()
df["step"] = (df["epoch"] - 1) * max_batch + df["batch"]
df = df.sort_values("step")
df["moving"] = df["avg_reward"].rolling(window).mean()

plt.figure(figsize=(8, 5))
plt.plot(df["step"], df["moving"], label=f"{window}-batch moving avg", color="#0d3d73", linewidth=1.2)
plt.axhline(0, color="grey", lw=0.5)
plt.axvline(x=13750, color="red", linestyle="--", linewidth=0.8)
plt.text(13750 + 200, 0.05, "epoch 1", color="black", fontsize=12)
plt.ylim(0, 1)
plt.xlabel("Global batch", fontsize=20, labelpad=12)
plt.ylabel("Reward", fontsize=20, labelpad=12)
plt.xticks(fontsize=16, rotation=20)
plt.yticks(fontsize=20)
plt.title("Semantic Similarity", fontsize=25, pad=20, loc='left', x=0.23)
plt.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
for spine in plt.gca().spines.values(): spine.set_edgecolor('lightgrey')
plt.legend(fontsize=22)
plt.tight_layout()
plt.savefig("./final_plots/semsim.pdf")
plt.close()


##### SCORE #####
file_path = pathlib.Path("./txt_files/rewards_files/reward_tracking_log_score.csv")
df = pd.read_csv(file_path)
max_batch = df["batch"].max()
df["step"] = (df["epoch"] - 1) * max_batch + df["batch"]
df = df.sort_values("step")
df = df[df["step"] <= 11000]
df["moving"] = df["avg_reward"].rolling(window).mean()

plt.figure(figsize=(8, 5))
plt.plot(df["step"], df["moving"], label=f"{window}-batch moving avg", color="#3399ff", linewidth=1.2)
plt.axhline(0, color="grey", lw=0.5)
plt.axvline(x=13750, color="red", linestyle="--", linewidth=0.8)
plt.text(13750 + 300, 2.6, "epoch 1", color="black", fontsize=12)
plt.ylim(2.5, 3.5)
plt.xlabel("Global batch", fontsize=20, labelpad=12)
plt.ylabel("Reward", fontsize=20, labelpad=12)
plt.xticks(fontsize=16, rotation=20)
plt.yticks(fontsize=20)
plt.title("LLM Score", fontsize=25, pad=20, loc='left', x=0.36)
plt.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
for spine in plt.gca().spines.values(): spine.set_edgecolor('lightgrey')
plt.legend(fontsize=22)
plt.tight_layout()
plt.savefig("./final_plots/score.pdf")
plt.close()


##### CONFIDENCE #####
file_path = pathlib.Path("./txt_files/rewards_tracking/reward_tracking_conf_last.csv")
df = pd.read_csv(file_path)
max_batch = df["batch"].max()
df["step"] = (df["epoch"] - 1) * max_batch + df["batch"]
df = df.sort_values("step")
df["moving"] = df["avg_reward"].rolling(window).mean()

plt.figure(figsize=(8, 5))
plt.plot(df["step"], df["moving"], label=f"{window}-batch moving avg", color="#66b3ff", linewidth=1.2)
plt.axhline(0, color="grey", lw=0.5)
plt.axvline(x=8645, color="red", linestyle="--", linewidth=0.8)
plt.text(8645 + 150, 0.25, "epoch 1", color="black", fontsize=12)
plt.ylim(0.2, 1)
plt.xlabel("Global batch", fontsize=20, labelpad=12)
plt.ylabel("Reward", fontsize=20, labelpad=12)
plt.xticks(fontsize=16, rotation=20)
plt.yticks(fontsize=20)
plt.title("Confidence", fontsize=25, pad=20, loc='left', x=0.33)
plt.grid(True, linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
for spine in plt.gca().spines.values(): spine.set_edgecolor('lightgrey')
plt.legend(fontsize=22)
plt.tight_layout()
plt.savefig("./final_plots/reward_trend_plot_confidence.pdf")
plt.close()
