import matplotlib.pyplot as plt
import numpy as np
import os

episodes_file = "episods.npy"
rewards_file = "rewards.npy"
loss_file = "loss.npy"
# episode_rewards = np.load(rewards_file)
# Check if file exists before loading
if os.path.exists(rewards_file):
    episode_rewards = np.load(rewards_file)
    print("Rewards:", episode_rewards)
else:
    print(f"File not found: {rewards_file}. Please check the file path and try again.")
episode_lengths = np.load(episodes_file)
loss = np.load(loss_file)
# Plot the data
plt.figure(figsize=(12, 6))

# Plot rewards
plt.subplot(3, 1, 1)
plt.plot(episode_rewards, label="Episode Reward")
plt.title("Episode Rewards")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()

# Plot episode lengths
plt.subplot(3, 1, 2)
plt.plot(episode_lengths, label="Episode Length", color="orange")
plt.title("Episode Lengths")
plt.xlabel("Episodes")
plt.ylabel("Length")
plt.legend()

# Plot episode lengths
plt.subplot(3, 1, 3)
plt.plot(loss, label="Model Loss")
plt.title("Loss in Episode")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


