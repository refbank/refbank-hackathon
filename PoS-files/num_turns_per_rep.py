import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths
message_path = "harmonized_data/hawkins2019_continual//messages.csv"
trial_path = "harmonized_data/hawkins2019_continual//trials.csv"

# Extract paper name from folder
paper_name = os.path.basename(os.path.dirname(message_path))

# Load data
messages_df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)

# Merge to get rep_num
df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')

# Count number of messages per trial
turns_per_trial = df.groupby('trial_id').size().reset_index(name='num_turns')

# Merge with rep_num
turns_per_trial = pd.merge(turns_per_trial, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')

# Average number of turns per rep_num
turns_by_rep = turns_per_trial.groupby('rep_num')['num_turns'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=turns_by_rep, x='rep_num', y='num_turns', marker='o')
plt.title(f"{paper_name}: Average Number of Turns per Trial Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Number of Turns per Trial")
plt.grid(True)
plt.tight_layout()
plt.show()
