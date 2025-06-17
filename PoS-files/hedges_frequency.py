import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths
message_path = "harmonized_data/boyce2024_interaction/messages.csv"
trial_path = "harmonized_data/boyce2024_interaction/trials.csv"

# Extract paper name from the folder name
paper_name = os.path.basename(os.path.dirname(message_path))

# Load data
messages_df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)

# Merge to get rep_num
df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')

# Filter to describer messages only
df = df[df["role"] == "describer"].copy()

# Define hedge words
hedge_words = {
    "maybe", "perhaps", "probably", "possibly", "seems", "i think", "i guess",
    "sort of", "kind of", "somewhat", "a little", "not sure", "might", "could",
    "likely", "appears", "looks like", "i feel like", "i suppose"
}

# Count hedge words in each message
def count_hedges(text):
    if pd.isna(text):
        return 0
    text = text.lower()
    return sum(text.count(hedge) for hedge in hedge_words)

df["hedge_count"] = df["text"].apply(count_hedges)

# Aggregate: average hedge count per rep_num
hedge_by_rep = df.groupby("rep_num")["hedge_count"].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=hedge_by_rep, x="rep_num", y="hedge_count", marker="o")
plt.title(f"{paper_name}: Average Hedge Word Use Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Hedge Count per Message")
plt.grid(True)
plt.tight_layout()
plt.show()
