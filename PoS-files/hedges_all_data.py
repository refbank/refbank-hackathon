import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# List of dataset folders
dataset_paths = [
    "harmonized_data/boyce2024_interaction",
    "harmonized_data/eliav2023_semantic",
    "harmonized_data/hawkins2019_continual",
    "harmonized_data/hawkins2020_characterizing_cued",
    "harmonized_data/hawkins2020_characterizing_uncued",
    "harmonized_data/hawkins2021_respect",
    "harmonized_data/hawkins2023_frompartners"

]

# Define hedge words
hedge_words = {
    "maybe", "perhaps", "probably", "possibly", "seems", "i think", "i guess",
    "sort of", "kind of", "somewhat", "a little", "not sure", "might", "could",
    "likely", "appears", "looks like", "i feel like", "i suppose"
}

# Function to count hedge words in a message
def count_hedges(text):
    if pd.isna(text):
        return 0
    text = text.lower()
    return sum(text.count(hedge) for hedge in hedge_words)

# Store all results in one list
all_hedge_data = []

for dataset_path in dataset_paths:
    message_path = os.path.join(dataset_path, "messages.csv")
    trial_path = os.path.join(dataset_path, "trials.csv")
    paper_name = os.path.basename(dataset_path)

    # Load data
    messages_df = pd.read_csv(message_path)
    trials_df = pd.read_csv(trial_path)

    # Merge and filter
    df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')
    df = df[df["role"] == "describer"].copy()
    df["hedge_count"] = df["text"].apply(count_hedges)

    # Aggregate
    hedge_by_rep = df.groupby("rep_num")["hedge_count"].mean().reset_index()
    hedge_by_rep["dataset"] = paper_name  # Label dataset

    all_hedge_data.append(hedge_by_rep)

# Combine into one DataFrame
combined_df = pd.concat(all_hedge_data, ignore_index=True)

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=combined_df, x="rep_num", y="hedge_count", hue="dataset", marker="o")
plt.title("Average Hedge Word Use Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Hedge Count per Message")
plt.grid(True)
plt.tight_layout()
plt.show()
