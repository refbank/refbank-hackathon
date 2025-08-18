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
    "harmonized_data/hawkins2023_frompartners",
    "harmonized_data/leung2024_scaffolding",
    "harmonized_data/mankewitz2025_compositional"
]

# Hedge words
hedge_words = {
    "maybe", "perhaps", "probably", "possibly", "seems", "i think", "i guess",
    "sort of", "kind of", "somewhat", "a little", "not sure", "might", "could",
    "likely", "appears", "looks like", "i feel like", "i suppose"
}

# Count hedge words and words
def count_hedges_and_words(text):
    if pd.isna(text):
        return pd.Series({"hedge_count": 0, "word_count": 0})
    text_lower = text.lower()
    hedge_count = sum(text_lower.count(hedge) for hedge in hedge_words)
    word_count = len([w for w in text_lower.split() if w.isalpha()])
    return pd.Series({"hedge_count": hedge_count, "word_count": word_count})

# Store results
all_hedge_data = []

for dataset_path in dataset_paths:
    message_path = os.path.join(dataset_path, "messages.csv")
    trial_path = os.path.join(dataset_path, "trials.csv")
    paper_name = os.path.basename(dataset_path)

    # Load
    messages_df = pd.read_csv(message_path)
    trials_df = pd.read_csv(trial_path)

    # Merge and filter
    df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')
    df = df[df["role"] == "describer"].copy()

    # Count hedges + words
    counts_df = df["text"].apply(count_hedges_and_words)
    df = pd.concat([df, counts_df], axis=1)

    # Remove empty messages
    df = df[df["word_count"] > 0].copy()

    # Normalize: hedges per word
    df["hedge_rate"] = df["hedge_count"] / df["word_count"]

    # Aggregate: mean hedge rate & preserve for CI plotting
    df["dataset"] = paper_name
    all_hedge_data.append(df[["rep_num", "hedge_rate", "dataset"]])

# Combine into one DataFrame
combined_df = pd.concat(all_hedge_data, ignore_index=True)

# Save raw per-message normalized data (useful for future)
combined_df.to_csv("all_hedge_rates.csv", index=False)

# Plot with seaborn 95% CI
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=combined_df,
    x="rep_num", y="hedge_rate",
    hue="dataset", marker="o",
    errorbar=("ci", 95)  # 95% confidence interval
)
plt.title("Average Hedge Word Rate (per word) Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Hedge Rate per Word")
plt.grid(True)
plt.tight_layout()
plt.show()
