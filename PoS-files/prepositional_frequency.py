import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import spacy

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

nlp = spacy.load("en_core_web_sm")

def preps_and_words_series(texts):
    """Vectorized helper: parse a list of texts with nlp.pipe and return two lists:
       prep_count and word_count per text."""
    prep_counts, word_counts = [], []
    for doc in nlp.pipe((t if isinstance(t, str) else "" for t in texts), batch_size=128):
        prep_counts.append(sum(1 for tok in doc if tok.dep_ == "prep"))
        word_counts.append(sum(1 for tok in doc if tok.is_alpha))
    return prep_counts, word_counts

all_rows = []

for dataset_path in dataset_paths:
    msg_path = os.path.join(dataset_path, "messages.csv")
    tri_path = os.path.join(dataset_path, "trials.csv")
    dataset_name = os.path.basename(dataset_path)

    # Load & merge
    messages_df = pd.read_csv(msg_path)
    trials_df   = pd.read_csv(tri_path)
    df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')
    df = df[df["role"] == "describer"].copy()

    # Count preps/words with spaCy (fast via pipe)
    prep_counts, word_counts = preps_and_words_series(df["text"].tolist())
    df["prep_count"] = prep_counts
    df["word_count"] = word_counts

    # Drop empty messages and compute rate
    df = df[df["word_count"] > 0].copy()
    df["prep_rate"] = df["prep_count"] / df["word_count"]
    df["dataset"] = dataset_name

    all_rows.append(df[["rep_num", "prep_rate", "dataset"]])

combined = pd.concat(all_rows, ignore_index=True)

combined.to_csv("all_preposition_rates.csv", index=False)

plt.figure(figsize=(12, 7))
sns.lineplot(
    data=combined,
    x="rep_num", y="prep_rate",
    hue="dataset", marker="o",
    errorbar=("ci", 95)  # 95% bootstrapped CI
)
plt.title("Prepositional Phrase Rate (per word) Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Prepositions per Word")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
