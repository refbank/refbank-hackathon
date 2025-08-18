import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# All datasets
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

# Question detection
wh_words = {"what","where","who","when","why","how","which","whose"}
def is_question(text: str) -> int:
    if not isinstance(text, str):
        return 0
    t = text.lower().strip()
    if t.endswith("?"):
        return 1
    first = re.split(r"\W+", t)[0] if t else ""
    return int(first in wh_words)

# Build combined dataframe
all_rows = []
for dataset_path in dataset_paths:
    msg_path = os.path.join(dataset_path, "messages.csv")
    tri_path = os.path.join(dataset_path, "trials.csv")
    dataset_name = os.path.basename(dataset_path)

    messages_df = pd.read_csv(msg_path)
    trials_df   = pd.read_csv(tri_path)

    df = pd.merge(messages_df, trials_df[["trial_id","rep_num"]], on="trial_id", how="left")
    df = df[df["role"] == "describer"].copy()
    df["is_question"] = df["text"].apply(is_question)
    df["dataset"] = dataset_name

    all_rows.append(df[["rep_num","is_question","dataset"]])

combined = pd.concat(all_rows, ignore_index=True)

# Plot shaded 95% CI per dataset
plt.figure(figsize=(12, 7))
sns.lineplot(
    data=combined,
    x="rep_num", y="is_question",
    hue="dataset", marker="o",
    errorbar=("ci", 95)  # Seaborn automatically shades CI
)
plt.title("Question Frequency (proportion of describer messages) Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Proportion of Questions")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
