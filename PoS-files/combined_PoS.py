import pandas as pd
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load spaCy and tqdm
nlp = spacy.load("en_core_web_sm")
tqdm.pandas()

# SET DATASET NAME HERE
dataset_name = "hawkins2020_characterizing_cued" 

# File paths
message_path = f"harmonized_data/{dataset_name}/messages.csv"
trial_path = f"harmonized_data/{dataset_name}/trials.csv"

# Load and merge
messages = pd.read_csv(message_path)
trials = pd.read_csv(trial_path)
df = pd.merge(messages, trials[['trial_id', 'rep_num']], on='trial_id', how='left')
df = df[df["role"] == "describer"].copy()
df['text'] = df['text'].fillna("")

# Apply spaCy
df['spacy_doc'] = df['text'].progress_apply(nlp)

# ------------------------------
# Pronoun count
df['pronoun_count'] = df['spacy_doc'].apply(lambda doc: sum(1 for token in doc if token.pos_ == "PRON"))
pronoun_df = df.groupby("rep_num")["pronoun_count"].mean().reset_index()
pronoun_df["metric"] = "Pronoun Count"
pronoun_df = pronoun_df.rename(columns={"pronoun_count": "value"})

# ------------------------------
# Nominal proportion
def count_nominals(doc):
    nominals = [t for t in doc if t.pos_ in ["NOUN", "PROPN", "PRON"]]
    non_nominals = [t for t in doc if t.pos_ not in ["NOUN", "PROPN", "PRON"] and t.is_alpha]
    return len(nominals), len(non_nominals)

df[["num_nominals", "num_non_nominals"]] = df['spacy_doc'].apply(lambda doc: pd.Series(count_nominals(doc)))
df["total"] = df["num_nominals"] + df["num_non_nominals"]
df = df[df["total"] > 0]
df["prop_nominal"] = df["num_nominals"] / df["total"]
nominal_df = df.groupby("rep_num")["prop_nominal"].mean().reset_index()
nominal_df["metric"] = "Nominal Proportion"
nominal_df = nominal_df.rename(columns={"prop_nominal": "value"})

# ------------------------------
# Hedge count
hedge_words = {
    "maybe", "perhaps", "probably", "possibly", "seems", "i think", "i guess",
    "sort of", "kind of", "somewhat", "a little", "not sure", "might", "could",
    "likely", "appears", "looks like", "i feel like", "i suppose"
}
def count_hedges(text):
    text = text.lower()
    return sum(text.count(hw) for hw in hedge_words)

df["hedge_count"] = df["text"].apply(count_hedges)
hedge_df = df.groupby("rep_num")["hedge_count"].mean().reset_index()
hedge_df["metric"] = "Hedge Count"
hedge_df = hedge_df.rename(columns={"hedge_count": "value"})

# ------------------------------
# Combine and Plot
combined_df = pd.concat([pronoun_df, nominal_df, hedge_df], ignore_index=True)

plt.figure(figsize=(10, 6))
sns.lineplot(data=combined_df, x="rep_num", y="value", hue="metric", marker="o")
plt.title(f"{dataset_name}: Discourse Features Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Feature Value")
plt.grid(True)
plt.tight_layout()
plt.legend(title="Metric")
plt.show()
