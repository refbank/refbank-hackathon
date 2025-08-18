import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# File paths
message_path = "harmonized_data/hawkins2020_characterizing_cued/messages.csv"
trial_path = "harmonized_data/hawkins2020_characterizing_cued/trials.csv"
paper_name = os.path.basename(os.path.dirname(message_path))

# Load and merge
messages_df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)
df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')
df = df[df["role"] == "describer"].copy()

# Extract features
def extract_mod_features(text):
    if pd.isna(text):
        return {
            "num_adjectives": 0,
            "num_noun_chunks": 0,
            "avg_np_len": 0.0,
            "num_modifiers": 0,
            "num_preps": 0,
            "word_count": 0
        }
    doc = nlp(text)
    adjectives = [t for t in doc if t.pos_ == "ADJ"]
    noun_chunks = list(doc.noun_chunks)
    preps = [t for t in doc if t.pos_ == "ADP" and t.dep_ == "prep"]
    modifiers = [t for t in doc if t.dep_ in {"amod", "nummod", "advmod"}]
    words = [t for t in doc if t.is_alpha]

    avg_np_len = sum(len(chunk) for chunk in noun_chunks) / len(noun_chunks) if noun_chunks else 0.0

    return {
        "num_adjectives": len(adjectives),
        "num_noun_chunks": len(noun_chunks),
        "avg_np_len": avg_np_len,
        "num_modifiers": len(modifiers),
        "num_preps": len(preps),
        "word_count": len(words)
    }

# Apply
mod_features = df["text"].apply(extract_mod_features).apply(pd.Series)
df = pd.concat([df, mod_features], axis=1)

# Filter for valid rows
df = df[df["word_count"] > 0].copy()

# Normalize
df["adj_rate"] = df["num_adjectives"] / df["word_count"]
df["np_rate"] = df["num_noun_chunks"] / df["word_count"]
df["mod_rate"] = df["num_modifiers"] / df["word_count"]
df["prep_rate"] = df["num_preps"] / df["word_count"]
# avg_np_len is already a rate per noun phrase

# Melt long for plotting
label_map = {
    "adj_rate": "Adjectives per word",
    "np_rate": "Noun phrases per word",
    "avg_np_len": "Average noun phrase length",
    "mod_rate": "Modifiers per word (amod, advmod, nummod)",
    "prep_rate": "Prepositional modifiers per word"
}

long_df = pd.melt(
    df,
    id_vars="rep_num",
    value_vars=["adj_rate", "np_rate", "avg_np_len", "mod_rate", "prep_rate"],
    var_name="Feature",
    value_name="Value"
)
long_df["Feature"] = long_df["Feature"].map(label_map)

# Plot with CI
plt.figure(figsize=(12, 6))
sns.lineplot(data=long_df, x="rep_num", y="Value", hue="Feature", ci=95, marker="o")

plt.title(f"{paper_name}: Modifier & Specificity Feature Rates Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Rate (per word)")
plt.grid(True)
plt.tight_layout()
plt.legend(title="Feature", loc="upper right")
plt.show()

