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

# Load data
messages_df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)

# Merge to get rep_num
df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')
df = df[df["role"] == "describer"].copy()

# Extract modifier/specificity features
def extract_mod_features(text):
    if pd.isna(text):
        return {
            "num_adjectives": 0,
            "num_noun_chunks": 0,
            "avg_np_len": 0.0,
            "num_modifiers": 0,
            "num_preps": 0
        }
    
    doc = nlp(text)
    adjectives = [t for t in doc if t.pos_ == "ADJ"]
    noun_chunks = list(doc.noun_chunks)
    preps = [t for t in doc if t.pos_ == "ADP" and t.dep_ == "prep"]
    modifiers = [t for t in doc if t.dep_ in {"amod", "nummod", "advmod"}]

    avg_np_len = sum(len(chunk) for chunk in noun_chunks) / len(noun_chunks) if noun_chunks else 0.0

    return {
        "num_adjectives": len(adjectives),
        "num_noun_chunks": len(noun_chunks),
        "avg_np_len": avg_np_len,
        "num_modifiers": len(modifiers),
        "num_preps": len(preps)
    }

# Apply to all messages
mod_features = df["text"].apply(extract_mod_features).apply(pd.Series)
df = pd.concat([df, mod_features], axis=1)

# Aggregate and reshape for plotting 
features_to_plot = ["num_adjectives", "num_noun_chunks", "avg_np_len", "num_modifiers", "num_preps"]
summary_df = df.groupby("rep_num")[features_to_plot].mean().reset_index()

# Rename for more readable labels
label_map = {
    "num_adjectives": "Average number of adjectives",
    "num_noun_chunks": "Average number of noun phrases",
    "avg_np_len": "Average noun phrase length",
    "num_modifiers": "Average number of modifiers (amod, advmod, nummod)",
    "num_preps": "Average number of prepositional modifiers"
}

long_df = pd.melt(summary_df, id_vars="rep_num", var_name="Feature", value_name="Average")
long_df["Feature"] = long_df["Feature"].map(label_map)

# Plot all features on a single line plot 
plt.figure(figsize=(12, 6))
sns.lineplot(data=long_df, x="rep_num", y="Average", hue="Feature", marker="o")
plt.title(f"{paper_name}: Modifier & Specificity Features Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Count / Length per Message")
plt.grid(True)
plt.tight_layout()
plt.legend(title="Feature", loc="upper right")
plt.show()
