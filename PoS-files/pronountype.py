import pandas as pd
import spacy
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
message_path = "harmonized_data/hawkins2020_characterizing_cued/messages.csv"
trial_path = "harmonized_data/hawkins2020_characterizing_cued/trials.csv"

# Extract paper name from path
paper_name = os.path.basename(os.path.dirname(message_path))

df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)

# Load data
messages = pd.read_csv(message_path)
trials = pd.read_csv(trial_path)

# Merge to get rep_num
df = pd.merge(df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')
df = df[df["role"] == "describer"].copy()

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Pronoun category mapping
pronoun_categories = {
    "i": "personal", "you": "personal", "he": "personal", "she": "personal", "it": "personal",
    "we": "personal", "they": "personal", "me": "personal", "him": "personal", "her": "personal",
    "us": "personal", "them": "personal",

    "my": "possessive", "your": "possessive", "his": "possessive", "her": "possessive",
    "its": "possessive", "our": "possessive", "their": "possessive",

    "this": "demonstrative", "that": "demonstrative", "these": "demonstrative", "those": "demonstrative",

    "myself": "reflexive", "yourself": "reflexive", "himself": "reflexive", "herself": "reflexive",
    "itself": "reflexive", "ourselves": "reflexive", "themselves": "reflexive",

    "someone": "indefinite", "anyone": "indefinite", "everyone": "indefinite", "no one": "indefinite",
    "something": "indefinite", "anything": "indefinite", "nothing": "indefinite", "everything": "indefinite",

    "who": "relative", "whom": "relative", "which": "relative", "that": "relative"
}

# Count pronouns per category per rep
category_by_rep = defaultdict(lambda: Counter())

for _, row in df.iterrows():
    text = str(row["text"]).lower()
    rep = row["rep_num"]
    doc = nlp(text)
    for token in doc:
        word = token.text.lower()
        if token.pos_ == "PRON" or word in pronoun_categories:
            category = pronoun_categories.get(word, "other")
            category_by_rep[rep][category] += 1

# Convert to DataFrame
records = []
for rep_num, cat_counts in category_by_rep.items():
    for cat, count in cat_counts.items():
        records.append({
            "rep_num": rep_num,
            "category": cat,
            "count": count
        })

df_counts = pd.DataFrame(records)

# Normalize to average per message (optional)
rep_message_counts = df.groupby("rep_num").size().rename("n_messages").reset_index()
df_counts = df_counts.merge(rep_message_counts, on="rep_num")
df_counts["avg_count_per_msg"] = df_counts["count"] / df_counts["n_messages"]

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_counts, x="rep_num", y="avg_count_per_msg", hue="category", marker="o")
plt.title(f"{paper_name}: Pronoun Category Use Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Avg Count per Message")
plt.grid(True)
plt.tight_layout()
plt.show()
