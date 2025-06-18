import pandas as pd
import spacy
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths
message_path = "hawkins2020_characterizing_uncued/messages.csv"
trial_path = "hawkins2020_characterizing_uncued/trials.csv"
paper_name = os.path.basename(os.path.dirname(message_path))

# Load data
df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)
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
personal_pronoun_counts = Counter()

for _, row in df.iterrows():
    text = str(row["text"]).lower()
    rep = row["rep_num"]
    doc = nlp(text)
    for token in doc:
        word = token.text.lower()
        if token.pos_ == "PRON" or word in pronoun_categories:
            category = pronoun_categories.get(word, "other")
            category_by_rep[rep][category] += 1
            if category == "personal":
                personal_pronoun_counts[word] += 1

# Convert to DataFrame for lineplot
records = []
for rep_num, cat_counts in category_by_rep.items():
    for cat, count in cat_counts.items():
        records.append({
            "rep_num": rep_num,
            "category": cat,
            "count": count
        })

df_counts = pd.DataFrame(records)

# Normalize by number of messages per repetition
rep_message_counts = df.groupby("rep_num").size().rename("n_messages").reset_index()
df_counts = df_counts.merge(rep_message_counts, on="rep_num")
df_counts["avg_count_per_msg"] = df_counts["count"] / df_counts["n_messages"]

# Define consistent color palette for categories
category_palette = {
    "personal": "#66c2a5",
    "possessive": "#fc8d62",
    "demonstrative": "#8da0cb",
    "reflexive": "#e78ac3",
    "indefinite": "#a6d854",
    "relative": "#ffd92f",
    "other": "#e5c494"
}

# Plot pronoun category use over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_counts, x="rep_num", y="avg_count_per_msg",
             hue="category", palette=category_palette, marker="o")
plt.title(f"{paper_name}: Pronoun Category Use Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Avg Count per Message")
plt.grid(True)
plt.tight_layout()
plt.legend(title="Category")
plt.show()

# Sort by descending count
df_personal = pd.DataFrame({
    "pronoun": list(personal_pronoun_counts.keys()),
    "count": list(personal_pronoun_counts.values())
}).sort_values("count", ascending=False)

# Generate consistent color palette (in this order)
pronoun_order = df_personal["pronoun"].tolist()
colors = sns.color_palette("Blues_d", n_colors=len(pronoun_order))
pronoun_palette = dict(zip(pronoun_order, colors))

# Plot personal pronouns
plt.figure(figsize=(10, 5))
sns.barplot(data=df_personal, x="pronoun", y="count",
            palette=pronoun_palette, order=pronoun_order)
plt.title(f"{paper_name}: Personal Pronoun Frequencies")
plt.xlabel("Pronoun")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
