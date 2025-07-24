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

# Extract paper name
paper_name = os.path.basename(os.path.dirname(message_path))

# Load data
messages_df = pd.read_csv(message_path)
trials_df = pd.read_csv(trial_path)
df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')

# Filter to describer messages only
df = df[df["role"] == "describer"].copy()

# Count articles and word count per message
def count_articles_and_words(text):
    if pd.isna(text):
        return pd.Series({"definite_count": 0, "indefinite_count": 0, "word_count": 0})
    doc = nlp(text.lower())
    definite = sum(1 for token in doc if token.text == "the" and token.tag_ == "DT")
    indefinite = sum(1 for token in doc if token.text in {"a", "an"} and token.tag_ == "DT")
    words = sum(1 for token in doc if token.is_alpha)
    return pd.Series({
        "definite_count": definite,
        "indefinite_count": indefinite,
        "word_count": words
    })

# Apply the function
df[["definite_count", "indefinite_count", "word_count"]] = df["text"].apply(count_articles_and_words)

# Filter out messages with 0 words to avoid divide-by-zero
df = df[df["word_count"] > 0].copy()

# Normalize to get rates per word
df["definite_rate"] = df["definite_count"] / df["word_count"]
df["indefinite_rate"] = df["indefinite_count"] / df["word_count"]

# Average rates by repetition
article_by_rep = df.groupby("rep_num")[["definite_rate", "indefinite_rate"]].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=article_by_rep, x="rep_num", y="definite_rate", marker="o", label="‘the’ per word")
sns.lineplot(data=article_by_rep, x="rep_num", y="indefinite_rate", marker="o", label="‘a/an’ per word")

plt.title(f"{paper_name}: Article Usage Rate per Word Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Article Rate per Word")
plt.legend(title="Article Type")
plt.grid(True)
plt.tight_layout()
plt.show()
