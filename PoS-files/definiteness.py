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
df = pd.merge(messages_df, trials_df[['trial_id', 'rep_num']], on='trial_id', how='left')

# Filter to describer messages
df = df[df["role"] == "describer"].copy()

# Count articles and words
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

# Apply
df[["definite_count", "indefinite_count", "word_count"]] = df["text"].apply(count_articles_and_words)
df = df[df["word_count"] > 0].copy()
df["definite_rate"] = df["definite_count"] / df["word_count"]
df["indefinite_rate"] = df["indefinite_count"] / df["word_count"]

# Melt to long format so we can plot both lines in one call
long_df = pd.melt(
    df,
    id_vars=["rep_num"],
    value_vars=["definite_rate", "indefinite_rate"],
    var_name="Article Type",
    value_name="Rate"
)
long_df["Article Type"] = long_df["Article Type"].map({
    "definite_rate": "‘the’ per word",
    "indefinite_rate": "‘a/an’ per word"
})

# Plot with CI
plt.figure(figsize=(10, 6))
sns.lineplot(data=long_df, x="rep_num", y="Rate", hue="Article Type", ci=95, marker="o")

plt.title(f"{paper_name}: Article Usage Rate per Word Over Repetitions")
plt.xlabel("Repetition Number")
plt.ylabel("Average Article Rate per Word")
plt.legend(title="Article Type")
plt.grid(True)
plt.tight_layout()
plt.show()
