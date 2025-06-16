import pandas as pd
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Load spaCy
nlp = spacy.load("en_core_web_sm")
tqdm.pandas()

# File paths (replace with the dataset file path)
message_path = "harmonized_data/hawkins2019_continual/messages.csv"
trial_path = "harmonized_data/hawkins2019_continual/trials.csv"

# Extract paper name from path
paper_name = os.path.basename(os.path.dirname(message_path))

# Load data
messages = pd.read_csv(message_path)
trials = pd.read_csv(trial_path)

# Merge to get rep_num
merged = pd.merge(messages, trials[['trial_id', 'rep_num']], on='trial_id', how='left')

# Filter for describers only
merged = merged[merged['role'] == 'describer'].copy()

# Apply spaCy
merged['spacy_doc'] = merged['text'].fillna("").progress_apply(nlp)

# Extract pronouns
def extract_pronouns(doc):
    return [token.text.lower() for token in doc if token.pos_ == "PRON"]

merged['pronouns'] = merged['spacy_doc'].apply(extract_pronouns)
merged['pronoun_count'] = merged['pronouns'].apply(len)

# Compute averages
avg_pronouns_per_rep = (
    merged.groupby('rep_num')['pronoun_count']
    .mean()
    .reset_index(name='avg_pronoun_count')
    .sort_values('rep_num')
)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(avg_pronouns_per_rep['rep_num'], avg_pronouns_per_rep['avg_pronoun_count'], marker='o')
plt.title(f"{paper_name}: Average Pronoun Frequency per Repetition")
plt.xlabel("Repetition Number")
plt.ylabel("Average # of Pronouns per Message")
plt.grid(True)
plt.tight_layout()
plt.show()
