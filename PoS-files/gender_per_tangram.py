import pandas as pd
import spacy
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import cairosvg
import io
import os

# Load data
messages_path = "harmonized_data/hawkins2020_characterizing_cued/messages.csv"
trials_path = "harmonized_data/hawkins2020_characterizing_cued/trials.csv"

df = pd.read_csv(messages_path)
trials_df = pd.read_csv(trials_path)

# Merge trial info: include target
df = pd.merge(df, trials_df[["trial_id", "target"]], on="trial_id", how="left")
df = df[df["role"] == "describer"].copy()

# Keep only targets A–L
valid_targets = list("ABCDEFGHIJKL")
df = df[df["target"].isin(valid_targets)]

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Gendered term categories
gender_terms = {
    "male": ["man", "men", "boy", "boys", "he", "his", "him", "guy", "gentleman", "gentlemen"],
    "female": ["woman", "women", "girl", "girls", "she", "her", "hers", "lady", "ladies"],
    "neutral": ["person", "people", "child", "children", "individual", "thing", "it", "they", "them", "their"]
}
term_to_category = {word: cat for cat, words in gender_terms.items() for word in words}

# Count gendered terms by target
counts_by_target = defaultdict(lambda: Counter())

for _, row in df.iterrows():
    target = row["target"]
    text = str(row["text"]).lower()
    doc = nlp(text)
    for token in doc:
        word = token.text.lower()
        if word in term_to_category:
            category = term_to_category[word]
            counts_by_target[target][category] += 1

# Build dataframe from counts
records = []
for target, cat_counts in counts_by_target.items():
    for category, count in cat_counts.items():
        records.append({
            "target": target,
            "category": category,
            "count": count
        })

df_target_counts = pd.DataFrame(records)

# Ensure alphabetical target order
df_target_counts["target"] = pd.Categorical(df_target_counts["target"],
                                            categories=valid_targets,
                                            ordered=True)

# Define consistent order and colors
all_categories = ["male", "female", "neutral"]
fixed_palette = {
    "male": "#66c2a5",
    "female": "#fc8d62",
    "neutral": "#8da0cb"
}

# Set up grid
num_targets = len(valid_targets)
cols = 3
rows = (num_targets + cols - 1) // cols

fig = plt.figure(figsize=(15, rows * 4))
gs = gridspec.GridSpec(rows, cols, figure=fig)

for idx, target in enumerate(valid_targets):
    ax = fig.add_subplot(gs[idx])
    
    # Get data for target
    data = df_target_counts[df_target_counts["target"] == target].copy()

    # Fill in missing categories
    for cat in all_categories:
        if cat not in data["category"].values:
            data = pd.concat([data, pd.DataFrame({
                "target": [target],
                "category": [cat],
                "count": [0]
            })], ignore_index=True)

    # Barplot with fixed order and palette
    sns.barplot(
        data=data,
        x="category",
        y="count",
        hue="category",
        order=all_categories,
        palette=fixed_palette,
        legend=False,
        ax=ax
    )

    ax.set_title(f"Tangram {target}")
    ax.set_xlabel("")
    ax.set_ylabel("Count")

    # Try to embed image (SVG or fallback)
    svg_path = f"image_metada/images/page-{target.lower()}.svg"
    if os.path.exists(svg_path):
        try:
            png_data = cairosvg.svg2png(url=svg_path)
            image = Image.open(io.BytesIO(png_data))
            imagebox = OffsetImage(image, zoom=0.3)
            ab = AnnotationBbox(imagebox, (1.05, 0.7), xycoords='axes fraction', frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error rendering SVG for target {target}: {e}")
            ax.text(1.05, 0.7, "SVG\nnot\navailable", transform=ax.transAxes,
                    ha='center', va='center')
    else:
        ax.text(1.05, 0.7, "No SVG", transform=ax.transAxes,
                ha='center', va='center')

plt.tight_layout()
plt.show()
