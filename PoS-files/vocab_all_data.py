import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Content POS tags
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

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

all_game_rows = []

for dataset_path in dataset_paths:
    message_path = os.path.join(dataset_path, "messages.csv")
    trial_path = os.path.join(dataset_path, "trials.csv")
    dataset_name = os.path.basename(dataset_path)

    # Load data
    messages_df = pd.read_csv(message_path)
    trials_df = pd.read_csv(trial_path)

    # Merge rep_num + game_id
    df = pd.merge(
        messages_df,
        trials_df[["trial_id", "rep_num", "game_id"]],
        on="trial_id",
        how="left"
    )

    # Use both matcher and describer
    df = df[df["role"].isin(["describer", "matcher"])].copy()
    df["text"] = df["text"].fillna("")

    # Parse texts efficiently
    docs = list(nlp.pipe(df["text"].tolist(), batch_size=128))

    def get_content_words(doc):
        return [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in CONTENT_POS and token.is_alpha
        ]

    def get_all_words(doc):
        return [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
        ]

    df["content_words"] = [get_content_words(doc) for doc in docs]
    df["all_words"] = [get_all_words(doc) for doc in docs]

    # Aggregate per game and repetition
    rows = []
    for (game_id, rep_num), group in df.groupby(["game_id", "rep_num"]):
        content_words = [w for sublist in group["content_words"] for w in sublist]
        all_words = [w for sublist in group["all_words"] for w in sublist]

        if len(all_words) == 0:
            continue

        content_vocab_size = len(set(content_words))
        content_total_words = len(content_words)
        content_ttr = (
            content_vocab_size / content_total_words
            if content_total_words > 0 else None
        )

        vocab_size_all = len(set(all_words))
        total_words_all = len(all_words)

        rows.append({
            "game_id": game_id,
            "rep_num": rep_num,
            "dataset": dataset_name,

            # Existing content-word measures
            "content_vocab_size": content_vocab_size,
            "content_total_words": content_total_words,
            "content_ttr": content_ttr,

            # New all-word measures
            "vocab_size_all": vocab_size_all,
            "total_words_all": total_words_all
        })

    game_df = pd.DataFrame(rows)

    # Normalize all-word curves within each game using rep 1
    rep1 = game_df[game_df["rep_num"] == 1][
        ["game_id", "vocab_size_all", "total_words_all"]
    ].rename(columns={
        "vocab_size_all": "vocab_size_all_rep1",
        "total_words_all": "total_words_all_rep1"
    })

    game_df = game_df.merge(rep1, on="game_id", how="left")

    game_df["norm_vocab_size_all"] = (
        game_df["vocab_size_all"] / game_df["vocab_size_all_rep1"]
    )
    game_df["norm_total_words_all"] = (
        game_df["total_words_all"] / game_df["total_words_all_rep1"]
    )

    all_game_rows.append(game_df)

# Combine all datasets
combined_df = pd.concat(all_game_rows, ignore_index=True)

# Raw results
combined_df.to_csv("all_vocab_data.csv", index=False)

# Plot 1: Raw content-word vocab size
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=combined_df,
    x="rep_num",
    y="content_vocab_size",
    hue="dataset",
    marker="o",
    errorbar=("ci", 95)
)
plt.title("Vocabulary Size (Content Words) Over Repetitions")
plt.xlabel("Repetition")
plt.ylabel("Unique Content Words")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Existing content-word Normalized
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=combined_df,
    x="rep_num",
    y="content_ttr",
    hue="dataset",
    marker="o",
    errorbar=("ci", 95)
)
plt.title("Vocabulary Diversity (Content Words, Normalized) Over Repetitions")
plt.xlabel("Repetition")
plt.ylabel("Type-Token Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Existing per-game content-word vocab size
example_dataset = "hawkins2020_characterizing_cued"
subset = combined_df[combined_df["dataset"] == example_dataset]

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=subset,
    x="rep_num",
    y="content_vocab_size",
    units="game_id",
    estimator=None,
    alpha=0.25
)
sns.lineplot(
    data=subset,
    x="rep_num",
    y="content_vocab_size",
    color="black",
    linewidth=2,
    label="Average"
)
plt.title(f"{example_dataset}: Content Vocabulary Size Per Game")
plt.xlabel("Repetition")
plt.ylabel("Unique Content Words")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 4: Existing per-game content-word Normalized
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=subset,
    x="rep_num",
    y="content_ttr",
    units="game_id",
    estimator=None,
    alpha=0.25
)
sns.lineplot(
    data=subset,
    x="rep_num",
    y="content_ttr",
    color="black",
    linewidth=2,
    label="Average"
)
plt.title(f"{example_dataset}: Content Vocabulary Diversity (Normalized) Per Game")
plt.xlabel("Repetition")
plt.ylabel("Type-Token Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 5: Raw all-word vocabulary size
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=combined_df,
    x="rep_num",
    y="vocab_size_all",
    hue="dataset",
    marker="o",
    errorbar=("ci", 95)
)
plt.title("Vocabulary Size (All Words) Over Repetitions")
plt.xlabel("Repetition")
plt.ylabel("Unique Words")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 6: Raw total words (all words)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=combined_df,
    x="rep_num",
    y="total_words_all",
    hue="dataset",
    marker="o",
    errorbar=("ci", 95)
)
plt.title("Total Words (All Words) Over Repetitions")
plt.xlabel("Repetition")
plt.ylabel("Total Words")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 7: Normalized comparison
# unique words vs total words for one dataset
compare_subset = combined_df[combined_df["dataset"] == example_dataset].copy()

compare_long = compare_subset.melt(
    id_vars=["game_id", "rep_num", "dataset"],
    value_vars=["norm_vocab_size_all", "norm_total_words_all"],
    var_name="measure",
    value_name="value"
)

measure_labels = {
    "norm_vocab_size_all": "Normalized Unique Words",
    "norm_total_words_all": "Normalized Total Words"
}
compare_long["measure"] = compare_long["measure"].map(measure_labels)

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=compare_long,
    x="rep_num",
    y="value",
    hue="measure",
    marker="o",
    errorbar=("ci", 95)
)
plt.title(f"{example_dataset}: Normalized Unique Words vs Total Words")
plt.xlabel("Repetition")
plt.ylabel("Value (relative to repetition 1)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 8: Normalized comparison across datasets
compare_long_all = combined_df.melt(
    id_vars=["game_id", "rep_num", "dataset"],
    value_vars=["norm_vocab_size_all", "norm_total_words_all"],
    var_name="measure",
    value_name="value"
)
compare_long_all["measure"] = compare_long_all["measure"].map(measure_labels)

g = sns.FacetGrid(
    compare_long_all,
    col="dataset",
    col_wrap=3,
    hue="measure",
    sharey=True,
    height=3.5
)
g.map_dataframe(
    sns.lineplot,
    x="rep_num",
    y="value",
    marker="o",
    errorbar=("ci", 95)
)
g.add_legend()
g.set_axis_labels("Repetition", "Value (relative to repetition 1)")
g.set_titles("{col_name}")
plt.subplots_adjust(top=0.9)
g.figure.suptitle("Normalized Unique Words vs Total Words Across Datasets")
plt.show()