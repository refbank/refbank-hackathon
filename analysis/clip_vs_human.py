import pandas as pd
import matplotlib.pyplot as plt

# Load and clean human data ===
human_path = "data/tgmatcheryoked-trials.csv"  
df_human = pd.read_csv(human_path)

# Drop rows where 'correct' is NaN
df_human = df_human[df_human["correct"].notna()]

# Convert 'correct' to boolean
df_human["correct"] = df_human["correct"].astype(str).str.lower().map({"true": True, "false": False})

# Choose the appropriate repNum column (update this if needed)
rep_column = "orig_repNum" if "orig_repNum" in df_human.columns else "repNum"

# Drop any rows with missing repNum
df_human = df_human[df_human[rep_column].notna()]
df_human[rep_column] = df_human[rep_column].astype(int)

# Compute human accuracy per repNum
human_accuracy = df_human.groupby(rep_column)["correct"].mean().reset_index()
human_accuracy.columns = ["repNum", "human_accuracy"]

print("Human accuracy by repNum:")
print(human_accuracy)

# Load CLIP predictions 
clip_path = "data/stimulus_predictions/clip_stimulus_level_predictions_model-openai_clip-vit-large-patch14.csv"
df_clip = pd.read_csv(clip_path)

# Compute CLIP correctness
df_clip["correct"] = df_clip["prediction"] == df_clip["label"]

# Compute CLIP accuracy per repNum
clip_accuracy = df_clip.groupby("repNum")["correct"].mean().reset_index()
clip_accuracy.columns = ["repNum", "clip_accuracy"]

print("\nCLIP accuracy by repNum:")
print(clip_accuracy)

# Merge and plot
merged = pd.merge(human_accuracy, clip_accuracy, on="repNum", how="inner")

# Sort by repNum for plotting
merged = merged.sort_values("repNum")

# STEP 4: Plot
plt.figure(figsize=(8, 5))
plt.plot(merged["repNum"], merged["human_accuracy"], marker="o", label="Human", linewidth=2)
plt.plot(merged["repNum"], merged["clip_accuracy"], marker="o", label="CLIP", linewidth=2)
plt.xlabel("Repetition Number (repNum)")
plt.ylabel("Accuracy")
plt.title("Human vs CLIP Accuracy by Repetition Number")
plt.ylim(0, 1.0)
plt.xticks(merged["repNum"].unique())
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
