import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load CLIP predictions 
clip_path = "data/stimulus_predictions/clip_stimulus_level_predictions_model-openai_clip-vit-large-patch14.csv"
df = pd.read_csv(clip_path)

# We're assuming labels and predictions are uppercase letters A-L
y_true = df["label"]
y_pred = df["prediction"]

# Ensure same ordering for matrix and axis labels
classes = sorted(list(set(y_true) | set(y_pred)))

# Build confusion matrix ===
cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")  # row-normalized

# Plot as heatmap 
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Tangram")
plt.ylabel("True Tangram")
plt.title("CLIP Confusion Matrix (Normalized by Row)")
plt.tight_layout()
plt.show()
