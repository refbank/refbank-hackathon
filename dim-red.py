# %%
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# %%
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set pandas display options to show more text
pd.set_option('display.max_colwidth', 150)
pd.set_option('display.max_rows', 20)

# %%
DATA_LOC = "harmonized_data"
dirs = os.listdir(DATA_LOC)

all_msg = pd.DataFrame()
all_embed = pd.DataFrame()
for dir in sorted(dirs):
    if os.path.isdir(os.path.join(DATA_LOC, dir)):
        print(f"Reading {dir}...")

        df_msg = pd.read_csv(os.path.join(DATA_LOC, dir, "messages.csv"), on_bad_lines='skip')
        df_trials = pd.read_csv(os.path.join(DATA_LOC, dir, "trials.csv"))
        df_con = pd.read_csv(os.path.join(DATA_LOC, dir, "conditions.csv"))
        df_out = df_trials.merge(df_con, on="condition_id", how="left") \
                            .merge(df_msg, on="trial_id", how="left")
        all_msg = pd.concat([all_msg, df_out])

        df_embed = pd.read_csv(os.path.join(DATA_LOC, dir, "embeddings.csv"))
        all_embed = pd.concat([all_embed, df_embed])

all_msg["text"] = all_msg["text"].fillna("").astype(str)

all_msg_concat = all_msg[all_msg["role"] == "describer"] \
                        .groupby(["paper_id", "game_id", "trial_id"])["text"] \
                        .apply(", ".join) \
                        .reset_index()

# %%
# i want to do a dim reduction on the embeddings.
# the way to do this is:
# run a PCA on the embeddings
# then show the df of scores (joined to the all_msg_concat df), arranged in descending order of the first PC
# we will then create an "interpretable dimension" by providing a list of words that are associated with the first PC
# we will embed these using `model`, then show the projections of the embeddings onto the interpretable dimension
# we then subtract the projections from the embeddings
# then we re-run a PCA on the residuals
# and so on until we have a desired number of dimensions

def create_interpretable_dimension(embeddings, interpretable_words, model):
    """Create an interpretable dimension from a list of words."""
    # Get embeddings for interpretable words
    word_embeddings = model.encode(interpretable_words)
    # Average the word embeddings to get the interpretable dimension
    interpretable_dim = word_embeddings.mean(axis=0)
    # Normalize the dimension
    interpretable_dim = interpretable_dim / np.linalg.norm(interpretable_dim)
    return interpretable_dim

def project_and_subtract(embeddings, dimension):
    """Project embeddings onto dimension and subtract the projections."""
    # Project embeddings onto dimension
    projections = np.dot(embeddings, dimension)
    # Reshape for broadcasting
    projections = projections.reshape(-1, 1)
    # Subtract projections from embeddings
    residuals = embeddings - projections * dimension
    return residuals, projections

# Get the embedding columns
embed_cols = [col for col in all_embed.columns if col.startswith('dim_')]
X = all_embed[embed_cols].values

# Number of dimensions to extract
n_dims = 10
interpretable_dims = []
projections_list = []
dimension_names = []
residuals = X.copy()
cumulative_variance = 0
total_variance = np.var(X, axis=0).sum()
dimension_count = 0

while dimension_count < n_dims:
    print(f"\nExtracting dimension {dimension_count + 1} of {n_dims}")
    
    # Run PCA on residuals to find direction
    pca = PCA(n_components=5)
    pca.fit(residuals)
    
    # Get the principal components
    pcs = pca.components_
    
    # Join with message data and sort by PC scores
    scores = np.dot(residuals, pcs.T)  # This will give us scores for all 5 PCs
    df_scores = pd.DataFrame({
        'paper_id': all_embed['paper_id'],
        'game_id': all_embed['game_id'],
        'trial_id': all_embed['trial_id']
    })
    
    # Add scores for each PC
    for i in range(5):
        df_scores[f'pc{i+1}_score'] = scores[:, i]
    
    # Merge with message data
    df_scores = df_scores.merge(all_msg_concat, on=['paper_id', 'game_id', 'trial_id'])
    
    # Display examples for each PC
    for i in range(5):
        print(f"\nPC {i+1} - Top examples:")
        print(df_scores.sort_values(f'pc{i+1}_score', ascending=False)[['text', f'pc{i+1}_score']].head(10))
        print(f"\nPC {i+1} - Bottom examples:")
        print(df_scores.sort_values(f'pc{i+1}_score', ascending=False)[['text', f'pc{i+1}_score']].tail(10))
    
    # Ask for interpretable words
    print("\nPlease provide a list of words that capture this dimension (comma-separated):")
    interpretable_words = input().split(',')
    interpretable_words = [w.strip() for w in interpretable_words]
    
    # Create interpretable dimension
    interpretable_dim = create_interpretable_dimension(residuals, interpretable_words, model)
    
    # Calculate variance explained by this interpretable dimension
    projections = np.dot(residuals, interpretable_dim)
    variance_explained = np.var(projections) / total_variance
    cumulative_variance += variance_explained
    
    # Display variance metrics
    print(f"\nVariance explained by this dimension: {variance_explained:.3f}")
    print(f"Cumulative variance explained: {cumulative_variance:.3f}")
    
    # Ask whether to accept this dimension
    print("\nDo you want to keep this dimension? (y/n):")
    keep_dimension = input().lower().strip() == 'y'
    
    if keep_dimension:
        # Store the first word as the dimension name
        dimension_names.append(interpretable_words[0])
        interpretable_dims.append(interpretable_dim)
        projections_list.append(projections.reshape(-1, 1))
        
        # Project and subtract
        residuals = residuals - projections.reshape(-1, 1) * interpretable_dim
        dimension_count += 1
    else:
        print("Skipping this dimension...")
        # If we reject the dimension, we don't update residuals or increment count
        continue

# Create final dataframe with all projections
final_df = pd.DataFrame({
    'paper_id': all_embed['paper_id'],
    'game_id': all_embed['game_id'],
    'trial_id': all_embed['trial_id']
})

for i, (proj, name) in enumerate(zip(projections_list, dimension_names)):
    final_df[name] = proj.flatten()

# Merge with message data
final_df = final_df.merge(all_msg_concat, on=['paper_id', 'game_id', 'trial_id'])

# Display final results
print("\nFinal dimensions:")
print(final_df[['text'] + dimension_names].head())

# Save results
final_df.to_csv('interpretable_dimensions.csv', index=False)
print("\nResults saved to interpretable_dimensions.csv")
