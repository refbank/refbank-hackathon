import os
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

### read data
df_data1 = pd.read_csv("concreteness/brysbaert2014.csv")
df_data2 = pd.read_csv("concreteness/muraki2022.csv")

df_data1["expression"] = df_data1["Word"]
df_data1["concreteness"] = df_data1["Conc.M"]

df_data2["expression"] = df_data2["Expression"]
df_data2["concreteness"] = df_data2["Mean_C"]

df_data = pd.concat([df_data1[["expression", "concreteness"]], df_data2[["expression", "concreteness"]]]).reset_index(drop=True)

df_embeddings = model.encode(df_data["expression"].tolist(), show_progress_bar=True)

embed_df = pd.DataFrame(df_embeddings, columns=[f"dim_{i+1}" for i in range(df_embeddings.shape[1])])
embed_out = pd.concat([df_data, embed_df], axis=1)

embed_out.to_csv("concreteness/concreteness_embeddings.csv", index=False)