import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

features_df = pd.read_csv("output/features_handcrafted.csv")
meta_df = pd.read_csv("selected_metadata.csv")

# Merge on filename
df = features_df.merge(
    meta_df[["Image Index", "Finding Labels"]],
    left_on="filename",
    right_on="Image Index",
)
df = df.drop(columns=["Image Index"])

# Extract features and labels
X = df.drop(columns=["filename", "Finding Labels"]).values
y = df["Finding Labels"].values

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette="tab10", s=50, alpha=0.8
)

plt.title("t-SNE of Handcrafted Features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Finding", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("output/tsne_handcrafted.png", dpi=300)
plt.show()
