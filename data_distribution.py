import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("youtube_comments_with_3sentiment.csv")  
category_column = 'category'       


category_counts = df[category_column].dropna().value_counts()
label_to_index = {label: idx for idx, label in enumerate(category_counts.index)}
index_to_label = {v: k for k, v in label_to_index.items()}

labels = list(label_to_index.keys())
counts = list(category_counts.values)
indices = list(label_to_index.values())


plt.figure(figsize=(10, 6))
sns.barplot(x=indices, y=counts, palette="Set2")


for i, count in enumerate(counts):
    plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=9)

plt.xlabel("Category Index")
plt.ylabel("Count")
plt.title("Category Distribution")
plt.xticks(rotation=0)
plt.tight_layout()

plt.gcf().text(0.77, 0.91, 'Category : Label value', fontsize=12, fontweight='bold', va='center', ha='left')


mapping_text = "\n".join([f"{label}: {idx}" for idx, label in index_to_label.items()])
plt.gcf().text(0.8, 0.7, mapping_text, fontsize=9, va='center', ha='left',
               bbox=dict(facecolor='white', edgecolor='gray', alpha=0))


plt.savefig("category_distribution.png", bbox_inches='tight')
print("Saved as category_distribution.png")
plt.show()
