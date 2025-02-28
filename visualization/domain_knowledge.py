import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample dataset (Ensure to replace with actual data)
data = {
    "Method": ["MATS (Ours)", "DAILSQL(SC)", "CodeS-15B", "CodeS-7B", "REDSQL-3B\n+NatSQL", "REDSQL-3B", "Graphix\n+PICARD"],
    "College": [84.0, 79.6, 82.4, 83.3, 80.6, 83.3, 78.7],
    "Competition": [92.0, 79.0, 85.5, 82.3, 80.6, 83.9, 82.3],
    "Geography": [71.0, 76.7, 75.0, 75.8, 52.5, 65.0, 64.2],
    "Social": [95.0, 83.9, 83.9, 82.1, 76.8, 80.4, 82.1],
    "Transportation": [97.0, 85.0, 88.8, 87.5, 86.3, 80.0, 98.8],
    "Overall": [87.1, 83.6, 84.9, 85.4, 84.1, 81.8, 80.9]
}

# DB Count Data (Ensure to replace with actual data)
db_count = {"College": 10, "Competition": 5, "Geography": 3, "Social": 2, "Transportation": 12}

# Convert to DataFrame
df = pd.DataFrame(data)
df.set_index("Method", inplace=True)

# Transpose DataFrame to swap axes
df = df.T

# Create a figure with two side-by-side subplots, adjusting colors and layout
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3.5), gridspec_kw={'width_ratios': [4, 0.8]})

# Create the heatmap in the first subplot with a colormap similar to the reference image
sns.heatmap(df, annot=True, cmap="YlGnBu", linewidths=0.5, fmt=".1f", cbar=False, ax=axes[0])
axes[0].set_xlabel("", fontsize=8)
axes[0].set_ylabel("DB Domain", fontsize=8)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, ha="right", fontsize=6)
axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=6)

# Create the DB count bar plot in the second subplot
domains = list(db_count.keys())
db_values = list(db_count.values())
axes[1].barh(domains, db_values, color="#1f77b4", alpha=0.8)  # Adjusted to a similar blue tone
axes[1].set_xlabel("#DB Count", fontsize=6)  # Reduce x-axis title size
axes[1].set_yticklabels([])  # Remove y-axis ticks
axes[1].set_xticks(range(0, max(db_values) + 1, max(2, max(db_values) // 4)))  # Keep sparse x-axis ticks
axes[1].tick_params(axis='x', labelsize=6)  # Reduce x-axis tick label size

# Adjust layout for better fitting
plt.tight_layout()

# Show the plot
plt.show()