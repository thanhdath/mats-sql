import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample dataset (Ensure to replace with actual data)
data = {
    "Method": ["MATS (Ours)", "DAILSQL(SC)", "CodeS-15B", "CodeS-7B", "REDSQL-3B\n+NatSQL", "REDSQL-3B", "Graphix\n+PICARD"],
    "w/o JOIN": [92.49, 89.1, 90.6, 89.6, 89.0, 90.1, 88.3],
    "w/ JOIN": [79.9, 75.0, 76.2, 78.9, 76.7, 69.1, 69.6],
    "w/o Subquery": [88.85, 84.2, 86.0, 86.7, 85.8, 83.2, 82.1],
    "w/ Subquery": [72.29, 63.6, 51.5, 45.5, 33.3, 39.4, 45.5],
    "w/o Logical\nConnector": [88.98, 85.3, 86.4, 87.0, 85.8, 83.9, 83.1],
    "w/ Logical\nConnector": [72.22, 65.6, 68.9, 68.9, 66.7, 60.0, 58.9],
    "w/o ORDER-BY": [88.08, 84.3, 85.1, 86.3, 83.6, 81.7, 80.9],
    "w/ ORDER-BY": [85.65, 81.0, 84.4, 82.3, 86.1, 82.3, 81.0],
    "Overall": [87.1, 83.6, 84.9, 85.4, 84.1, 81.8, 80.9]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df.set_index("Method", inplace=True)

# Transpose DataFrame to swap axes
df = df.T

# Remove duplicates by stripping subset names
df.index = df.index.str.strip()
df = df.loc[~df.index.duplicated(keep='first')]

# Set up the figure size
plt.figure(figsize=(4.5, 3.5))

# Create the heatmap
sns.heatmap(df, annot=True, cmap="YlGnBu", linewidths=0.5, fmt=".1f", cbar=False)

# Labels
plt.xlabel("", fontsize=8)
plt.ylabel("Subset", fontsize=8)

# Rotate x-axis labels for better readability
plt.xticks(rotation=90, ha="right", fontsize=6)
plt.yticks(fontsize=6)

# Show the plot
plt.tight_layout()
plt.show()
