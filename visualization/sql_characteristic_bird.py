import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the dataset with all extracted EX (%) values including MATS (Ours)
data = {
    "Subset": [
        "w/o JOIN", "w/ JOIN", "w/o Subquery", "w/ Subquery", 
        "w/o Logical\nConnector", "w/ Logical\nConnector", 
        "w/o ORDER-BY", "w/ ORDER-BY", "Overall"
    ],
    "MATS (Ours)": [68.02, 59.21, 63.12, 40.71, 65.33, 56.04, 63.67, 52.75, 64.74],
    "DAILSQL(SC)": [61.4, 53.9, 56.9, 37.9, 59.1, 51.3, 58.3, 46.3, 55.9],
    # "DAILSQL": [60.4, 52.2, 55.4, 35.6, 56.6, 51.0, 57.1, 43.0, 54.3],
    # "C3SQL": [55.3, 48.4, 51.2, 33.3, 54.5, 44.1, 53.5, 37.2, 50.2],
    "CodeS-15B": [63.5, 56.8, 59.8, 36.8, 62.2, 53.2, 61.1, 48.2, 58.5],
    "CodeS-7B": [63.2, 54.8, 58.5, 32.2, 60.6, 51.8, 59.6, 46.6, 57.0],
    # "SFT CodeS-3B": [61.2, 52.7, 56.3, 31.0, 59.8, 48.0, 57.9, 43.0, 54.9],
    # "SFT CodeS-1B": [57.1, 47.9, 51.6, 28.7, 55.3, 43.2, 53.4, 37.9, 50.3],
    "REDSQL-3B": [52.0, 41.1, 44.7, 31.0, 49.4, 36.3, 47.2, 31.1, 43.9],
    "REDSQL-L Large": [45.9, 36.1, 39.6, 21.8, 45.7, 28.6, 41.9, 25.6, 38.6],
    "REDSQL-L Base": [40.9, 30.4, 33.9, 19.5, 38.7, 25.3, 35.5, 23.6, 33.1],
    
}

# Convert to DataFrame and set index
df = pd.DataFrame(data)
df.set_index("Subset", inplace=True)

# Create a figure with a single heatmap
fig = plt.figure(figsize=(4.5, 3.5))

# Create the heatmap
sns.heatmap(df, annot=True, cmap="YlGnBu", linewidths=0.5, fmt=".1f", cbar=False)

plt.set_xlabel("")
plt.set_ylabel("Subset", fontsize=8)
plt.set_xticklabels(plt.get_xticklabels(), rotation=90, ha="right", fontsize=6)
plt.set_yticklabels(plt.get_yticklabels(), fontsize=6)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
