import matplotlib.pyplot as plt
import numpy as np

# Data from the table
strategies = ["No Thought", "Chain-of-Thought", "Few-shot Thoughts"]
categories = ["simple", "moderate", "challenging", "overall"]

# Execution accuracy scores
scores = np.array([
    [59.46, 37.5, 30.34, 50.07],
    [60.11, 40.09, 37.93, 51.96],
    [63.03, 43.75, 38.62, 54.89]
])

# Bar width and positions
bar_width = 0.25
x = np.arange(len(categories))

# Creating figure with aspect ratio suitable for a research paper (single-column)
fig, ax = plt.subplots(figsize=(4.5, 2.5))

# Plot bars for each strategy
for i, strategy in enumerate(strategies):
    ax.bar(x + i * bar_width, scores[i], width=bar_width, label=strategy)

# Labels and formatting
ax.set_xticks(x + bar_width)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel("EX%", fontsize=9)

# Move legend to the top
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True)

# Tight layout for better fit
plt.tight_layout()

# Show the plot
plt.show()
