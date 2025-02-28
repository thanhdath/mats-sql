import numpy as np
import matplotlib.pyplot as plt

# Data for plotting
candidates = np.array([1, 5, 10, 20, 30])
achieved_accuracy = np.array([59.17, 63.75, 64.73, 62.91, 62.78])
upper_bound = np.array([59.17, 69.6, 72.7, 76, 77.8])
lower_bound = np.array([59.17, 59.17, 59.17, 59.17, 59.17])

# Define figure size to fit a research paper column
fig_width = 4  # Adjusted for single-column fit
fig_height = fig_width * 0.75  # Maintain aspect ratio

# Create the figure
plt.figure(figsize=(fig_width, fig_height), dpi=300)  # High DPI for publication quality

# Plot the curves
plt.plot(candidates, upper_bound, label="Upper Bound", linestyle="--", color="red")
plt.plot(candidates, achieved_accuracy, label="MATS", marker="o", color="blue")
plt.plot(candidates, lower_bound, label="Lower Bound", linestyle="--", color="green")

# Fill between (shading) without adding legend
plt.fill_between(candidates, achieved_accuracy, upper_bound, color="gray", alpha=0.3)
plt.fill_between(candidates, lower_bound, achieved_accuracy, color="gray", alpha=0.3)

# Labels and formatting
plt.xlabel("Number of Candidates", fontsize=10)
plt.ylabel(r"EX\%", fontsize=10)  # LaTeX-style notation for EX%
plt.xticks(candidates, fontsize=9)
plt.yticks(fontsize=9)
plt.legend(fontsize=9, loc="upper left")  # Move legend to top left
plt.grid(True, linewidth=0.5)

# Remove unnecessary borders for a cleaner publication look
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# Save the figure in a high-quality format for LaTeX insertion
plt.savefig("accuracy_vs_bounds.pdf", bbox_inches="tight", format="pdf")

# Show the figure
plt.show()
