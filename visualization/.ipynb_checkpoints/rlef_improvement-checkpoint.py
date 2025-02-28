import matplotlib.pyplot as plt

# Data for Spider Dev and BIRD Dev
steps = [0, 1, 2, 3]  # RLEF iterations
labels = ['SFT', 'Iter 1', 'Iter 2', 'Iter 3']

# MATS 3B Planner Data
spider_3b = [83.3, 84, 85.4, 85.2]
bird_3b = [53.65, 56.32, 59.32, 58.6]

# MATS Data
spider_mats = [85.5, 86.3, 87.1, 87]
bird_mats = [59.06, 60.82, 64.73, 62.58]

# Reference model performance for horizontal lines
ref_lines = {
    "GPT-4": {"color": "cyan", "style": "-.", "spider": 76.8, "bird": 49.15},
    "CodeS-15B": {"color": "red", "style": ":", "spider": 84.9, "bird": 58.47},
    "MAC-SQL + GPT-4": {"color": "purple", "style": "--", "spider": 86.75, "bird": 59.59},
    "DIN-SQL + GPT-4": {"color": "green", "style": "--", "spider": 82.8, "bird": 50.72}
}

# Adjust figure size for a single-column layout
fig, axes = plt.subplots(2, 1, figsize=(3.5, 5), sharex=True)

# Spider Dev subplot
axes[0].plot(steps, spider_3b, marker='o', color='blue', label='MATS 3B Planner', linewidth=1.5)
axes[0].plot(steps, spider_mats, marker='s', color='red', label='MATS', linewidth=1.5)
for label, data in ref_lines.items():
    axes[0].axhline(y=data["spider"], color=data["color"], linestyle=data["style"], label=label, linewidth=1)
axes[0].set_title('Spider Dev', fontsize=10, pad=8)
axes[0].grid(alpha=0.3)

# BIRD Dev subplot
axes[1].plot(steps, bird_3b, marker='o', color='blue', linewidth=1.5)
axes[1].plot(steps, bird_mats, marker='s', color='red', linewidth=1.5)
for label, data in ref_lines.items():
    axes[1].axhline(y=data["bird"], color=data["color"], linestyle=data["style"], linewidth=1)
axes[1].set_title('BIRD Dev', fontsize=10, pad=8)
axes[1].set_xticks(steps)
axes[1].set_xticklabels(labels, rotation=45, fontsize=8)
axes[1].grid(alpha=0.3)

# Remove axis titles (labels)
axes[0].set_ylabel('')
axes[1].set_ylabel('')
axes[1].set_xlabel('')

# Add a single legend outside the subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=7, ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1))

# Adjust layout for compactness
plt.tight_layout(pad=1.0)
plt.show()
