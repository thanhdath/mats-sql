# import matplotlib.pyplot as plt

# # Data
# gamma_values = [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
# bird_dev_values = [54.95, 55.54, 56, 55.54, 56.39, 56.19, 57.11, 57.24, 57.24, 56.52, 56.32]
# count_values = [14, 14, 16, 15, 14, 12, 10, 5, 8, 6, 8]

# # Adjust figure size for single-column fit in a 2-column research paper
# fig_width = 3.5  # Typical width for a single column in inches
# fig_height = 2.5  # Adjusted height for readability

# # Create figure and axis objects with adjusted size
# fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))

# # First Y-axis (EX%)
# ax1.set_xlabel(r"$\lambda$", fontsize=10)  # Using LaTeX formatting for lambda
# ax1.set_ylabel("EX%", color="tab:blue", fontsize=10)
# ax1.plot(gamma_values, bird_dev_values, marker="o", linestyle="-", color="tab:blue")
# ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=8)
# ax1.tick_params(axis="x", labelsize=8)

# # Second Y-axis (No. Syntax Error)
# ax2 = ax1.twinx()
# ax2.set_ylabel("No. Syntax Error", color="tab:red", fontsize=10)
# ax2.plot(gamma_values, count_values, marker="s", linestyle="--", color="tab:red")
# ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=8)

# # Grid
# ax1.grid(True, linestyle="--", alpha=0.6)

# # Adjust layout for better spacing
# plt.tight_layout()

# # Show plot
# plt.show()


import matplotlib.pyplot as plt

# Data
gamma_values = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
bird_dev_values = [54.95, 56.19, 57.11, 57.24, 57.24, 56.52, 56.32]
count_values = [14, 12, 10, 5, 8, 6, 8]

# Adjust figure size for single-column fit in a 2-column research paper
fig_width = 3.5  # Typical width for a single column in inches
fig_height = 2.5  # Adjusted height for readability

# Create figure and axis objects with adjusted size
fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))

# First Y-axis (EX%)
ax1.set_xlabel(r"$\lambda$", fontsize=10)  # Using LaTeX formatting for lambda
ax1.set_ylabel("EX%", color="tab:blue", fontsize=10)
ax1.plot(gamma_values, bird_dev_values, marker="o", linestyle="-", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=8)
ax1.tick_params(axis="x", labelsize=8)

# Second Y-axis (No. Syntax Error)
ax2 = ax1.twinx()
ax2.set_ylabel("No. Syntax Error", color="tab:red", fontsize=10)
ax2.plot(gamma_values, count_values, marker="s", linestyle="--", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=8)

# Grid
ax1.grid(True, linestyle="--", alpha=0.6)

# Adjust layout for better spacing
plt.tight_layout()

# Show plot
plt.show()
