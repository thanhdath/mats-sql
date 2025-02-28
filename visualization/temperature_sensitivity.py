import matplotlib.pyplot as plt

# Given data
x_values = [0, 0.3, 0.5, 1.0]  # Temperature values
y_values = [59.12, 63.58, 63.98, 64.13]  # Mean values (A)
y_errors = [0.09, 0.13, 0.23, 0.25]  # Standard deviation (B)

# Adjust figure size for 1-column fit in a 2-column research paper (~3.5 inches wide)
plt.figure(figsize=(3.5, 2.5))

# Create error bar plot
plt.errorbar(x_values, y_values, yerr=y_errors, fmt='o-', capsize=3, capthick=1)

# Labels with optimized font size for readability
plt.xlabel("Temperature", fontsize=9)
plt.ylabel("EX%", fontsize=9)
plt.xticks(x_values, fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True, linestyle="--", alpha=0.7)

# Tight layout for better fit in a research paper column
plt.tight_layout()

# Show the plot
plt.show()
