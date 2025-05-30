import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/olivia/Documents/Spring 2025/MCB_546/MCB_546_Final_Project/phase2a_sydney/phase2a_sydney_parameter_scan/alpha_x_scan_summary_metrics.csv")

df['FoldChange'] = df['alpha_x'] / 216.0

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

q_values = [0.4, 0.63, 0.8]

for i, q in enumerate(q_values):
    subset = df[df['Q'] == q]
    axes[i].plot(np.log2(subset['FoldChange']), subset['Amp_Full'], marker='o', color = 'red')
    axes[i].set_title(f"Q = {q}")
    axes[i].set_xlabel("log2(Fold Change from Alpha_X = 216)")
    plt.suptitle('Phase 2a - Alpha X Parameter Scan Effects')
    if i == 0:
        axes[i].set_ylabel("Average Amplitude")

plt.tight_layout()
plt.savefig('phase2a_sydney_summary_plot.png')
plt.show()
