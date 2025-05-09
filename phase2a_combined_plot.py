import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_olivia = pd.read_csv("/Users/olivia/Documents/Spring 2025/MCB_546/MCB_546_Final_Project/phase2a_olivia/phase2a_olivia_parameter_scan/alpha_z_scan_summary_metrics.csv")
df_sydney = pd.read_csv("/Users/olivia/Documents/Spring 2025/MCB_546/MCB_546_Final_Project/phase2a_sydney/phase2a_sydney_parameter_scan/alpha_x_scan_summary_metrics.csv")
df_adriana = pd.read_csv("/Users/olivia/Documents/Spring 2025/MCB_546/MCB_546_Final_Project/phase2a_adriana/phase2a_adriana_parameter_scan/alpha_w_scan_summary_metrics.csv")
df_sean = pd.read_csv("/Users/olivia/Documents/Spring 2025/MCB_546/MCB_546_Final_Project/phase2a_sean/phase2a_sean_parameter_scan/alpha_y_scan_summary_metrics.csv")



df_olivia['FoldChange'] = df_olivia['alpha_z'] / 216.0
df_sean['FoldChange'] = df_sean['alpha_y'] / 216.0
df_adriana['FoldChange'] = df_adriana['alpha_w'] / 216.0
df_sydney['FoldChange'] = df_sydney['alpha_x'] / 216.0

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

q_values = [0.4, 0.63, 0.8]
amp_baselines = [132.35, 130.26, 135.53]

for i, q in enumerate(q_values):
    subset_olivia = df_olivia[df_olivia['Q'] == q]
    axes[i].plot(np.log2(subset_olivia['FoldChange']), subset_olivia['Amp_Full'], marker='o', color = 'gold', label = 'Alpha Z')
    subset_sean = df_sean[df_sean['Q'] == q]
    axes[i].plot(np.log2(subset_sean['FoldChange']), subset_sean['Amp_Full'], marker='o', color = 'dodgerblue', label = 'Alpha Y')
    subset_adriana = df_adriana[df_adriana['Q'] == q]
    axes[i].plot(np.log2(subset_adriana['FoldChange']), subset_adriana['Amp_Full'], marker='o', color = 'lime', label = 'Alpha W')
    subset_sydney = df_sydney[df_sean['Q'] == q]
    axes[i].plot(np.log2(subset_sydney['FoldChange']), subset_sydney['Amp_Full'], marker='o', color = 'red', label = 'Alpha X')
    axes[i].axhline(y=amp_baselines[i], color = 'black', linestyle = '--')

    axes[i].set_title(f"Q = {q}")
    axes[i].set_xlabel("log2(Fold Change from Alpha_ = 216)")
    plt.legend()
    plt.suptitle('Phase 2a Combined - New Alpha Parameter Scan Effects')
    if i == 0:
        axes[i].set_ylabel("Average Amplitude")

plt.tight_layout()
plt.savefig('phase2a_combined_summary_plot.png')
plt.show()
