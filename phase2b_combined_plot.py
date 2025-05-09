import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_combined = pd.read_csv('/Users/olivia/Documents/Spring 2025/MCB_546/MCB_546_Final_Project/phase_2b/phase2b_parameter_scan/simulation_results.csv')


df_combined['FoldChange'] = df_combined['Alpha_Value'] / 216.0

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

q_values = [0.4, 0.63, 0.8]
amp_baselines = [132.35, 130.26, 135.53]
alpha_vals = ['alpha_w', 'alpha_x', 'alpha_y', 'alpha_z']
alpha_colors = {'alpha_w':'lime', 'alpha_x':'red', 'alpha_y':'dodgerblue', 'alpha_z':'gold'}
alpha_labels = {'alpha_w':'Alpha W', 'alpha_x':'Alpha X', 'alpha_y':'Alpha Y', 'alpha_z':'Alpha Z'}

for i, q in enumerate(q_values):
    subset_q = df_combined[df_combined['Q'] == q]
    for alpha in alpha_vals:
        subset_alpha = subset_q[subset_q['Alpha_Param'] == alpha]
        axes[i].plot(np.log2(subset_alpha['FoldChange']), subset_alpha['Amp_Full'], marker = 'o', color = alpha_colors[alpha], label = alpha_labels[alpha])
    axes[i].set_title(f"Q = {q}")
    axes[i].set_xlabel("log2(Fold Change from Alpha_ = 216)")
    axes[i].axhline(y=amp_baselines[i], color = 'black', linestyle = '--')
    plt.legend()
    plt.suptitle('Phase 2b Combined - New Alpha Parameter Scan Effects')
    if i == 0:
        axes[i].set_ylabel("Average Amplitude")

plt.tight_layout()
plt.savefig('phase2b_combined_summary_plot.png')
plt.show()
