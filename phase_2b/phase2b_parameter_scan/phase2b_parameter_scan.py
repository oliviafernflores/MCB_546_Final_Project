import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.signal import find_peaks
import os
from matplotlib.lines import Line2D
import matplotlib.cm as cm

np.random.seed(42)
random.seed(42)

N = 10
n = 2
alpha = 216
alpha_z = 0.01
alpha_w = 0.01
alpha_x = 0.01
alpha_y = 0.01
kappa = 20
ks0 = 1
ks1 = 0.01
beta = 2
Q_values = [0.4, 0.63, 0.8]
t_span = (0, 600)
t_eval = np.linspace(*t_span, 6000)

cmap = plt.get_cmap("tab20")
colors = cmap(np.linspace(0, 1, 13))

combined_data = {}

def idx(i, offset):
    return i * 15 + offset

def repressilator_system(t, y, Q):
    dydt = np.zeros_like(y)
    Ai, Bi, Ci = [], [], []
    ai, bi, ci = [], [], []
    zi, Zi = [], []
    wi, Wi = [], []
    xi, Xi = [], []
    yi, Yi = [], []
    Si = []

    for i in range(N):
        Ai.append(y[idx(i, 0)])
        Bi.append(y[idx(i, 1)])
        Ci.append(y[idx(i, 2)])
        ai.append(y[idx(i, 3)])
        bi.append(y[idx(i, 4)])
        ci.append(y[idx(i, 5)])
        zi.append(y[idx(i, 6)])
        Zi.append(y[idx(i, 7)])
        wi.append(y[idx(i, 8)])
        Wi.append(y[idx(i, 9)])
        xi.append(y[idx(i, 10)])
        Xi.append(y[idx(i, 11)])
        yi.append(y[idx(i, 12)])
        Yi.append(y[idx(i, 13)])
        Si.append(y[N * 15 + i])

    Se = Q * np.mean(Si)

    for i in range(N):
        dai = alpha / (1 + Ci[i] ** n) - ai[i] + ((alpha * Wi[i]**n)/(1+Wi[i]**n))
        dbi = alpha / ((1 + Ai[i] ** n) * (1 + Zi[i] ** n)) - bi[i] + ((alpha * Xi[i]**n)/(1+Xi[i]**n))
        dci = alpha / ((1 + Bi[i] ** n) * (1 + Yi[i] ** n)) + kappa * Si[i] / (1 + Si[i]) - ci[i]

        beta_i = np.random.normal(loc=1.0, scale=0.1)
        dAi = beta_i * (ai[i] - Ai[i])
        dBi = beta_i * (bi[i] - Bi[i])
        dCi = beta_i * (ci[i] - Ci[i])

        dzi = (alpha_z * Bi[i] ** n)/ (1 + Bi[i] ** n) - zi[i]
        dwi = (alpha_w * Ai[i] ** n)/ (1 + Ai[i] ** n) - wi[i]
        dxi = (alpha_x * Ai[i] ** n)/ (1 + Ai[i] ** n) - xi[i]
        dyi = (alpha_y * Bi[i] ** n)/ (1 + Bi[i] ** n) - yi[i]

        dZi = beta_i * (zi[i] - Zi[i])
        dWi = beta_i * (wi[i] - Wi[i])
        dXi = beta_i * (xi[i] - Xi[i])
        dYi = beta_i * (yi[i] - Yi[i])

        eta = {0.4: 0.4433, 0.63: 1.1367}.get(Q, 2.67)
        dSi = -ks0 * Si[i] + ks1 * Ai[i] + eta * (Se - Si[i])

        dydt[idx(i, 0)] = dAi
        dydt[idx(i, 1)] = dBi
        dydt[idx(i, 2)] = dCi
        dydt[idx(i, 3)] = dai
        dydt[idx(i, 4)] = dbi
        dydt[idx(i, 5)] = dci
        dydt[idx(i, 6)] = dzi
        dydt[idx(i, 7)] = dZi
        dydt[idx(i, 8)] = dwi
        dydt[idx(i, 9)] = dWi
        dydt[idx(i, 10)] = dxi
        dydt[idx(i, 11)] = dXi
        dydt[idx(i, 12)] = dyi
        dydt[idx(i, 13)] = dYi
        dydt[N * 15 + i] = dSi

    return dydt

def compute_synchrony_metric(bi_matrix):
    mean_centered = bi_matrix - np.mean(bi_matrix, axis=1, keepdims=True)
    var_over_cells = np.var(mean_centered, axis=1)
    return np.mean(var_over_cells)

def compute_amplitude_period_metrics(time, bi_matrix):
    amplitudes = []
    periods = []

    for i in range(bi_matrix.shape[1]):
        signal = bi_matrix[:, i]
        peaks, _ = find_peaks(signal)
        troughs, _ = find_peaks(-signal)

        if len(peaks) > 1:
            periods.append(np.mean(np.diff(time[peaks])))

        if len(peaks) > 0 and len(troughs) > 0:
            min_len = min(len(peaks), len(troughs))
            if min_len > 0:
                amp = (signal[peaks[:min_len]] - signal[troughs[:min_len]]) / 2
                amplitudes.append(np.mean(amp))

    mean_amp = np.mean(amplitudes) if amplitudes else 0.0
    mean_period = np.mean(periods) if periods else 0.0

    return mean_amp, mean_period


def simulate_and_plot(Q, ax_full, ax_zoom, color):
    np.random.seed(0)
    y0 = np.random.rand(N * 15 + N) * 0.1

    sol = solve_ivp(lambda t, y: repressilator_system(t, y, Q),
                    t_span, y0, t_eval=t_eval)

    time = sol.t
    bi_matrix = np.zeros((len(time), N))
    for i in range(N):
        bi_matrix[:, i] = sol.y[idx(i, 4), :]

    sync_full = compute_synchrony_metric(bi_matrix)
    amp_full, period_full = compute_amplitude_period_metrics(time, bi_matrix)

    for i in range(N):
        ax_full.plot(time, bi_matrix[:, i], color = color)
    ax_full.set_title(
        f"Q = {Q}")
    ax_full.set_ylabel("mRNA bi")
    ax_full.grid(True)

    zoom_mask = time <= 50
    bi_matrix_zoom = bi_matrix[zoom_mask]
    time_zoom = time[zoom_mask]
    amp_zoom, period_zoom = compute_amplitude_period_metrics(time_zoom, bi_matrix_zoom)
    sync_zoom = compute_synchrony_metric(bi_matrix_zoom)

    for i in range(N):
        ax_zoom.plot(time_zoom, bi_matrix_zoom[:, i], color = color)
    ax_zoom.set_title(
        f"Q = {Q}")
    ax_zoom.set_ylabel("mRNA bi")
    ax_zoom.set_xlim(0, 50)
    ax_zoom.grid(True)
    
    return {
        'Q': Q,
        'Sync_Full': round(sync_full, 2),
        'Amp_Full': round(amp_full * 2, 2),
        'Period_Full': round(period_full, 2),
        'Sync_0_50': round(sync_zoom, 2),
        'Amp_0_50': round(amp_zoom * 2, 2),
        'Period_0_50': round(period_zoom, 2),
        'time': time,
        'bi_matrix': bi_matrix
    }

output_dir = "scan_outputs"
os.makedirs(output_dir, exist_ok=True)
results = []

fold_changes = [1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]
alpha_params = ['alpha_z', 'alpha_w', 'alpha_x', 'alpha_y']
base_alpha = 216

original_values = {
    'alpha_z': alpha_z,
    'alpha_w': alpha_w,
    'alpha_x': alpha_x,
    'alpha_y': alpha_y,
}

for param in alpha_params:
    for fold in fold_changes:
        alpha_value = base_alpha * fold
        globals()[param] = alpha_value

        fig_full, axes_full = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        fig_zoom, axes_zoom = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

        color = colors[fold_changes.index(fold)]

        for i, Q in enumerate(Q_values):
            result = simulate_and_plot(Q, axes_full[i], axes_zoom[i], color)
            combined_data[(param, alpha_value, Q)] = (result['time'], result['bi_matrix'])

            result.update({
                'Alpha_Param': param,
                'Alpha_Value': alpha_value,
            })
            
            del result['time']
            del result['bi_matrix']
            results.append(result)

        axes_full[-1].set_xlabel("Time (s)")
        axes_zoom[-1].set_xlabel("Time (s)")
        title_suffix = f"{param}_{alpha_value:.2f}"
        # fig_full.suptitle(f"{title_suffix} - Group Model", fontsize=16)
        # fig_zoom.suptitle(f"{title_suffix} - Group Model", fontsize=16)
        fig_full.tight_layout(rect=[0, 0, 1, 0.97])
        fig_zoom.tight_layout(rect=[0, 0, 1, 0.97])

        fname_suffix = f"{param}_{alpha_value:.2f}"
        fig_full.savefig(os.path.join(output_dir, f"full_600s_{fname_suffix}.png"))
        fig_zoom.savefig(os.path.join(output_dir, f"zoom_0_50s_{fname_suffix}.png"))
        plt.close(fig_full)
        plt.close(fig_zoom)

        globals()[param] = original_values[param]

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by=['Alpha_Param', 'Q', 'Alpha_Value'])

df_results.to_csv("simulation_results.csv", index=False)
print("Parameter scan complete. Results saved to 'simulation_results.csv'.")




for param in alpha_params:
    alpha_values = [base_alpha * fc for fc in fold_changes]
    
    for Q in Q_values:
        fig_all, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

        for ax, q_val in zip(axes, Q_values):
            for i, alpha_val in enumerate(alpha_values):
                key = (param, alpha_val, q_val)
                if key in combined_data:
                    t, mat = combined_data[key]
                    color = colors[i]
                    for j in range(N):
                        ax.plot(t, mat[:, j], color=color, alpha=0.6)

            ax.set_title(f"Q = {q_val}")
            ax.set_ylabel("mRNA bi")
            ax.grid(True)

        axes[-1].set_xlabel("Time (s)")
        fig_all.suptitle(f"All Cells: mRNA bi per {param} | Q = {Q}", fontsize=16)
        fig_all.tight_layout(rect=[0, 0, 1, 0.96])

        legend_lines = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(alpha_values))]
        legend_labels = [f"{param}={a:.2f}" for a in alpha_values]
        fig_all.legend(legend_lines, legend_labels, loc='upper right')

        fig_all.savefig(f"combined_all_cells_{param}.png")
        plt.close(fig_all)

    fig_avg_all = plt.figure(figsize=(12, 18))
    for i, Q in enumerate(Q_values):
        ax_avg = fig_avg_all.add_subplot(3, 1, i + 1)

        for j, alpha_val in enumerate(alpha_values):
            key = (param, alpha_val, Q)
            if key in combined_data:
                t, mat = combined_data[key]
                avg_trace = np.mean(mat, axis=1)
                ax_avg.plot(t, avg_trace, label=f"{param}={alpha_val:.2f}", color=colors[j])

        ax_avg.set_title(f"Average mRNA bi per {param} | Q = {Q}")
        ax_avg.set_xlabel("Time (s)")
        ax_avg.set_ylabel("Average mRNA bi")
        ax_avg.grid(True)

    fig_avg_all.legend(legend_lines, legend_labels, loc='upper right')
    fig_avg_all.tight_layout()
    fig_avg_all.savefig(f"combined_avg_curve_all_Qs_{param}.png")
    plt.close(fig_avg_all)

    for Q in Q_values:
        fig_zoom, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

        for ax, q_val in zip(axes, Q_values):
            for i, alpha_val in enumerate(alpha_values):
                key = (param, alpha_val, q_val)
                if key in combined_data:
                    t, mat = combined_data[key]
                    mask = t <= 50
                    t_zoom = t[mask]
                    mat_zoom = mat[mask, :]
                    color = colors[i]
                    for j in range(N):
                        ax.plot(t_zoom, mat_zoom[:, j], color=color, alpha=0.6)

            ax.set_title(f"Q = {q_val}")
            ax.set_ylabel("mRNA bi")
            ax.grid(True)

        axes[-1].set_xlabel("Time (s)")
        fig_zoom.suptitle(f"Zoomed 0–50s: mRNA bi per {param} | Q = {Q}", fontsize=16)
        fig_zoom.tight_layout(rect=[0, 0, 1, 0.96])

        fig_zoom.legend(legend_lines, legend_labels, loc='upper right')
        fig_zoom.savefig(f"combined_all_cells_zoomed_0_50s_{param}.png")
        plt.close(fig_zoom)
