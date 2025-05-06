import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.signal import find_peaks
from matplotlib.lines import Line2D

np.random.seed(42)
random.seed(42)

N = 10
n = 2
alpha = 216
kappa = 20
ks0 = 1
ks1 = 0.01
beta = 2
Q_values = [0.4, 0.63, 0.8]
t_span = (0, 600)
t_eval = np.linspace(*t_span, 6000)

alpha_x_base = 216
fold_changes = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]
alpha_x_values = [alpha_x_base * fc for fc in fold_changes]
cmap = plt.get_cmap("tab10")
colors = cmap(np.linspace(0, 1, len(alpha_x_values)))

def idx(i, offset):
    return i * 9 + offset

def repressilator_system(t, y, Q, alpha_x_val):
    dydt = np.zeros_like(y)
    Ai, Bi, Ci = [], [], []
    ai, bi, ci = [], [], []
    xi, Xi, Si = [], [], []

    for i in range(N):
        Ai.append(y[idx(i, 0)])
        Bi.append(y[idx(i, 1)])
        Ci.append(y[idx(i, 2)])
        ai.append(y[idx(i, 3)])
        bi.append(y[idx(i, 4)])
        ci.append(y[idx(i, 5)])
        xi.append(y[idx(i, 6)])
        Xi.append(y[idx(i, 7)])
        Si.append(y[N * 9 + i])

    Se = Q * np.mean(Si)

    for i in range(N):
        dai = alpha / (1 + Ci[i] ** n) - ai[i]
        dbi = alpha / (1 + Ai[i] ** n) - bi[i] + (alpha*Xi[i]**n)/(1+Xi[i]**n)
        dci = alpha / (1 + Bi[i] ** n) + kappa * Si[i] / (1 + Si[i]) - ci[i]

        beta_i = np.random.normal(loc=1.0, scale=0.1)
        dAi = beta * (ai[i] - Ai[i])
        dBi = beta * (bi[i] - Bi[i])
        dCi = beta * (ci[i] - Ci[i])

        eta = {0.4: 0.4433, 0.63: 1.1367}.get(Q, 2.67)
        dSi = -ks0 * Si[i] + ks1 * Ai[i] + eta * (Se - Si[i])
        
        dxi = alpha_x_val / (1 + Ai[i] ** n) - xi[i]
        dXi = beta_i * (xi[i] - Xi[i])

        dydt[idx(i, 0)] = dAi
        dydt[idx(i, 1)] = dBi
        dydt[idx(i, 2)] = dCi
        dydt[idx(i, 3)] = dai
        dydt[idx(i, 4)] = dbi
        dydt[idx(i, 5)] = dci
        dydt[idx(i, 6)] = dxi
        dydt[idx(i, 7)] = dXi
        dydt[N * 9 + i] = dSi

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
            amp = (signal[peaks[:min_len]] - signal[troughs[:min_len]]) / 2
            amplitudes.append(np.mean(amp))
    return np.mean(amplitudes), np.mean(periods)

combined_traces = {}
average_traces = {}
results = []

for alpha_x_val, color in zip(alpha_x_values, colors):
    for Q in Q_values:
        np.random.seed(0)
        y0 = np.random.rand(N * 9 + N) * 0.1

        sol = solve_ivp(lambda t, y: repressilator_system(t, y, Q, alpha_x_val),
                        t_span, y0, t_eval=t_eval)

        time = sol.t
        bi_matrix = np.zeros((len(time), N))
        for i in range(N):
            bi_matrix[:, i] = sol.y[idx(i, 4), :]

        combined_traces[(alpha_x_val, Q)] = (time, bi_matrix)
        average_traces[(alpha_x_val, Q)] = np.mean(bi_matrix, axis=1)

        sync_full = compute_synchrony_metric(bi_matrix)
        amp_full, period_full = compute_amplitude_period_metrics(time, bi_matrix)

        zoom_mask = time <= 50
        bi_matrix_zoom = bi_matrix[zoom_mask]
        time_zoom = time[zoom_mask]
        sync_zoom = compute_synchrony_metric(bi_matrix_zoom)
        amp_zoom, period_zoom = compute_amplitude_period_metrics(time_zoom, bi_matrix_zoom)

        results.append({
            'alpha_x': round(alpha_x_val, 2),
            'Q': Q,
            'Sync_Full': round(sync_full, 2),
            'Amp_Full': round(amp_full * 2, 2),
            'Period_Full': round(period_full, 2),
            'Sync_0_50': round(sync_zoom, 2),
            'Amp_0_50': round(amp_zoom * 2, 2),
            'Period_0_50': round(period_zoom, 2),
        })

        fig_full, ax_full = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        for ax, q_val in zip(ax_full, Q_values):
            for i in range(N):
                ax.plot(time, bi_matrix[:, i], color=color, alpha=0.6)
            ax.set_title(f"Q={q_val} | alpha_x={alpha_x_val:.1f}")
            ax.set_ylabel("mRNA bi")
            ax.grid(True)
        ax_full[-1].set_xlabel("Time (s)")
        fig_full.tight_layout()
        fig_full.savefig(f"full_alpha_x_{alpha_x_val:.1f}.png")
        plt.close(fig_full)

        fig_zoom, ax_zoom = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        for ax, q_val in zip(ax_zoom, Q_values):
            for i in range(N):
                ax.plot(time_zoom, bi_matrix_zoom[:, i], color=color, alpha=0.6)
            ax.set_title(f"Zoom 0-50s | Q = {q_val} | alpha_x = {alpha_x_val:.1f}")
            ax.set_ylabel("mRNA bi")
            ax.grid(True)
        ax_zoom[-1].set_xlabel("Time (s)")
        fig_zoom.tight_layout()
        fig_zoom.savefig(f"zoom_alpha_x_{alpha_x_val:.1f}.png")
        plt.close(fig_zoom)

df_results = pd.DataFrame(results)
df_results.to_csv("alpha_x_scan_summary_metrics.csv", index=False)
print("Saved alpha_x_scan_summary_metrics.csv")

for Q in Q_values:
    fig_all, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    for ax, q_val in zip(axes, Q_values):
        for (a_z, q_key), (t, mat) in combined_traces.items():
            if q_key == q_val:
                color = colors[alpha_x_values.index(a_z)]
                for i in range(N):
                    ax.plot(t, mat[:, i], color=color, alpha=0.6)
        ax.set_title(f"Q = {q_val}")
        ax.set_ylabel("mRNA bi")
        ax.grid(True)
    
    axes[-1].set_xlabel("Time (s)")
    fig_all.suptitle("All Cells: mRNA bi per alpha_x", fontsize=16)
    fig_all.tight_layout(rect=[0, 0, 1, 0.96])

    legend_lines = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(alpha_x_values))]
    legend_labels = [f"αx={a:.1f}" for a in alpha_x_values]
    fig_all.legend(legend_lines, legend_labels, loc='upper right')

    fig_all.savefig(f"combined_all_cells.png")
    plt.close(fig_all)

fig_avg_all = plt.figure(figsize=(12, 18))

for i, Q in enumerate(Q_values):
    ax_avg = fig_avg_all.add_subplot(3, 1, i + 1)
    
    for alpha_x_val, color in zip(alpha_x_values, colors):
        avg_trace = np.mean([average_traces[(alpha_x_val, Q)] for Q in Q_values], axis=0)
        ax_avg.plot(time, avg_trace, label=f"αx={alpha_x_val:.1f}", color=color)

    ax_avg.set_title(f"Average mRNA bi per alpha_x | Q = {Q}")
    ax_avg.set_xlabel("Time (s)")
    ax_avg.set_ylabel("Average mRNA bi")
    ax_avg.grid(True)

fig_avg_all.legend(legend_lines, legend_labels, loc='upper right')

fig_avg_all.tight_layout()
fig_avg_all.savefig("combined_avg_curve_all_Qs.png")
plt.close(fig_avg_all)

for Q in Q_values:
    fig_all, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    for ax, q_val in zip(axes, Q_values):
        for (a_z, q_key), (t, mat) in combined_traces.items():
            if q_key == q_val:
                mask = t <= 50
                t_zoomed = t[mask]
                mat_zoomed = mat[mask, :]
                
                color = colors[alpha_x_values.index(a_z)]
                for i in range(N):
                    ax.plot(t_zoomed, mat_zoomed[:, i], color=color, alpha=0.6)
        ax.set_title(f"Q = {q_val}")
        ax.set_ylabel("mRNA bi")
        ax.grid(True)
    
    axes[-1].set_xlabel("Time (s)")
    fig_all.suptitle("All Cells (Zoomed 0-50s): mRNA bi per alpha_x", fontsize=16)
    fig_all.tight_layout(rect=[0, 0, 1, 0.96])

    legend_lines = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(alpha_x_values))]
    legend_labels = [f"αx={a:.1f}" for a in alpha_x_values]
    fig_all.legend(legend_lines, legend_labels, loc='upper right')

    fig_all.savefig(f"combined_all_cells_zoomed_0_50s.png")
    plt.close(fig_all)
