import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.signal import find_peaks

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

        dzi = alpha_z / (1 + Bi[i] ** n) - zi[i]
        dwi = alpha_w / (1 + Ai[i] ** n) - wi[i]
        dxi = alpha_x / (1 + Ai[i] ** n) - xi[i]
        dyi = alpha_y / (1 + Bi[i] ** n) - yi[i]

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
            amp = (signal[peaks[:min_len]] - signal[troughs[:min_len]]) / 2
            amplitudes.append(np.mean(amp))

    return np.mean(amplitudes), np.mean(periods)

def simulate_and_plot(Q, ax_full, ax_zoom):
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
        ax_full.plot(time, bi_matrix[:, i])
    ax_full.set_title(
        f"Q = {Q} | Avg Sync = {sync_full:.4f} | Avg Amp = {amp_full*2:.2f} | Avg Period = {period_full:.2f}")
    ax_full.set_ylabel("mRNA bi")
    ax_full.grid(True)

    zoom_mask = time <= 50
    bi_matrix_zoom = bi_matrix[zoom_mask]
    time_zoom = time[zoom_mask]
    amp_zoom, period_zoom = compute_amplitude_period_metrics(time_zoom, bi_matrix_zoom)
    sync_zoom = compute_synchrony_metric(bi_matrix_zoom)

    for i in range(N):
        ax_zoom.plot(time_zoom, bi_matrix_zoom[:, i])
    ax_zoom.set_title(
        f"Q = {Q} | Avg Sync = {sync_zoom:.4f} | Avg Amp = {amp_zoom*2:.2f} | Avg Period = {period_zoom*2:.2f}")
    ax_zoom.set_ylabel("mRNA bi")
    ax_zoom.set_xlim(0, 50)
    ax_zoom.grid(True)

    df = pd.DataFrame(bi_matrix, columns=[f'Cell_{i+1}' for i in range(N)])
    df.insert(0, 'Time', time)
    # df.to_csv(f"phase2b_bi_levels_Q{Q}.csv", index=False)

    print(f"Q={Q}")
    print(f"  Avg Sync (full): {sync_full:.4f}")
    print(f"  Avg Amp (full): {amp_full*2:.2f}")
    print(f"  Avg Period (full): {period_full*2:.2f}")
    print(f"  Avg Sync (0-50s): {sync_zoom:.4f}")
    print(f"  Avg Amp (0-50s): {amp_zoom*2:.2f}")
    print(f"  Avg Period (0-50s): {period_zoom*2:.2f}")
    print('-'*40)

fig_full, axes_full = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig_zoom, axes_zoom = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for i, Q in enumerate(Q_values):
    simulate_and_plot(Q, axes_full[i], axes_zoom[i])

axes_full[-1].set_xlabel("Time (s)")
axes_zoom[-1].set_xlabel("Time (s)")
fig_full.suptitle("Phase 2b (Group Model): Full 600s", fontsize=16)
fig_zoom.suptitle("Phase 2b (Group Model): First 50s", fontsize=16)

fig_full.tight_layout(rect=[0, 0, 1, 0.97])
fig_zoom.tight_layout(rect=[0, 0, 1, 0.97])
fig_full.savefig("phase2b_full_600s.png")
fig_zoom.savefig("phase2b_zoom_0_50s.png")
plt.show()
