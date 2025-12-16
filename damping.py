import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

M = 0.00274
def load_damping(path="damping1.txt"):
    """Load time and displacement columns from damping1.txt."""
    data = np.genfromtxt(path, skip_header=2, delimiter="\t")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    t = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    
    x = [n+0.003 for n in x]
    y = [n+0.008 for n in y]
    
    t = t[:4200]
    x = x[:4200]
    y = y[:4200]
    return t, x, y


def exp_decay(t, k):
    return np.exp(-k * t / (2 * M))


def extract_peak_to_peak_amplitudes(t, signal):
    """Compute peak-to-peak amplitudes using alternating max/min extrema."""
    max_idx, _ = find_peaks(signal)
    min_idx, _ = find_peaks(-signal)
    events = []
    for idx in max_idx:
        events.append((t[idx], signal[idx], "max"))
    for idx in min_idx:
        events.append((t[idx], signal[idx], "min"))
    events.sort(key=lambda x: x[0])

    amp_times = []
    amp_values = []
    for i in range(len(events) - 1):
        t1, v1, label1 = events[i]
        t2, v2, label2 = events[i + 1]
        if label1 == label2:
            continue  # need a high-low or low-high pair
        amp = abs(v1 - v2)
        if amp > 0:
            amp_times.append(0.5 * (t1 + t2))
            amp_values.append(amp)
    return np.array(amp_times), np.array(amp_values)


def fit_envelope(t_peaks, r_peaks):
    """Fit exponential envelope r(t)=r0*exp(-k t/(2M)); only k is fitted."""
    # Keep only positive amplitudes
    positive_mask = r_peaks > 0
    t_peaks = t_peaks[positive_mask]
    r_peaks = r_peaks[positive_mask]

    # Guard against empty peak detection
    if len(t_peaks) == 0:
        raise ValueError("No positive peaks found in signal.")

    # Improved initial guess using slope of ln(A/A0) vs t
    norm_peaks = r_peaks / r_peaks[0]
    if len(t_peaks) > 1:
        slope, _ = np.polyfit(t_peaks, np.log(norm_peaks), 1)
        k0 = max(-2 * M * slope, 1e-4)
    else:
        k0 = 1.0

    (k_opt,), _ = curve_fit(
        exp_decay,
        t_peaks,
        norm_peaks,
        p0=[k0],
        bounds=(0, np.inf),
        maxfev=10000,
    )
    return t_peaks, r_peaks, k_opt


def main():
    t, x, y = load_damping()
    r = np.sqrt(np.square(x) + np.square(y))

    def process_axis(label, signal, color):
        smooth = gaussian_filter1d(signal, sigma=2)
        amp_times, amplitudes = extract_peak_to_peak_amplitudes(t, smooth)
        t_peaks, r_peaks, k_opt = fit_envelope(amp_times, amplitudes)
        return {
            "label": label,
            "smooth": smooth,
            "amp_times": amp_times,
            "amps": r_peaks,
            "k": k_opt,
            "r0": r_peaks[0],
            "color": color,
        }

    result_r = process_axis("r", r, "tab:blue")

    print(f"r: r0={result_r['r0']:.6f}, k={result_r['k']:.6f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    res = result_r
    t_env = np.linspace(res["amp_times"].min(), res["amp_times"].max(), 400)
    # fit_label = rf"$D(t) = {res['r0']:.6f}\,e^{{-{res['k']:.6f} t/(2M)}}$"
    fit_label = rf"$A(t) = A_0 e^{{\frac{{- \gamma t}}{{2m}}}}, A_0={res['r0']:.6f}, \gamma = {res['k']:.6f}$"
    ax.plot(res["amp_times"], res["amps"], "o", label="Peak-to-peak amplitudes", c="tab:orange")
    # ax.plot(
    #     t_env,
    #     res["r0"] * exp_decay(t_env, res["k"]),
    #     "--",
    #     label=fit_label,
    #     c="tab:red",
    # )
    ax.set_ylabel("A(t) (m)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Damping Envelope Fit for A")
    ax.legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
