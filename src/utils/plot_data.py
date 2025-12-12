import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_iLvC(
    t: np.ndarray,
    iL: np.ndarray,
    vC: np.ndarray,
    T: float,
    title: str,
    show_tail_10: bool = True,
    show_tail_1: bool = True,
) -> None:
    if T <= 0:
        raise ValueError("T must be positive.")
    if t.size == 0:
        return

    def _plot_section(
        t_: np.ndarray,
        iL_: np.ndarray,
        vC_: np.ndarray,
        section_title: str,
    ) -> None:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True)
        ax[0].plot(t_, iL_, color="tab:orange")
        ax[0].set_title(section_title)
        ax[0].set_ylabel("iL [A]")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(t_, vC_, color="tab:blue")
        ax[1].set_ylabel("vC [V]")
        ax[1].set_xlabel("t [s]")
        ax[1].grid(True, alpha=0.3)
        ax[1].xaxis.set_major_locator(MaxNLocator(nbins=12))

        fig.tight_layout()

    # 全体
    _plot_section(t, iL, vC, section_title=title)

    # 末尾の拡大（10周期 / 1周期）
    t_end = float(t[-1])

    if show_tail_10:
        window = 10.0 * T
        mask = t >= (t_end - window)
        _plot_section(t[mask], iL[mask], vC[mask], section_title=f"{title} (tail 10T)")

    if show_tail_1:
        window = 1.0 * T
        mask = t >= (t_end - window)
        _plot_section(t[mask], iL[mask], vC[mask], section_title=f"{title} (tail 1T)")


def plot_u_vs_iL_vC(
    t: np.ndarray,
    u: np.ndarray,
    vs: np.ndarray,
    iL: np.ndarray,
    vC: np.ndarray,
    title: str = "Training Data",
) -> None:
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # --- u (スイッチング信号) ---
    axs[0].step(t[1:] * 1e6, u, where="post", color="blue", linewidth=1.5)
    axs[0].set_ylabel("u", fontsize=12)
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title(f"{title}: Switching signal u", fontsize=14, fontweight="bold")

    # --- vs (入力電圧) ---
    axs[1].plot(t[1:] * 1e6, vs, color="green", linewidth=1.5)
    axs[1].set_ylabel("vs [V]", fontsize=12)
    axs[1].grid(True, alpha=0.3)
    axs[1].set_title(f"{title}: Input voltage vs", fontsize=14, fontweight="bold")

    # --- iL (インダクタ電流) ---
    axs[2].plot(t * 1e6, iL, color="tab:blue", linewidth=1.5)
    axs[2].set_ylabel("iL [A]", fontsize=12)
    axs[2].grid(True, alpha=0.3)
    axs[2].set_title(f"{title}: Inductor current iL", fontsize=14, fontweight="bold")

    # --- vC (キャパシタ電圧) ---
    axs[3].plot(t * 1e6, vC, color="tab:orange", linewidth=1.5)
    axs[3].set_ylabel("vC [V]", fontsize=12)
    axs[3].set_xlabel("Time [μs]", fontsize=12)
    axs[3].grid(True, alpha=0.3)
    axs[3].set_title(f"{title}: Capacitor voltage vC", fontsize=14, fontweight="bold")

    fig.tight_layout()


def plot_compare_tail(
    t1: np.ndarray,
    iL1: np.ndarray,
    vC1: np.ndarray,
    label1: str,
    t2: np.ndarray,
    iL2: np.ndarray,
    vC2: np.ndarray,
    label2: str,
    T: float,
    N_cycles: float = 10.0,
    title: str = "Waveform Comparison",
) -> None:
    """
    Overlay the last N cycles of two types of data
    """
    # Get mask for N cycles at the end
    t_end1 = float(t1[-1])
    t_end2 = float(t2[-1])
    mask1 = t1 >= (t_end1 - N_cycles * T)
    mask2 = t2 >= (t_end2 - N_cycles * T)
    # Adjust t1, t2 to start from zero in the window
    t1_view = t1[mask1] - t1[mask1][0]
    t2_view = t2[mask2] - t2[mask2][0]
    iL1_view = np.array(iL1)[mask1]
    vC1_view = np.array(vC1)[mask1]
    iL2_view = np.array(iL2)[mask2]
    vC2_view = np.array(vC2)[mask2]

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    # iL
    ax[0].plot(t1_view * 1e6, iL1_view, label=label1, color="tab:blue", alpha=0.7)
    ax[0].plot(
        t2_view * 1e6,
        iL2_view,
        label=label2,
        color="tab:orange",
        alpha=0.7,
        linestyle="--",
    )
    ax[0].set_ylabel("iL [A]")
    ax[0].set_title(f"{title}: iL (Last {N_cycles:.0f} Cycles)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    # vC
    ax[1].plot(t1_view * 1e6, vC1_view, label=label1, color="tab:blue", alpha=0.7)
    ax[1].plot(
        t2_view * 1e6,
        vC2_view,
        label=label2,
        color="tab:orange",
        alpha=0.7,
        linestyle="--",
    )
    ax[1].set_ylabel("vC [V]")
    ax[1].set_xlabel("Time [μs]")
    ax[1].set_title(f"{title}: vC (Last {N_cycles:.0f} Cycles)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
