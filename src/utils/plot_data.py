import matplotlib.pyplot as plt
import numpy as np


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
        ax[0].plot(t_, iL_, color="tab:blue")
        ax[0].set_title(section_title)
        ax[0].set_ylabel("iL [A]")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(t_, vC_, color="tab:orange")
        ax[1].set_ylabel("vC [V]")
        ax[1].set_xlabel("t [s]")
        ax[1].grid(True, alpha=0.3)

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
