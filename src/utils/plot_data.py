from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

color_iL = "tab:blue"
color_vC = "tab:orange"


PlotStyle = Literal["line", "scatter", "line+marker"]


def _plot_series(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    style: PlotStyle,
    color: str,
    label: str | None = None,
    alpha: float = 1.0,
    linestyle: str = "-",
    linewidth: float = 1.5,
    marker: str = "o",
    markersize: float = 3.0,
    zorder: int | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    matplotlibの描画を line / scatter / line+marker で切り替える薄いラッパ。
    - style="line": ax.plot
    - style="scatter": ax.scatter
    - style="line+marker": ax.plot(marker付き)
    """
    kwargs: dict[str, Any] = {}
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    if style == "scatter":
        # sは面積なので、見た目がmarkersizeっぽくなるように二乗を使う
        ax.scatter(
            x,
            y,
            color=color,
            label=label,
            alpha=alpha,
            marker=marker,
            s=float(markersize) ** 2,
            zorder=zorder,
            **kwargs,
        )
        return

    # line / line+marker
    plot_kwargs: dict[str, Any] = {
        "color": color,
        "label": label,
        "alpha": alpha,
        "linestyle": linestyle,
        "linewidth": linewidth,
    }
    if style == "line+marker":
        plot_kwargs.update(
            {
                "marker": marker,
                "markersize": markersize,
            }
        )
    if zorder is not None:
        plot_kwargs["zorder"] = zorder
    plot_kwargs.update(kwargs)

    ax.plot(x, y, **plot_kwargs)


def plot_iLvC(
    t: np.ndarray,
    iL: np.ndarray,
    vC: np.ndarray,
    T: float,
    title: str,
    show_tail: tuple = (10.0, 1.0),
    plot_style: PlotStyle = "line",
    marker: str = "o",
    markersize: float = 3.0,
    linewidth: float = 1.5,
) -> list[tuple[plt.Figure, plt.Axes]]:
    if T <= 0:
        raise ValueError("T must be positive.")
    if t.size == 0:
        return []

    def _plot_section(
        t_: np.ndarray,
        iL_: np.ndarray,
        vC_: np.ndarray,
        section_title: str,
    ) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True)
        _plot_series(
            ax[0],
            t_,
            iL_,
            style=plot_style,
            color=color_iL,
            alpha=1.0,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
        )
        ax[0].set_title(section_title)
        ax[0].set_ylabel("iL [A]")
        ax[0].grid(True, alpha=0.3)

        _plot_series(
            ax[1],
            t_,
            vC_,
            style=plot_style,
            color=color_vC,
            alpha=1.0,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
        )
        ax[1].set_ylabel("vC [V]")
        ax[1].set_xlabel("t [s]")
        ax[1].grid(True, alpha=0.3)
        ax[1].xaxis.set_major_locator(MaxNLocator(nbins=12))

        fig.tight_layout()
        return fig, ax

    figs = []
    # 全体
    fig, ax = _plot_section(t, iL, vC, section_title=title)
    figs.append((fig, ax))

    # 末尾の拡大（10周期 / 1周期）
    t_end = float(t[-1])

    for tail_window in show_tail:
        window = tail_window * T
        mask = t >= (t_end - window)
        fig, ax = _plot_section(
            t[mask], iL[mask], vC[mask], section_title=f"{title} (tail {tail_window}T)"
        )
        figs.append((fig, ax))
    return figs


def plot_u_vs_iL_vC(
    t: np.ndarray,
    u: np.ndarray,
    vs: np.ndarray,
    iL: np.ndarray,
    vC: np.ndarray,
    title: str = "Training Data",
) -> tuple[plt.Figure, plt.Axes]:
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
    axs[2].plot(t * 1e6, iL, color=color_iL, linewidth=1.5)
    axs[2].set_ylabel("iL [A]", fontsize=12)
    axs[2].grid(True, alpha=0.3)
    axs[2].set_title(f"{title}: Inductor current iL", fontsize=14, fontweight="bold")

    # --- vC (キャパシタ電圧) ---
    axs[3].plot(t * 1e6, vC, color=color_vC, linewidth=1.5)
    axs[3].set_ylabel("vC [V]", fontsize=12)
    axs[3].set_xlabel("Time [μs]", fontsize=12)
    axs[3].grid(True, alpha=0.3)
    axs[3].set_title(f"{title}: Capacitor voltage vC", fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig, axs


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
    iL_range: tuple[float, float] | None = None,
    vC_range: tuple[float, float] | None = None,
    style1: PlotStyle = "line",
    style2: PlotStyle = "line",
    marker1: str = "o",
    marker2: str = "s",
    markersize: float = 3.0,
    linewidth: float = 1.5,
) -> tuple[plt.Figure, plt.Axes]:
    """
    2種類のデータの末尾N周期分を重ねて表示
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if t1.size == 0 or t2.size == 0:
        raise ValueError("t1/t2 must be non-empty.")

    default_clip_line_kwargs: dict = {
        "color": "black",
        "linestyle": ":",
        "linewidth": 1.2,
        "alpha": 0.6,
    }

    def _draw_clip_lines(
        ax_: plt.Axes, y_lo: float, y_hi: float, base_label: str
    ) -> None:
        ax_.axhline(y=y_lo, label=f"{base_label} lo", **default_clip_line_kwargs)
        ax_.axhline(y=y_hi, label=f"{base_label} hi", **default_clip_line_kwargs)

    # N周期分のマスクを取得
    # - t1/t2 で別々の終端基準にすると横軸がズレるので、共通窓で揃える
    t_end1 = float(t1[-1])
    t_end2 = float(t2[-1])
    t_end = min(t_end1, t_end2)
    t_start = t_end - float(N_cycles) * float(T)

    mask1 = (t1 >= t_start) & (t1 <= t_end)
    mask2 = (t2 >= t_start) & (t2 <= t_end)

    if not np.any(mask1) or not np.any(mask2):
        raise ValueError(
            "Not enough samples in the last N_cycles window for t1/t2. "
            "Try decreasing N_cycles or check time ranges."
        )

    # 共通の開始時刻(t_start)を0に揃える
    t1_view = t1[mask1] - t_start
    t2_view = t2[mask2] - t_start
    iL1_view = np.array(iL1)[mask1]
    vC1_view = np.array(vC1)[mask1]
    iL2_view = np.array(iL2)[mask2]
    vC2_view = np.array(vC2)[mask2]

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    # iL
    _plot_series(
        ax[0],
        t1_view * 1e6,
        iL1_view,
        style=style1,
        color=color_iL,
        label=label1,
        alpha=0.7,
        linestyle="-",
        linewidth=linewidth,
        marker=marker1,
        markersize=markersize,
    )
    _plot_series(
        ax[0],
        t2_view * 1e6,
        iL2_view,
        style=style2,
        color=color_vC,
        label=label2,
        alpha=0.7,
        linestyle="--",
        linewidth=linewidth,
        marker=marker2,
        markersize=markersize,
    )
    ax[0].set_ylabel("iL [A]")
    ax[0].set_title(f"{title}: iL (Last {N_cycles:.0f} Cycles)")
    if iL_range is not None:
        _draw_clip_lines(ax[0], float(iL_range[0]), float(iL_range[1]), "iL clip")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    # vC
    _plot_series(
        ax[1],
        t1_view * 1e6,
        vC1_view,
        style=style1,
        color=color_iL,
        label=label1,
        alpha=0.7,
        linestyle="-",
        linewidth=linewidth,
        marker=marker1,
        markersize=markersize,
    )
    _plot_series(
        ax[1],
        t2_view * 1e6,
        vC2_view,
        style=style2,
        color=color_vC,
        label=label2,
        alpha=0.7,
        linestyle="--",
        linewidth=linewidth,
        marker=marker2,
        markersize=markersize,
    )
    ax[1].set_ylabel("vC [V]")
    ax[1].set_xlabel("Time [μs]")
    ax[1].set_title(f"{title}: vC (Last {N_cycles:.0f} Cycles)")
    if vC_range is not None:
        _draw_clip_lines(ax[1], float(vC_range[0]), float(vC_range[1]), "vC clip")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_param_learning_progress(
    param_history: dict[str, list[float]],
    L_true: float,
    C_true: float,
    R_true: float,
    epochs: int,
    figsize: tuple[int, int] = (12, 10),
) -> tuple[plt.Figure, plt.Axes]:
    """
    回路パラメータ推定の学習過程をプロットする
    """
    # エポック数に応じてx軸を作成
    epochs_list = list(range(1, epochs + 1))

    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # L
    axs[0].plot(
        epochs_list,
        param_history["L"],
        label="Estimated",
        linewidth=2,
        color="tab:blue",
    )
    axs[0].axhline(
        y=L_true,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True value ({L_true:.6e})",
    )
    axs[0].set_ylabel("L [H]")
    axs[0].set_title("Inductance L: Learning Progress", fontweight="bold")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=11)

    # C
    axs[1].plot(
        epochs_list,
        param_history["C"],
        label="Estimated",
        linewidth=2,
        color="tab:green",
    )
    axs[1].axhline(
        y=C_true,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True value ({C_true:.6e})",
    )
    axs[1].set_ylabel("C [F]")
    axs[1].set_title("Capacitance C: Learning Progress", fontweight="bold")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=11)

    # R
    axs[2].plot(
        epochs_list,
        param_history["R"],
        label="Estimated",
        linewidth=2,
        color="tab:orange",
    )
    axs[2].axhline(
        y=R_true,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True value ({R_true:.3f})",
    )
    axs[2].set_ylabel("R [Ω]")
    axs[2].set_xlabel("Epoch")
    axs[2].set_title("Resistance R: Learning Progress", fontweight="bold")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(fontsize=11)

    fig.tight_layout()
    return fig, axs


def plot_param_learning_progress_all(
    param_history: Mapping[str, Sequence[float]],
    true_params: Mapping[str, float] | None = None,
    *,
    figsize: tuple[int, int] = (12, 10),
    ncols: int = 1,
    units: Mapping[str, str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    param_history に含まれる全パラメータの学習過程を自動でプロットする。

    - param_history: {"L": [...], "C": [...], ...} のような辞書
    - true_params: {"L": 1e-3, "C": 1e-6, ...} のような真値（無いキーは破線を描かない）
    """
    if len(param_history) == 0:
        raise ValueError("param_history must be non-empty.")
    if ncols <= 0:
        raise ValueError("ncols must be positive.")

    default_units: dict[str, str] = {"L": "H", "C": "F", "R": "Ω"}
    unit_map: dict[str, str] = {**default_units, **(dict(units) if units else {})}

    keys: list[str] = list(param_history.keys())
    n_params = len(keys)
    nrows = int(np.ceil(n_params / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axs_flat: list[plt.Axes] = list(axs.ravel())

    for i, k in enumerate(keys):
        ax = axs_flat[i]
        y = np.asarray(param_history[k], dtype=float)
        if y.size == 0:
            raise ValueError(f"param_history['{k}'] is empty.")

        x = np.arange(1, y.size + 1, dtype=int)
        ax.plot(x, y, label="Estimated", linewidth=2, color="tab:blue")

        if true_params is not None and k in true_params:
            true_v = float(true_params[k])
            ax.axhline(
                y=true_v,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"True value ({true_v:.6e})",
            )

        unit = unit_map.get(k)
        ax.set_ylabel(f"{k} [{unit}]" if unit else k)
        ax.set_title(f"{k}: Learning Progress", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 余ったAxesは非表示
    for j in range(n_params, len(axs_flat)):
        axs_flat[j].set_visible(False)

    # 1段目以外でも見やすいように最下段だけxlabel（複数列なら全下段に付ける）
    for j in range(max(0, nrows - 1) * ncols, min(nrows * ncols, len(axs_flat))):
        if axs_flat[j].get_visible():
            axs_flat[j].set_xlabel("Epoch")

    fig.tight_layout()
    return fig, axs


def plot_buck_gru_components_tail(
    t: np.ndarray,
    iL_meas: np.ndarray,
    vC_meas: np.ndarray,
    iL_buck: np.ndarray,
    vC_buck: np.ndarray,
    iL_gru: np.ndarray,
    vC_gru: np.ndarray,
    T: float,
    N_cycles: float = 4.0,
    title: str = "Buck + GRU components",
    include_overlay: bool = True,
) -> tuple[plt.Figure, np.ndarray, plt.Figure, np.ndarray]:
    """
    Overlay(Measured vs Buck+GRU) / Measured と Buck の比較 / GRU / (Buck+GRU)
    を末尾N周期で比較描画する。

    - iL, vC でそれぞれ（overlay込みなら）4段プロットを返す
    - 入力はすべて同じ時間軸・同じ長さを想定（tは[s]）
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if t.size == 0:
        raise ValueError("t must be non-empty.")

    # 末尾N周期だけ表示（時間で切る）
    t0: float = float(t[-1] - N_cycles * T)
    mask: np.ndarray = np.asarray(t >= t0)
    if not np.any(mask):
        raise ValueError("mask is empty. Check N_cycles/T and t range.")

    x_us: np.ndarray = np.asarray(t[mask], dtype=float) * 1e6

    iL_sum = np.asarray(iL_buck, dtype=float) + np.asarray(iL_gru, dtype=float)
    vC_sum = np.asarray(vC_buck, dtype=float) + np.asarray(vC_gru, dtype=float)

    def _ylim(*arrs: np.ndarray) -> tuple[float, float]:
        y_min = float(min(np.min(a) for a in arrs))
        y_max = float(max(np.max(a) for a in arrs))
        y_rng = y_max - y_min
        if y_rng <= 0:
            y_rng = max(abs(y_min), 1.0)
        return (y_min - 0.05 * y_rng, y_max + 0.05 * y_rng)

    def _ylim_same_span_as(
        reference_ylim: tuple[float, float], arr: np.ndarray
    ) -> tuple[float, float]:
        """
        参照ylimの「幅」だけをコピーし、中心は0付近（収まらない場合はデータ中心）にする。

        - 軸は独立のまま（shareyしない）
        - 表示レンジ幅を揃えて、ノイズが過大に見えるのを防ぐ
        """
        ref_span: float = float(reference_ylim[1] - reference_ylim[0])
        if ref_span <= 0:
            ref_span = 1.0

        y = np.asarray(arr, dtype=float)
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        y_span = y_max - y_min

        # データが参照幅に収まらない場合だけ、幅を少し広げてクリップを避ける
        span: float = max(ref_span, 1.05 * y_span)

        # 原則は0中心（残差なので）
        center: float = 0.0
        low = center - span / 2.0
        high = center + span / 2.0

        # 0中心で収まらないならデータ中心へ移動（幅は維持）
        if (y_min < low) or (y_max > high):
            center = 0.5 * (y_min + y_max)
            low = center - span / 2.0
            high = center + span / 2.0

        return (float(low), float(high))

    # y軸範囲（Measured/Buck/Sumで揃える）
    iL_ylim = _ylim(
        np.asarray(iL_meas, dtype=float)[mask],
        np.asarray(iL_buck, dtype=float)[mask],
        np.asarray(iL_sum, dtype=float)[mask],
    )
    vC_ylim = _ylim(
        np.asarray(vC_meas, dtype=float)[mask],
        np.asarray(vC_buck, dtype=float)[mask],
        np.asarray(vC_sum, dtype=float)[mask],
    )
    iL_gru_ylim = _ylim_same_span_as(iL_ylim, np.asarray(iL_gru, dtype=float)[mask])
    vC_gru_ylim = _ylim_same_span_as(vC_ylim, np.asarray(vC_gru, dtype=float)[mask])

    # --- iL: overlay + Meas/Buck重ね + GRU + 合成 ---
    nrows: int = 4 if include_overlay else 3
    fig_iL, axs_iL = plt.subplots(nrows, 1, figsize=(14, 3 * nrows), sharex=True)
    axs_iL = np.asarray(axs_iL)

    row0: int = 0
    if include_overlay:
        axs_iL[0].plot(
            x_us,
            np.asarray(iL_meas, dtype=float)[mask],
            label="Measured",
            linewidth=2,
            alpha=0.85,
            color="tab:blue",
        )
        axs_iL[0].plot(
            x_us,
            np.asarray(iL_sum, dtype=float)[mask],
            label="Buck + GRU",
            linewidth=2,
            alpha=0.9,
            color="black",
        )
        axs_iL[0].set_ylabel("Inductor Current $i_L$ [A]", fontsize=12)
        axs_iL[0].set_title(
            f"{title} / Overlay (Measured vs Buck+GRU) (tail {N_cycles:g} cycles)",
            fontsize=14,
        )
        axs_iL[0].set_ylim(iL_ylim)
        axs_iL[0].grid(True, alpha=0.3)
        axs_iL[0].legend(fontsize=11)
        row0 = 1

    # Measured and BuckConverterCell を重ねて描画するプロット
    axs_iL[row0 + 0].plot(
        x_us,
        np.asarray(iL_meas, dtype=float)[mask],
        label="Measured",
        linewidth=2,
        alpha=0.85,
        color="tab:blue",
    )
    axs_iL[row0 + 0].plot(
        x_us,
        np.asarray(iL_buck, dtype=float)[mask],
        label="BuckConverterCell",
        linewidth=2,
        alpha=0.85,
        color="tab:red",
    )
    axs_iL[row0 + 0].set_ylabel("Inductor Current $i_L$ [A]", fontsize=12)
    axs_iL[row0 + 0].set_title(
        f"{title} / Measured vs BuckConverterCell (tail {N_cycles:g} cycles)",
        fontsize=14,
    )
    axs_iL[row0 + 0].set_ylim(iL_ylim)
    axs_iL[row0 + 0].grid(True, alpha=0.3)
    axs_iL[row0 + 0].legend(fontsize=11)

    axs_iL[row0 + 1].plot(
        x_us,
        np.asarray(iL_gru, dtype=float)[mask],
        label="GRU (pred residual)",
        linewidth=2,
        alpha=0.85,
        color="tab:green",
    )
    axs_iL[row0 + 1].set_ylabel("Inductor Current $i_L$ [A]", fontsize=12)
    axs_iL[row0 + 1].set_title(
        f"{title} / GRU output (tail {N_cycles:g} cycles)", fontsize=14
    )
    axs_iL[row0 + 1].set_ylim(iL_gru_ylim)
    axs_iL[row0 + 1].grid(True, alpha=0.3)
    axs_iL[row0 + 1].legend(fontsize=11)

    axs_iL[row0 + 2].plot(
        x_us,
        np.asarray(iL_sum, dtype=float)[mask],
        label="Buck + GRU",
        linewidth=2,
        alpha=0.9,
        color="black",
    )
    axs_iL[row0 + 2].set_ylabel("Inductor Current $i_L$ [A]", fontsize=12)
    axs_iL[row0 + 2].set_xlabel("Time [μs]", fontsize=12)
    axs_iL[row0 + 2].set_title(
        f"{title} / Buck+GRU (tail {N_cycles:g} cycles)", fontsize=14
    )
    axs_iL[row0 + 2].set_ylim(iL_ylim)
    axs_iL[row0 + 2].grid(True, alpha=0.3)
    axs_iL[row0 + 2].legend(fontsize=11)

    fig_iL.tight_layout()

    # --- vC: overlay + Meas/Buck重ね + GRU + 合成 ---
    fig_vC, axs_vC = plt.subplots(nrows, 1, figsize=(14, 3 * nrows), sharex=True)
    axs_vC = np.asarray(axs_vC)

    row0 = 0
    if include_overlay:
        axs_vC[0].plot(
            x_us,
            np.asarray(vC_meas, dtype=float)[mask],
            label="Measured",
            linewidth=2,
            alpha=0.85,
            color="tab:blue",
        )
        axs_vC[0].plot(
            x_us,
            np.asarray(vC_sum, dtype=float)[mask],
            label="Buck + GRU",
            linewidth=2,
            alpha=0.9,
            color="black",
        )
        axs_vC[0].set_ylabel("Capacitor Voltage $v_C$ [V]", fontsize=12)
        axs_vC[0].set_title(
            f"{title} / Overlay (Measured vs Buck+GRU) (tail {N_cycles:g} cycles)",
            fontsize=14,
        )
        axs_vC[0].set_ylim(vC_ylim)
        axs_vC[0].grid(True, alpha=0.3)
        axs_vC[0].legend(fontsize=11)
        row0 = 1

    # Measured and BuckConverterCell を重ねて描画するプロット
    axs_vC[row0 + 0].plot(
        x_us,
        np.asarray(vC_meas, dtype=float)[mask],
        label="Measured",
        linewidth=2,
        alpha=0.85,
        color="tab:blue",
    )
    axs_vC[row0 + 0].plot(
        x_us,
        np.asarray(vC_buck, dtype=float)[mask],
        label="BuckConverterCell",
        linewidth=2,
        alpha=0.85,
        color="tab:red",
    )
    axs_vC[row0 + 0].set_ylabel("Capacitor Voltage $v_C$ [V]", fontsize=12)
    axs_vC[row0 + 0].set_title(
        f"{title} / Measured vs BuckConverterCell (tail {N_cycles:g} cycles)",
        fontsize=14,
    )
    axs_vC[row0 + 0].set_ylim(vC_ylim)
    axs_vC[row0 + 0].grid(True, alpha=0.3)
    axs_vC[row0 + 0].legend(fontsize=11)

    axs_vC[row0 + 1].plot(
        x_us,
        np.asarray(vC_gru, dtype=float)[mask],
        label="GRU (pred residual)",
        linewidth=2,
        alpha=0.85,
        color="tab:green",
    )
    axs_vC[row0 + 1].set_ylabel("Capacitor Voltage $v_C$ [V]", fontsize=12)
    axs_vC[row0 + 1].set_title(
        f"{title} / GRU output (tail {N_cycles:g} cycles)", fontsize=14
    )
    axs_vC[row0 + 1].set_ylim(vC_gru_ylim)
    axs_vC[row0 + 1].grid(True, alpha=0.3)
    axs_vC[row0 + 1].legend(fontsize=11)

    axs_vC[row0 + 2].plot(
        x_us,
        np.asarray(vC_sum, dtype=float)[mask],
        label="Buck + GRU",
        linewidth=2,
        alpha=0.9,
        color="black",
    )
    axs_vC[row0 + 2].set_ylabel("Capacitor Voltage $v_C$ [V]", fontsize=12)
    axs_vC[row0 + 2].set_xlabel("Time [μs]", fontsize=12)
    axs_vC[row0 + 2].set_title(
        f"{title} / Buck+GRU (tail {N_cycles:g} cycles)", fontsize=14
    )
    axs_vC[row0 + 2].set_ylim(vC_ylim)
    axs_vC[row0 + 2].grid(True, alpha=0.3)
    axs_vC[row0 + 2].legend(fontsize=11)

    fig_vC.tight_layout()

    return fig_iL, axs_iL, fig_vC, axs_vC
