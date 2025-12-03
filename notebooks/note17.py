"""
note17.ipynb を一括で実行するための関数を定義し、
いろいろな条件下でシミュレーション、学習、評価を行う。
"""

import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from src.utils.save_log import (
    create_result_dir,
    init_log,
    save_figure_to_log,
    save_text_output,
)

# 乱数固定用の処理
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# 真値パラメータ（すべてのデータセットで共通）
L_true: float = 200e-6
C_true: float = 48e-6
R_true: float = 10

Vin: float = 20
Vref: float = 12
f_sw: float = 1e5  # スイッチング周波数
points_per_cycle: int = 200  # 1周期あたりのプロット数
cycles: int = 1000  # 周期数


# 学習に使うデータの長さ
tail_len = points_per_cycle * 7
train_ratio = 0.3
valid_ratio = 0.3
# test_ratio = 1 - train_ratio - valid_ratio

# 学習パラメータ
L_init = 100e-6
C_init = 100e-6
R_init = 8.0

# 異なるパラメータに異なる学習率を設定
lr_L = 2e-3
lr_C = 2e-2
lr_R = 2e-3

epochs = 1000


class BuckConverterCell(nn.Module):
    def __init__(self, L_init: float, C_init: float, R_init: float) -> None:
        super().__init__()
        # パラメータを対数空間で学習（正の値を保証）
        self.log_L = nn.Parameter(torch.log(torch.tensor(L_init)))
        self.log_C = nn.Parameter(torch.log(torch.tensor(C_init)))
        self.log_R = nn.Parameter(torch.log(torch.tensor(R_init)))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        h: [iL, vC]
        x: [vs, u, dt]
        """

        iL = h[:, 0]
        vC = h[:, 1]
        vs = x[:, 0]
        u = x[:, 1]
        dt = x[:, 2]
        vp = vs * u

        # パラメータを指数関数で変換（正の値を保証）
        L = torch.exp(self.log_L)
        C = torch.exp(self.log_C)
        R = torch.exp(self.log_R)

        # オイラー法
        iL_next = iL + (dt / L) * (vp - vC)
        vC_next = vC + (dt / C) * (iL - vC / R)

        return torch.stack([iL_next, vC_next], dim=1)


def buck_converter_training_from_simulation(
    L_true: float,
    C_true: float,
    R_true: float,
    Vin: float,
    Vref: float,
    f_sw: float,
    points_per_cycle: int,
    cycles: int,
    tail_len: int,
    train_ratio: float,
    valid_ratio: float,
    L_init: float,
    C_init: float,
    R_init: float,
    lr_L: float,
    lr_C: float,
    lr_R: float,
    epochs: int,
) -> None:
    """
    note17.ipynb を一括で実行するための関数
    """
    # 実験設定を辞書としてまとめる（重複検出用）
    config_dict = {
        "L_true": L_true,
        "C_true": C_true,
        "R_true": R_true,
        "Vin": Vin,
        "Vref": Vref,
        "f_sw": f_sw,
        "points_per_cycle": points_per_cycle,
        "cycles": cycles,
        "tail_len": tail_len,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "L_init": L_init,
        "C_init": C_init,
        "R_init": R_init,
        "lr_L": lr_L,
        "lr_C": lr_C,
        "lr_R": lr_R,
        "epochs": epochs,
    }

    # ログ保存の初期化
    result_dir = create_result_dir("note17", config_dict=config_dict)
    _ = init_log(result_dir, "notebooks/note17.py")
    print(f"ログ保存先: {result_dir}/log.md")
    print("実行中です...")

    duty: float = Vref / Vin
    T: float = 1 / f_sw  # 1周期の実時間

    t: np.ndarray = np.linspace(0, cycles * T, cycles * points_per_cycle + 1)
    dt: np.ndarray = np.diff(t)

    # スイッチング信号
    duty_phase = (t[:-1] % T) / T
    u = (duty_phase < duty).astype(int)

    # 入力電圧
    vs = np.ones(len(t) - 1) * Vin

    # モデルを作成
    model_true = BuckConverterCell(L_init=L_true, C_init=C_true, R_init=R_true)

    # numpy配列をテンソルに変換（dtをxに含める）
    x_tensor = torch.tensor(np.c_[vs, u, dt], dtype=torch.float32)

    il_list = []
    vc_list = []

    # シミュレーション実行
    with torch.no_grad():
        h_current: torch.Tensor = torch.zeros(1, 2)  # [i_L=0, v_C=0]
        il_list.append(h_current[0, 0].item())  # i_L
        vc_list.append(h_current[0, 1].item())  # v_C

        for j in range(len(t) - 1):
            h_current = model_true(h_current, x_tensor[j : j + 1])

            il_list.append(h_current[0, 0].item())  # i_L
            vc_list.append(h_current[0, 1].item())  # v_C

    iL = torch.tensor(np.array(il_list), dtype=torch.float32)
    vC = torch.tensor(np.array(vc_list), dtype=torch.float32)

    # 定常の10周期分
    t_10 = t[-tail_len - 1 :]
    dt_10 = dt[-tail_len:]
    u_10 = u[-tail_len:]
    vs_10 = vs[-tail_len:]
    iL_10 = iL[-tail_len - 1 :]
    vC_10 = vC[-tail_len - 1 :]

    # numpy配列をテンソルに変換
    dt_10_tensor = torch.tensor(dt_10, dtype=torch.float32)
    u_10_tensor = torch.tensor(u_10, dtype=torch.float32)
    vs_10_tensor = torch.tensor(vs_10, dtype=torch.float32)

    # train
    train_len = int(len(dt_10) * train_ratio)

    t_train = t_10[: train_len + 1]
    dt_train = dt_10_tensor[:train_len]
    u_train = u_10_tensor[:train_len]
    vs_train = vs_10_tensor[:train_len]
    iL_train = iL_10[: train_len + 1]
    vC_train = vC_10[: train_len + 1]

    h_train = torch.stack([iL_train[:-1], vC_train[:-1]], dim=1)
    x_train = torch.stack([vs_train, u_train, dt_train], dim=1)
    target_train = torch.stack([iL_train[1:], vC_train[1:]], dim=1)

    # valid
    valid_len = int(len(t_10) * valid_ratio)

    # t_valid = t_10[train_len : train_len + valid_len]
    dt_valid = dt_10_tensor[train_len : train_len + valid_len]
    u_valid = u_10_tensor[train_len : train_len + valid_len]
    vs_valid = vs_10_tensor[train_len : train_len + valid_len]
    iL_valid = iL_10[train_len : train_len + valid_len + 1]
    vC_valid = vC_10[train_len : train_len + valid_len + 1]

    h_valid = torch.stack([iL_valid[:-1], vC_valid[:-1]], dim=1)
    x_valid = torch.stack([vs_valid, u_valid, dt_valid], dim=1)
    target_valid = torch.stack([iL_valid[1:], vC_valid[1:]], dim=1)

    # test
    t_test = t_10[train_len + valid_len :]
    dt_test = dt_10_tensor[train_len + valid_len :]
    u_test = u_10_tensor[train_len + valid_len :]
    vs_test = vs_10_tensor[train_len + valid_len :]
    iL_test = iL_10[train_len + valid_len :]
    vC_test = vC_10[train_len + valid_len :]

    h_test = torch.stack([iL_test[:-1], vC_test[:-1]], dim=1)
    x_test = torch.stack([vs_test, u_test, dt_test], dim=1)
    target_test = torch.stack([iL_test[1:], vC_test[1:]], dim=1)

    model = BuckConverterCell(L_init=L_init, C_init=C_init, R_init=R_init)

    optimizer = optim.Adam(
        [
            {"params": [model.log_L], "lr": lr_L},
            {"params": [model.log_C], "lr": lr_C},
            {"params": [model.log_R], "lr": lr_R},
        ]
    )

    loss_fn = nn.MSELoss()

    # 損失履歴を保存
    loss_history = {"train": [], "valid": []}
    param_history = {"L": [], "C": [], "R": []}

    # 学習ループ
    for _ in range(epochs):
        # 学習モード
        model.train()
        optimizer.zero_grad()
        h_pred_train = model(h_train, x_train)
        train_loss = loss_fn(h_pred_train, target_train)
        train_loss.backward()
        optimizer.step()

        loss_history["train"].append(train_loss.item())

        # 検証モード（勾配計算なし）
        model.eval()
        with torch.no_grad():
            h_pred_valid = model(h_valid, x_valid)
            valid_loss = loss_fn(h_pred_valid, target_valid)
            loss_history["valid"].append(valid_loss.item())

        # パラメータの履歴を保存
        param_history["L"].append(model.log_L.exp().item())
        param_history["C"].append(model.log_C.exp().item())
        param_history["R"].append(model.log_R.exp().item())

    # テストデータでの評価
    model.eval()
    with torch.no_grad():
        h_pred_test = model(h_test, x_test)
        test_loss = loss_fn(h_pred_test, target_test)

    final_train_loss = loss_history["train"][-1]
    final_valid_loss = loss_history["valid"][-1]
    final_test_loss = test_loss.item()

    final_L = param_history["L"][-1]
    final_C = param_history["C"][-1]
    final_R = param_history["R"][-1]

    # 実験設定の表示
    result_text = []
    result_text.append("=" * 60)
    result_text.append("")
    result_text.append("【回路パラメータ】")
    result_text.append(
        f"  真の値: L = {L_true:.6e} [H], C = {C_true:.6e} [F], R = {R_true:.3f} [Ω]"
    )
    result_text.append(
        f"  初期値: L = {L_init:.6e} [H], C = {C_init:.6e} [F], R = {R_init:.3f} [Ω]"
    )
    result_text.append(
        f"  推論値: L = {final_L:.6e} [H], C = {final_C:.6e} [F], R = {final_R:.3f} [Ω]"
    )
    result_text.append("")
    result_text.append("【入力条件】")
    result_text.append(
        f"  Vin (入力電圧) = {Vin:.2f} [V], Vref (目標電圧) = {Vref:.2f} [V]"
    )
    result_text.append(f"  スイッチング周波数 f_sw = {f_sw:.0f} [Hz]")
    result_text.append(f"  1周期あたりのプロット数 = {points_per_cycle}")
    result_text.append("")
    result_text.append("【データ分割】")
    result_text.append(
        f"  シミュレーション時間: {cycles}周期 = {cycles * T * 1e6:.1f}μs"
    )
    result_text.append(
        f"  後ろから約{int(tail_len / points_per_cycle)}周期"
        f"({tail_len}ステップ, ={tail_len * dt_10.mean() * 1e6:.1f}μs)を使用"
    )
    result_text.append(f"  学習データ: {train_len} ステップ")
    result_text.append(f"  検証データ: {valid_len} ステップ")
    result_text.append(f"  テストデータ: {len(t_10) - train_len - valid_len} ステップ")
    result_text.append("")
    result_text.append("【最終Loss】")
    result_text.append(f"  学習データ: {final_train_loss:.6e}")
    result_text.append(f"  検証データ: {final_valid_loss:.6e}")
    result_text.append(f"  テストデータ: {final_test_loss:.6e}")
    result_text.append("")
    result_text.append("【学習設定】")
    result_text.append(f"  エポック数: {epochs}")
    result_text.append(f"  学習率: L = {lr_L:.2e}, C = {lr_C:.2e}, R = {lr_R:.2e}")

    result_output = "\n".join(result_text)

    # 結果をログに保存
    save_text_output(result_dir, result_output, "最終結果")

    # 学習に使ったデータ
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # 1: u (switching signal)
    axs[0].step(t_train[1:] * 1e6, u_train, where="post", color="blue", linewidth=1.5)
    axs[0].set_ylabel("u", fontsize=12)
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title(
        "Training Data: Switching signal u", fontsize=14, fontweight="bold"
    )

    # 2: vs (input voltage)
    axs[1].plot(t_train[1:] * 1e6, vs_train, color="green", linewidth=1.5)
    axs[1].set_ylabel("vs [V]", fontsize=12)
    axs[1].grid(True, alpha=0.3)
    axs[1].set_title("Training Data: Input voltage vs", fontsize=14, fontweight="bold")

    # 3: iL (inductor current)
    axs[2].plot(t_train * 1e6, iL_train, color="tab:blue", linewidth=1.5)
    axs[2].set_ylabel("iL [A]", fontsize=12)
    axs[2].grid(True, alpha=0.3)
    axs[2].set_title(
        "Training Data: Inductor current iL", fontsize=14, fontweight="bold"
    )

    # 4: vC (capacitor voltage)
    axs[3].plot(t_train * 1e6, vC_train, color="tab:orange", linewidth=1.5)
    axs[3].set_ylabel("vC [V]", fontsize=12)
    axs[3].set_xlabel("Time [μs]", fontsize=12)
    axs[3].grid(True, alpha=0.3)
    axs[3].set_title(
        "Training Data: Capacitor voltage vC", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()

    # 画像をログに保存
    save_figure_to_log(fig, "training_data", result_dir, "学習データ")

    # Lossの遷移をグラフ表示
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    epochs_list = range(1, epochs + 1)
    ax.plot(
        epochs_list, loss_history["train"], label="Train Loss", linewidth=2, alpha=0.8
    )
    ax.plot(
        epochs_list, loss_history["valid"], label="Valid Loss", linewidth=2, alpha=0.8
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title("Loss Transition", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    # 画像をログに保存
    save_figure_to_log(fig, "loss_transition", result_dir, "Lossの遷移")

    # 回路パラメータの学習による変化をグラフ表示
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    epochs_list = range(1, epochs + 1)

    # Lの変化
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
    axs[0].set_ylabel("L [H]", fontsize=12)
    axs[0].set_title("Inductance L: Learning Progress", fontsize=14, fontweight="bold")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=11)

    # Cの変化
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
    axs[1].set_ylabel("C [F]", fontsize=12)
    axs[1].set_title("Capacitance C: Learning Progress", fontsize=14, fontweight="bold")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=11)

    # Rの変化
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
    axs[2].set_ylabel("R [Ω]", fontsize=12)
    axs[2].set_xlabel("Epoch", fontsize=12)
    axs[2].set_title("Resistance R: Learning Progress", fontsize=14, fontweight="bold")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(fontsize=11)

    plt.tight_layout()

    # 画像をログに保存
    save_figure_to_log(
        fig, "parameter_learning", result_dir, "回路パラメータの学習による変化"
    )

    # テストデータでの予測結果を取得
    model.eval()
    with torch.no_grad():
        preds_test = model(h_test, x_test)

    # 予測結果をnumpy配列に変換
    preds_test_np = preds_test[:, :].cpu().detach().numpy()  # [n_test, 2]

    # 予測値（iL, vC）
    iL_test_pred = preds_test_np[:, 0]
    vC_test_pred = preds_test_np[:, 1]

    # グラフ表示
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # iLの比較
    axs[0].plot(
        t_test * 1e6,
        iL_test,
        label="True value",
        linewidth=2,
        alpha=0.7,
        color="blue",
    )
    axs[0].plot(
        t_test[1:] * 1e6,
        iL_test_pred,
        label="Predicted",
        linewidth=2,
        alpha=0.7,
        color="red",
        linestyle="--",
    )
    axs[0].set_ylabel("iL [A]", fontsize=12)
    axs[0].set_title(
        "Test Data: Inductor current iL Comparison", fontsize=14, fontweight="bold"
    )
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=11)

    # vCの比較
    axs[1].plot(
        t_test * 1e6,
        vC_test,
        label="True value",
        linewidth=2,
        alpha=0.7,
        color="blue",
    )
    axs[1].plot(
        t_test[1:] * 1e6,
        vC_test_pred,
        label="Predicted",
        linewidth=2,
        alpha=0.7,
        color="red",
        linestyle="--",
    )
    axs[1].set_ylabel("vC [V]", fontsize=12)
    axs[1].set_xlabel("Time [μs]", fontsize=12)
    axs[1].set_title(
        "Test Data: Capacitor voltage vC Comparison", fontsize=14, fontweight="bold"
    )
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=11)

    plt.tight_layout()

    # 画像をログに保存
    save_figure_to_log(fig, "test_prediction", result_dir, "テストデータでの予測結果")

    print("finish")
    print(f"ログ保存先: {result_dir}/log.md")
    print("実行が完了しました")


if __name__ == "__main__":
    buck_converter_training_from_simulation(
        L_true,
        C_true,
        R_true,
        Vin,
        Vref,
        f_sw,
        points_per_cycle,
        cycles,
        tail_len,
        train_ratio,
        valid_ratio,
        L_init,
        C_init,
        R_init,
        lr_L,
        lr_C,
        lr_R,
        epochs,
    )
