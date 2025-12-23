# %% [markdown]
# # やること
#

# %%
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from src.utils.load_data import load_data, path_list
from src.utils.models import BuckConverterCell, GRUModel
from src.utils.plot_data import (
    plot_buck_gru_components_tail,
    plot_compare_tail,
    plot_iLvC,
    plot_param_learning_progress,
    plot_u_vs_iL_vC,
)
from src.utils.save_log import create_result_dir, init_log, save_figure_to_log

# 乱数固定用の処理
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# %%
keep_log: bool = False


@dataclass(frozen=True)
class Params:
    data_path: str = path_list.path3

    # 真値パラメータ
    L_true: float = 223.6e-6
    C_true: float = 73.8e-6
    R_true: float = 5
    L_init = 200e-6
    C_init = 100e-6
    R_init = 8.0

    Vin: int = 10
    Vref: int = 5
    duty: float = Vref / Vin
    f_sw: float = 1e5  # スイッチング周波数
    T: float = 1 / f_sw  # 1周期の実時間
    cycles: int = 10
    cycles_sim: int = 1000  # この周期数分シミュレーションすれば定常であるとする
    samples_per_cycle: int = 200

    train_ratio = 0.3
    valid_ratio = 0.3
    # test_ratio = 1 - train_ratio - valid_ratio

    clipped_k: float = 2.0

    # 学習パラメータ

    # 異なるパラメータに異なる学習率を設定
    lr_l = 5e-2
    lr_c = 1e-2
    lr_r = 2e-2

    epochs = 50000

    # GRU学習用のデータ準備（時系列データ）
    seq_length = 10

    gru_lr = 0.001
    gru_epochs = 200


params = Params()


# %%
# ログ保存の初期化
if keep_log:
    result_dir = create_result_dir(
        "note3",
        base_dir="results/actual",
        config_dict=asdict(params),
    )
    log_path = init_log(result_dir, "actual_machine_notebooks/note3/note.ipynb")

# %% [markdown]
# # BuckConverterCell
#

# %% [markdown]
# ## 前処理
#
# 1. 実機データの読み込み
# 2. ダウンサンプリング
# 3. ノイズのクリッピング
# 4. train, valid, test に分割
#

# %%
# 実機データの読み込み
# t_raw, iL_raw, vC_raw = load_data(
#     path=params.data_path,
#     downsample_step=1,  # ダウンスケールしない
#     T=params.T,
#     cycles=params.cycles,
#     iL_label="CH2",
#     vC_label="CH1",
# )

# シミュレーションデータ
t_raw, iL_raw, vC_raw = load_data(
    path="../../data/buck_sim_Vin10.0_Vref5.0_fs100000.0_ppc200_cyc1000/buck_sim_Vin10.0_Vref5.0_fs100000.csv",
    skiprows=0,
    downsample_step=1,  # ダウンスケールしない
    T=params.T,
    cycles=params.cycles,
    start_time_offset=0,
    time_label="t",
    iL_label="iL",
    vC_label="vC",
)

# ダウンサンプリング
downsample_step = len(t_raw) // (params.samples_per_cycle * params.cycles)
t_downsampled: np.ndarray = t_raw[::downsample_step]
iL_downsampled: np.ndarray = iL_raw[::downsample_step]
vC_downsampled: np.ndarray = vC_raw[::downsample_step]


print(f"{len(t_downsampled)=}")
print(f"{len(iL_downsampled)=}")
print(f"{len(vC_downsampled)=}")


# ノイズのクリッピング
def clip_by_mean_std(x: np.ndarray, k: float) -> tuple[np.ndarray, float, float]:
    mu = float(np.mean(x))
    std = float(np.std(x))
    lo = mu - k * std
    hi = mu + k * std
    x_clipped = np.clip(x, lo, hi)
    return x_clipped.astype(x.dtype, copy=False), lo, hi


iL_clipped, iL_lo, iL_hi = clip_by_mean_std(iL_downsampled, k=params.clipped_k)
vC_clipped, vC_lo, vC_hi = clip_by_mean_std(vC_downsampled, k=params.clipped_k)

# プロット
fig, ax = plot_compare_tail(
    t_downsampled,
    iL_downsampled,
    vC_downsampled,
    "before clip",
    t_downsampled,
    iL_clipped,
    vC_clipped,
    "after clip",
    params.T,
    title="Comparison before and after clipping",
    iL_range=(iL_lo, iL_hi),
    vC_range=(vC_lo, vC_hi),
)
if keep_log:
    save_figure_to_log(fig, "compare_tail.png", result_dir)

# %%
# スイッチング信号
duty_phase = (t_downsampled[:-1] % params.T) / params.T
u = (duty_phase < params.duty).astype(int)

# 入力電圧
vs = np.ones(len(t_downsampled) - 1) * params.Vin

fig, ax = plot_u_vs_iL_vC(
    t_downsampled,
    u,
    vs,
    iL_clipped,
    vC_clipped,
    params.T,
)

if keep_log:
    save_figure_to_log(fig, "u_vs_iL_vC.png", result_dir)


# %%
# train, valid, test の分割
dt = t_downsampled[1:] - t_downsampled[:-1]

# numpy配列をテンソルに変換
dt_tensor = torch.tensor(dt, dtype=torch.float32)
u_tensor = torch.tensor(u, dtype=torch.float32)
vs_tensor = torch.tensor(vs, dtype=torch.float32)
iL_clipped_tensor = torch.tensor(iL_clipped, dtype=torch.float32)
vC_clipped_tensor = torch.tensor(vC_clipped, dtype=torch.float32)

# train
train_len = int(len(dt) * params.train_ratio)

t_train = t_downsampled[: train_len + 1]
dt_train = dt_tensor[:train_len]
u_train = u_tensor[:train_len]
vs_train = vs_tensor[:train_len]
iL_clipped_train = iL_clipped_tensor[: train_len + 1]
vC_clipped_train = vC_clipped_tensor[: train_len + 1]

h_clipped_train = torch.stack(
    [iL_clipped_train[:-1], vC_clipped_train[:-1]],
    dim=1,
)
x_train = torch.stack([vs_train, u_train, dt_train], dim=1)
target_clipped_train = torch.stack(
    [iL_clipped_train[1:], vC_clipped_train[1:]],
    dim=1,
)


# valid
valid_len = int(len(dt) * params.valid_ratio)

t_valid = t_downsampled[train_len : train_len + valid_len]
dt_valid = dt_tensor[train_len : train_len + valid_len]
u_valid = u_tensor[train_len : train_len + valid_len]
vs_valid = vs_tensor[train_len : train_len + valid_len]
iL_clipped_valid = iL_clipped_tensor[train_len : train_len + valid_len + 1]
vC_clipped_valid = vC_clipped_tensor[train_len : train_len + valid_len + 1]

h_clipped_valid = torch.stack(
    [iL_clipped_valid[:-1], vC_clipped_valid[:-1]],
    dim=1,
)
x_valid = torch.stack([vs_valid, u_valid, dt_valid], dim=1)
target_clipped_valid = torch.stack(
    [iL_clipped_valid[1:], vC_clipped_valid[1:]],
    dim=1,
)

# test
t_test = t_downsampled[train_len + valid_len :]
dt_test = dt_tensor[train_len + valid_len :]
u_test = u_tensor[train_len + valid_len :]
vs_test = vs_tensor[train_len + valid_len :]
iL_clipped_test = iL_clipped_tensor[train_len + valid_len :]
vC_clipped_test = vC_clipped_tensor[train_len + valid_len :]

h_clipped_test = torch.stack(
    [iL_clipped_test[:-1], vC_clipped_test[:-1]],
    dim=1,
)
x_test = torch.stack([vs_test, u_test, dt_test], dim=1)
target_clipped_test = torch.stack(
    [iL_clipped_test[1:], vC_clipped_test[1:]],
    dim=1,
)

print(f"{h_clipped_train.shape=}")
print(f"{x_train.shape=}")
print(f"{target_clipped_train.shape=}")

print(f"{h_clipped_valid.shape=}")
print(f"{x_valid.shape=}")
print(f"{target_clipped_valid.shape=}")

print(f"{h_clipped_test.shape=}")
print(f"{x_test.shape=}")
print(f"{target_clipped_test.shape=}")


# %% [markdown]
# ## 学習
#

# %%
buck_converter_cell = BuckConverterCell(
    L_init=params.L_init,
    C_init=params.C_init,
    R_init=params.R_init,
)

optimizer = optim.Adam(
    [
        {"params": [buck_converter_cell.log_L], "lr": params.lr_l},
        {"params": [buck_converter_cell.log_C], "lr": params.lr_c},
        {"params": [buck_converter_cell.log_R], "lr": params.lr_r},
    ]
)

loss_fn = nn.MSELoss()

# 損失履歴を保存
loss_history = {"train": [], "valid": []}
param_history = {"L": [], "C": [], "R": []}

# 学習ループ
for epoch in range(params.epochs):
    # 学習モード
    buck_converter_cell.train()
    optimizer.zero_grad()
    h_pred_train = buck_converter_cell(h_clipped_train, x_train)
    train_loss = loss_fn(h_pred_train, target_clipped_train)
    train_loss.backward()
    optimizer.step()

    loss_history["train"].append(train_loss.item())

    # 検証モード（勾配計算なし）
    buck_converter_cell.eval()
    with torch.no_grad():
        h_pred_valid = buck_converter_cell(h_clipped_valid, x_valid)
        valid_loss = loss_fn(h_pred_valid, target_clipped_valid)
        loss_history["valid"].append(valid_loss.item())

    # パラメータの履歴を保存
    param_history["L"].append(buck_converter_cell.log_L.exp().item())
    param_history["C"].append(buck_converter_cell.log_C.exp().item())
    param_history["R"].append(buck_converter_cell.log_R.exp().item())

    if epoch % 50 == 0:
        output_line = (
            f"Epoch {epoch}: Train Loss={train_loss.item():.6e}, "
            f"Valid Loss={valid_loss.item():.6e}\n"
            f"  L={buck_converter_cell.log_L.exp().item() * 1e6:.1f}µH "
            f"(true={params.L_true * 1e6:.1f})"
            f"C={buck_converter_cell.log_C.exp().item() * 1e6:.1f}µF "
            f"(true={params.C_true * 1e6:.1f})"
            f"R={buck_converter_cell.log_R.exp().item():.2f}Ω "
            f"(true={params.R_true:.1f})"
        )
        print(output_line)

# テストデータでの評価
buck_converter_cell.eval()
with torch.no_grad():
    h_pred_test = buck_converter_cell(h_clipped_test, x_test)
    test_loss = loss_fn(h_pred_test, target_clipped_test)

print(f"Final test loss: {test_loss.item():.6e}")


# %% [markdown]
# ## 結果確認
#

# %%
# Lossの遷移をグラフ表示
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

epochs_list = range(1, params.epochs + 1)
ax.plot(epochs_list, loss_history["train"], label="Train Loss", linewidth=2, alpha=0.8)
ax.plot(epochs_list, loss_history["valid"], label="Valid Loss", linewidth=2, alpha=0.8)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss (MSE)", fontsize=12)
ax.set_title("Loss Transition", fontsize=14, fontweight="bold")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.show()

# 画像をログに保存
if keep_log:
    save_figure_to_log(fig, "loss_transition", result_dir, "Lossの遷移")


# %%
# 回路パラメータの学習による変化をグラフ表示
fig, ax = plot_param_learning_progress(
    param_history,
    params.L_true,
    params.C_true,
    params.R_true,
    params.epochs,
)
plt.show()

# 画像をログに保存
if keep_log:
    save_figure_to_log(
        fig, "parameter_learning", result_dir, "回路パラメータの学習による変化"
    )


# %%
t_sim: np.ndarray = np.linspace(
    0,
    params.cycles_sim * params.T,
    params.cycles_sim * params.samples_per_cycle + 1,
)
t_additional = t_downsampled + t_sim[-1]
t_sim = np.concatenate([t_sim, t_additional])

dt_sim: np.ndarray = np.diff(t_sim)

# スイッチング信号
duty_phase = (t_sim[:-1] % params.T) / params.T
u_sim = (duty_phase < params.duty).astype(int)

# 入力電圧
vs_sim = np.ones(len(t_sim) - 1) * params.Vin
x_tensor = torch.tensor(np.c_[vs_sim, u_sim, dt_sim], dtype=torch.float32)

h_sim = [torch.zeros(1, 2)]

# シミュレーション実行
with torch.no_grad():
    for j in range(len(t_sim) - 1):
        h_current = buck_converter_cell(h_sim[-1], x_tensor[j : j + 1])
        h_sim.append(h_current)

h_sim_tensor = torch.cat(h_sim, dim=0)  # shape: (T+1, 2) みたいになる
iL_sim = h_sim_tensor[:, 0]
vC_sim = h_sim_tensor[:, 1]

# グラフ表示
fig, ax = plot_compare_tail(
    t_downsampled,
    iL_clipped,
    vC_clipped,
    "Measured",
    t_sim,
    iL_sim,
    vC_sim,
    "Predicted",
    T=params.T,
    title="Simulation Result",
)


if keep_log:
    save_figure_to_log(
        fig,
        "all_prediction",
        result_dir,
        f"シミュレーション結果(0から{params.cycles_sim}周期までシミュレーションし、定常箇所と実機のデータの比較)",
    )


# %%
fig_list = plot_iLvC(
    t_sim,
    iL_sim,
    vC_sim,
    T=params.T,
    title="Simulation Result",
)

# %%
# 一旦ここで中断
raise SystemExit("ここで処理を停止する")

# %%
# # sinカーブを2種類用意して末尾N周期だけplot_compare_tailで比較（例示）
# cycles: int = 100  # 100周期
# points_per_cycle: int = 1000  # 1周期あたりの点数

# # 時間軸, 2種類
# t1: np.ndarray = np.linspace(0, 2 * np.pi * cycles, points_per_cycle * cycles)
# t2: np.ndarray = np.linspace(0, 2 * np.pi * cycles, points_per_cycle * cycles)

# # sin波形
# y1: np.ndarray = np.sin(t1) + (t1 / t1[-1])
# y2: np.ndarray = np.sin(t2 + 0.2) + (t2 / t2[-1])  # 少し位相をずらしたsin波

# # vC(ここではy1/y2と同じ波形とする)
# iL1: np.ndarray = y1
# vC1: np.ndarray = y1
# iL2: np.ndarray = y2
# vC2: np.ndarray = y2


# fig_list1 = plot_iLvC(t1, iL1, vC1, T=2 * np.pi, title="sin(位相0) 末尾10周期")

# %% [markdown]
# # GRU
#

# %% [markdown]
# ## 前処理
#
# 1. 教師データ: (実機データのノイズクリッピング前の波形) - (学習済み BuckConverterCell のおよそ定常部分)
# 2. 入力データ: iL, vC, vs, u, dt
# 3. データシーケンスの作成
# 4. train, valid, test に分割
#


# %%
def create_sequences(
    h_data: torch.Tensor,
    x_data: torch.Tensor,
    noise_data: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    時系列データを作成
    入力: [iL, vC, vs, u, dt, iL_noise, vC_noise]
    出力: [iL_noise, vC_noise]
    """
    sequences = []
    targets = []

    for i in range(len(h_data) - seq_len):
        # 入力特徴量: [iL, vC, vs, u, dt] + 過去のノイズ [iL_noise, vC_noise]
        seq_input = []
        for j in range(seq_len):
            idx = i + j
            # 状態と入力
            features = torch.cat(
                [
                    h_data[idx],  # [iL, vC]
                    x_data[idx],  # [vs, u, dt]
                ]
            )
            # 過去のノイズ（存在する場合）
            if idx > 0:
                noise_features = noise_data[idx - 1]  # [iL_noise, vC_noise]
            else:
                noise_features = torch.zeros(2)
            # 結合: [iL, vC, vs, u, dt, iL_noise, vC_noise] = 7次元
            combined = torch.cat([features, noise_features])
            seq_input.append(combined)

        sequences.append(torch.stack(seq_input))
        targets.append(noise_data[i + seq_len])

    return torch.stack(sequences), torch.stack(targets)


# t_downsampled 部分を切り出す
h_sim_tensor = h_sim_tensor[1000 * params.samples_per_cycle + 1 :]
h_sim_tensor = h_sim_tensor[:-1]

iL_tensor = torch.tensor(iL_downsampled, dtype=torch.float32)
vC_tensor = torch.tensor(vC_downsampled, dtype=torch.float32)
h_meas_tensor = torch.stack([iL_tensor[:-1], vC_tensor[:-1]], dim=1)

n = min(h_meas_tensor.shape[0], h_sim_tensor.shape[0])
h_noise = h_meas_tensor[:n] - h_sim_tensor[:n]

# データセットを作成
iL_train = iL_tensor[: train_len + 1]
vC_train = vC_tensor[: train_len + 1]
h_train = torch.stack([iL_train[:-1], vC_train[:-1]], dim=1)
X_train_seq, y_train_seq = create_sequences(
    h_train,
    x_train,
    h_noise[: len(h_train)],
    params.seq_length,
)

iL_valid = iL_tensor[train_len : train_len + valid_len + 1]
vC_valid = vC_tensor[train_len : train_len + valid_len + 1]
h_valid = torch.stack([iL_valid[:-1], vC_valid[:-1]], dim=1)
X_valid_seq, y_valid_seq = create_sequences(
    h_valid,
    x_valid,
    h_noise[len(h_train) : len(h_train) + len(h_valid)],
    params.seq_length,
)

iL_test = iL_tensor[train_len + valid_len :]
vC_test = vC_tensor[train_len + valid_len :]
h_test = torch.stack([iL_test[:-1], vC_test[:-1]], dim=1)
X_test_seq, y_test_seq = create_sequences(
    h_test,
    x_test,
    h_noise[len(h_train) + len(h_valid) :],
    params.seq_length,
)

print(f"学習データ: {X_train_seq.shape} -> {y_train_seq.shape}")
print(f"検証データ: {X_valid_seq.shape} -> {y_valid_seq.shape}")
print(f"テストデータ: {X_test_seq.shape} -> {y_test_seq.shape}")
print(
    f"入力特徴量の次元: {X_train_seq.shape[2]} (iL, vC, vs, u, dt, iL_noise, vC_noise)"
)

# %%
# 学習データ
X_train_seq_np: np.ndarray = X_train_seq.detach().cpu().numpy()
y_train_seq_np: np.ndarray = y_train_seq.detach().cpu().numpy()

# 入力特徴量名と単位
feature_names: list[str] = ["iL", "vC", "vs", "u", "dt", "iL_noise", "vC_noise"]
feature_units: list[str] = ["[A]", "[V]", "[V]", "", "[s]", "[A]", "[V]"]


fig, axs = plt.subplots(7, 1, figsize=(14, 12), sharex=True)

last_timestep_features: np.ndarray = X_train_seq_np[:, -1, :]  # [N, 7]

# time_axisをt_train（μs単位）に
time_axis: np.ndarray = t_train[: X_train_seq_np.shape[0]] * 1e6  # μs

for i in range(7):
    axs[i].plot(time_axis, last_timestep_features[:, i], linewidth=1.5, alpha=0.7)
    axs[i].set_ylabel(f"{feature_names[i]} {feature_units[i]}", fontsize=11)
    axs[i].grid(True, alpha=0.3)
    axs[i].set_title(
        f"GRU Training Data: {feature_names[i]} (Value at Last Time Step)",
        fontsize=12,
        fontweight="bold",
    )

axs[6].set_xlabel("Time [μs]", fontsize=12)
plt.tight_layout()
plt.show()

if keep_log:
    save_figure_to_log(
        fig,
        "gru_training_data_features",
        result_dir,
        "GRU 学習データ",
    )


# %% [markdown]
# ## 学習
#

# %%
# 学習
gru_model = GRUModel(
    input_size=X_train_seq.shape[2],
    hidden_size=32,
    num_layers=2,
    output_size=2,
    seq_length=params.seq_length,
)


gru_optimizer = optim.Adam(gru_model.parameters(), lr=params.gru_lr)
gru_loss_fn = nn.MSELoss()

gru_loss_history = {"train": [], "valid": []}

for epoch in range(params.gru_epochs):
    # 学習モード
    gru_model.train()
    gru_optimizer.zero_grad()
    noise_pred_train = gru_model(X_train_seq)
    train_loss = gru_loss_fn(noise_pred_train, y_train_seq)
    train_loss.backward()
    gru_optimizer.step()

    gru_loss_history["train"].append(train_loss.item())

    # 検証モード
    gru_model.eval()
    with torch.no_grad():
        noise_pred_valid = gru_model(X_valid_seq)
        valid_loss = gru_loss_fn(noise_pred_valid, y_valid_seq)
        gru_loss_history["valid"].append(valid_loss.item())

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1}/{params.gru_epochs}: "
            f"Train Loss={train_loss.item():.6e}, "
            f"Valid Loss={valid_loss.item():.6e}"
        )

gru_model.eval()
with torch.no_grad():
    noise_pred_test = gru_model(X_test_seq)
    test_loss = gru_loss_fn(noise_pred_test, y_test_seq)
    print(f"Test Loss={test_loss.item():.6e}")


# %% [markdown]
# ## 結果確認
#

# %%
# Lossの遷移をグラフ表示
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

epochs_list = range(1, params.gru_epochs + 1)
ax.plot(
    epochs_list, gru_loss_history["train"], label="Train Loss", linewidth=2, alpha=0.8
)
ax.plot(
    epochs_list, gru_loss_history["valid"], label="Valid Loss", linewidth=2, alpha=0.8
)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss (MSE)", fontsize=12)
ax.set_title("Loss Transition", fontsize=14, fontweight="bold")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.show()

# 画像をログに保存
if keep_log:
    save_figure_to_log(fig, "gru_loss_transition", result_dir, "GRU Lossの遷移")


# %%
# GRUでノイズを予測（テスト）
gru_model.eval()
with torch.no_grad():
    noise_pred_test = gru_model(X_test_seq)


noise_pred_test_cpu = noise_pred_test.detach().cpu()
noise_true_test_cpu = y_test_seq.detach().cpu()

# plot_compare_tail を使って比較（末尾N周期）
seq_len = int(params.seq_length)

t_noise = np.asarray(t_test[:-1])[seq_len:]
min_len = min(
    t_noise.shape[0],
    int(noise_true_test_cpu.shape[0]),
    int(noise_pred_test_cpu.shape[0]),
)

t_noise = t_noise[:min_len]
true_noise = noise_true_test_cpu[:min_len].numpy()
pred_noise = noise_pred_test_cpu[:min_len].numpy()

fig1, _ = plot_compare_tail(
    t1=t_noise,
    iL1=true_noise[:, 0],
    vC1=true_noise[:, 1],
    label1="true noise",
    t2=t_noise,
    iL2=pred_noise[:, 0],
    vC2=pred_noise[:, 1],
    label2="pred noise (GRU)",
    T=float(params.T),
    N_cycles=4,
    title="GRU noise prediction (test)",
)


if keep_log:
    save_figure_to_log(
        fig1,
        "gru_noise_pred_test_tail10T",
        result_dir,
        "GRU noise予測(テスト, tail10T)",
    )


# %% [markdown]
# # BuckConverterCell + GRU の結果
#

# %%
# プロット
N_cycles: int = 4
T: float = float(params.T)
seq_len: int = int(params.seq_length)

# --- BuckConverterCell: 実機時間(t_downsampled)に対応する末尾部分を抽出 ---
n_meas: int = int(len(t_downsampled))
t_sim_meas_raw: np.ndarray = np.asarray(t_sim[-n_meas:], dtype=float)

iL_sim_meas: np.ndarray = np.asarray(
    iL_sim[-n_meas:].detach().cpu().numpy(), dtype=float
)
vC_sim_meas: np.ndarray = np.asarray(
    vC_sim[-n_meas:].detach().cpu().numpy(), dtype=float
)

# t_sim末尾は (t_downsampled + offset) なので、offset を引いて合わせる
offset: float = float(t_sim_meas_raw[0] - float(t_downsampled[0]))
t_sim_meas: np.ndarray = t_sim_meas_raw - offset  # ≒ t_downsampled

# --- test区間をBuck側の時間で切り出す ---
t_test_np: np.ndarray = np.asarray(t_test, dtype=float)
start_idx: int = int(np.argmin(np.abs(t_sim_meas - float(t_test_np[0]))))
end_idx: int = min(start_idx + int(t_test_np.shape[0]), int(t_sim_meas.shape[0]))

t_test_used: np.ndarray = t_sim_meas[start_idx:end_idx]
iL_sim_test: np.ndarray = iL_sim_meas[start_idx:end_idx]
vC_sim_test: np.ndarray = vC_sim_meas[start_idx:end_idx]

# --- Measured(test) を同じ長さだけ取り出す ---
start_test_idx: int = int(train_len + valid_len)  # t_testと同じ開始位置

iL_meas_test: np.ndarray = np.asarray(
    iL_downsampled[start_test_idx : start_test_idx + int(t_test_used.shape[0])],
    dtype=float,
)
vC_meas_test: np.ndarray = np.asarray(
    vC_downsampled[start_test_idx : start_test_idx + int(t_test_used.shape[0])],
    dtype=float,
)

# --- GRU出力(pred_noise)は t_test[:-1][seq_len:] に対応 ---
t_noise_local: np.ndarray = np.asarray(t_test_used[:-1], dtype=float)[seq_len:]
pred_noise_np: np.ndarray = np.asarray(pred_noise, dtype=float)

min_len2: int = int(
    min(
        t_noise_local.shape[0],
        pred_noise_np.shape[0],
        iL_sim_test.shape[0] - 1 - seq_len,
        vC_sim_test.shape[0] - 1 - seq_len,
        iL_meas_test.shape[0] - 1 - seq_len,
        vC_meas_test.shape[0] - 1 - seq_len,
    )
)

if min_len2 <= 0:
    raise ValueError(
        "長さが合わず、描画できません。"
        " t_test/t_sim/pred_noise/seq_length の整合を確認してください。"
    )

# すべて同じ長さに揃える
# t は [s]
t_view: np.ndarray = t_noise_local[:min_len2]

iL_meas: np.ndarray = np.asarray(iL_meas_test[:-1], dtype=float)[
    seq_len : seq_len + min_len2
]
vC_meas: np.ndarray = np.asarray(vC_meas_test[:-1], dtype=float)[
    seq_len : seq_len + min_len2
]

iL_buck: np.ndarray = np.asarray(iL_sim_test[:-1], dtype=float)[
    seq_len : seq_len + min_len2
]
vC_buck: np.ndarray = np.asarray(vC_sim_test[:-1], dtype=float)[
    seq_len : seq_len + min_len2
]

iL_gru: np.ndarray = pred_noise_np[:min_len2, 0]
vC_gru: np.ndarray = pred_noise_np[:min_len2, 1]

fig_iL, _, fig_vC, _ = plot_buck_gru_components_tail(
    t=t_view,
    iL_meas=iL_meas,
    vC_meas=vC_meas,
    iL_buck=iL_buck,
    vC_buck=vC_buck,
    iL_gru=iL_gru,
    vC_gru=vC_gru,
    T=T,
    N_cycles=float(N_cycles),
    title="note3 (test)",
    include_overlay=True,
)
plt.show()

if keep_log:
    save_figure_to_log(
        fig_iL,
        "tail4T_iL_meas_buck_gru_sum",
        result_dir,
        "iL: Measured / Buck / GRU / Buck+GRU（末尾4周期）",
    )
    save_figure_to_log(
        fig_vC,
        "tail4T_vC_meas_buck_gru_sum",
        result_dir,
        "vC: Measured / Buck / GRU / Buck+GRU（末尾4周期）",
    )
