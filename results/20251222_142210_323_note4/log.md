# actual_machine_notebooks/note4/note_sim.ipynb 実行ログ

実行日時: 2025-12-22 14:22:10

---

## Params

```
C_init: 0.0001
C_true: 7.5e-05
L_init: 0.0002
L_true: 0.00022
R_init: 8.0
R_true: 5.0
T: 1e-05
Vin: 10.0
Vref: 5.0
batch_size: 256
cycles: 1200
duty: 0.5
epochs_buck: 8000
epochs_gru: 300
f_sw: 100000.0
grad_clip_norm: 5.0
gru_lr: 0.001
lr_c: 0.001
lr_l: 0.005
lr_r: 0.002
noise_std_iL: 0.02
noise_std_vC: 0.02
samples_per_cycle: 200
seq_length: 10
train_cycles: 10
train_ratio: 0.5
valid_ratio: 0.25
win_len: 50
```

## 擬似計測波形

![sim_pseudo_measured](images/sim_pseudo_measured.png)

## 学習データ（末尾10周期）

![training_data_tail](images/training_data_tail.png)

## Buck 推定値

```
L_hat=2.200003e-04
C_hat=7.543847e-05
R_hat=5.000073e+00

```

## Buck 損失遷移

![buck_loss_transition](images/buck_loss_transition.png)

## Buck パラメータ推定の推移

![buck_param_learning](images/buck_param_learning.png)

## Buck rollout比較(train)

![buck_rollout_train](images/buck_rollout_train.png)

## Buck rollout比較(valid)

![buck_rollout_valid](images/buck_rollout_valid.png)

## Buck rollout比較(test)

![buck_rollout_test](images/buck_rollout_test.png)

## GRU 学習データ（特徴量・全体）

![gru_training_data_features](images/gru_training_data_features.png)

## GRU 損失遷移

![gru_loss_transition](images/gru_loss_transition.png)

## GRU 評価

```
GRU test loss=3.685277e-07

```

## iL: Measured / Buck / GRU / Buck+GRU（末尾4周期）

![tail4T_iL_meas_buck_gru_sum](images/tail4T_iL_meas_buck_gru_sum.png)

## vC: Measured / Buck / GRU / Buck+GRU（末尾4周期）

![tail4T_vC_meas_buck_gru_sum](images/tail4T_vC_meas_buck_gru_sum.png)

## Buck+GRU 評価(test)

```
MSE buck=8.811593e-12
MSE buck+gru=8.608980e-06

```

