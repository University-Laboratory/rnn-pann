# actual_machine_notebooks/note4/note.ipynb 実行ログ

実行日時: 2025-12-24 18:33:23

---

# 主な変更点

1. 今まで duty を 0.5 で固定していたが、それだと BuckConverterCell の出力がほぼ Vin\*0.5 で固定されてしまうので、実機データの vC の平均値を用いて duty を計算するようにした
2. 実機データの iL と vC に FIR フィルターをかけて滑らかになるようにしてから BuckConverterCell の学習に使うようにした

## 実機データ(加工前)

![実機データ(加工前)_0](<images/実機データ(加工前)_0.png>)

![実機データ(加工前)_1](<images/実機データ(加工前)_1.png>)

## BuckConverterCell の入力波形

FIR フィルターをかけてからダウンサンプリング

scipy の firwin を使って FIR フィルターをかけている。
[scipy.signal.firwin](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html)

![BuckConverterCellの入力波形](images/BuckConverterCellの入力波形.png)

## BuckConverterCell の学習結果 Loss の遷移

![BuckConverterCellの学習結果_Lossの遷移](images/BuckConverterCellの学習結果_Lossの遷移.png)

## 回路パラメータの学習による変化

![parameter_learning](images/parameter_learning.png)

## シミュレーション結果(0 から 1000 周期までシミュレーションし、定常箇所と実機のデータの比較)

![all_prediction](images/all_prediction.png)

## シミュレーション結果

![シミュレーション結果_0](images/シミュレーション結果_0.png)

![シミュレーション結果_1](images/シミュレーション結果_1.png)

![シミュレーション結果_2](images/シミュレーション結果_2.png)

![シミュレーション結果_3](images/シミュレーション結果_3.png)

## GRU 学習データ

![gru_training_data_features](images/gru_training_data_features.png)

## GRU Loss の遷移

![gru_loss_transition](images/gru_loss_transition.png)

## GRU noise 予測(テスト)

![gru_noise_pred_test](images/gru_noise_pred_test.png)

## iL: Measured / Buck / GRU / Buck+GRU（末尾 4 周期）

![tail4T_iL_meas_buck_gru_sum](images/tail4T_iL_meas_buck_gru_sum.png)

## vC: Measured / Buck / GRU / Buck+GRU（末尾 4 周期）

![tail4T_vC_meas_buck_gru_sum](images/tail4T_vC_meas_buck_gru_sum.png)
