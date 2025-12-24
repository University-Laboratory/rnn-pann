# actual_machine_notebooks/note4/note.ipynb 実行ログ

実行日時: 2025-12-24 18:20:48

---

![実機データ(加工前)](images/実機データ(加工前).png)

![BuckConverterCellの入力波形](images/BuckConverterCellの入力波形.png)

![BuckConverterCellの学習結果 Lossの遷移](images/BuckConverterCellの学習結果 Lossの遷移.png)

## 回路パラメータの学習による変化

![parameter_learning](images/parameter_learning.png)

## シミュレーション結果(0から1000周期までシミュレーションし、定常箇所と実機のデータの比較)

![all_prediction](images/all_prediction.png)

![シミュレーション結果](images/シミュレーション結果.png)

## GRU 学習データ

![gru_training_data_features](images/gru_training_data_features.png)

## GRU Lossの遷移

![gru_loss_transition](images/gru_loss_transition.png)

## GRU noise予測(テスト, tail10T)

![gru_noise_pred_test_tail10T](images/gru_noise_pred_test_tail10T.png)

## iL: Measured / Buck / GRU / Buck+GRU（末尾4周期）

![tail4T_iL_meas_buck_gru_sum](images/tail4T_iL_meas_buck_gru_sum.png)

## vC: Measured / Buck / GRU / Buck+GRU（末尾4周期）

![tail4T_vC_meas_buck_gru_sum](images/tail4T_vC_meas_buck_gru_sum.png)

