# actual_machine_notebooks/note3/note_sim.ipynb 実行ログ

実行日時: 2025-12-20 16:32:51

---

![u_vs_iL_vC.png](images/u_vs_iL_vC.png)

## Lossの遷移

![loss_transition](images/loss_transition.png)

## 回路パラメータの学習による変化

![parameter_learning](images/parameter_learning.png)

## シミュレーション結果(0から1000周期までシミュレーションし、定常箇所と実機のデータの比較)

![all_prediction](images/all_prediction.png)

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

