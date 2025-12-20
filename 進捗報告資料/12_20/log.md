# actual_machine_notebooks/note3/note_sim.ipynb 実行ログ

[詳細](../../actual_machine_notebooks/note3/note_sim.ipynb)

実行日時: 2025-12-20 16:32:51

---

# シミュレーションして学習データ作成

![u_vs_iL_vC.png](images/u_vs_iL_vC.png)

## BuckConverterCell の学習結果: Loss の遷移

![loss_transition](images/loss_transition.png)

## 回路パラメータの学習による変化

![parameter_learning](images/parameter_learning.png)

エポックを 1000 でやっていたが、C が収束しづらかったので 50000 に増やした

## シミュレーション結果

![all_prediction](images/all_prediction.png)
ほぼほぼ一致

## GRU 学習データ

![gru_training_data_features](images/gru_training_data_features.png)

iL_noise, vC_noise は 10^-6 オーダーで、一応ちょっとだけズレてる

## GRU Loss の遷移

![gru_loss_transition](images/gru_loss_transition.png)

## GRU noise 予測(テスト, tail10T)

![gru_noise_pred_test_tail10T](images/gru_noise_pred_test_tail10T.png)
意味不明な予測結果になっている

iLの方は、まだ誤差と言える範囲な気がするが、vCの方は、BuckConverterCellの予測結果を明らかに変えてしまうほどの出力になってる
## iL: Measured / Buck / GRU / Buck+GRU（末尾 4 周期）

![tail4T_iL_meas_buck_gru_sum](images/tail4T_iL_meas_buck_gru_sum.png)

## vC: Measured / Buck / GRU / Buck+GRU（末尾 4 周期）

![tail4T_vC_meas_buck_gru_sum](images/tail4T_vC_meas_buck_gru_sum.png)
