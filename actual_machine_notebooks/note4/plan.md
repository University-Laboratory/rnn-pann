## 手順 0：現状を固定して“比較基準”を作る（1 回だけ）

- いまの teacher forcing 学習のまま、

  - (A) 1-step 予測損失（いまの train_loss）
  - (B) rollout 損失（後で作る関数で、例えば 1 周期=200 点ぶん回して MSE）
    を両方ログ出ししておく。
    → 後で「改修が効いているか」が一発でわかる。

---

## 手順 1：BuckConverterCell を roll-out で回す関数を追加（学習はまだ変えない）

### 1.1 まず“推論用”に rollout 関数を作る

- 入力：`h0`（初期状態）と `x_seq`（[T,3]）
- 出力：予測状態列 `h_pred`（[T,2]）

（前回の `rollout_cell` をそのまま `src/utils/models.py` か notebook に追加）

### 1.2 教師データと同じ区間で rollout を可視化

- `h0 = [iL_meas[0], vC_meas[0]]` を使う
- `x_seq = x_train[:T]`（例えば末尾 4 周期分だけ）で回す
- 予測と実測を重ね描き
  → **ここでズレが出るのが正常**（teacher forcing と違うため）

この段階では「関数が正しく回っているか」だけ確認する。

---

## 手順 2：Buck の学習ループを “窓 rollout 損失” に置き換える（ここが本丸）

いきなり全長を回すと重い＆勾配が不安定になりやすいので、**窓（window）で学習**する。

### 2.1 window 長を決める

- 最初は `win_len = 50`（=0.25 周期）くらい
- 安定したら 100→200 と伸ばす

### 2.2 データを「状態列」と「入力列」に整理

- 状態列 `h_series` は長さ N+1（h[0]..h[N]）
- 入力列 `x_series` は長さ N（x[0]..x[N-1]）

あなたはすでに `iL_*` と `vC_*` を `+1` の長さで持っているので、

```python
h_train_series = torch.stack([iL_train, vC_train], dim=1)  # [N+1,2]
x_train_series = x_train  # [N,3]
```

の形にする。

### 2.3 窓を作る関数を用意（make_windows）

- `h0_batch: [B,2]`
- `x_batch: [B,win_len,3]`
- `y_batch: [B,win_len,2]`（真の次状態列）

### 2.4 学習ループを置き換え

- loss は `rollout_mse_loss(cell, h0_b, x_b, y_b)` の 1 本にする
- まずは train だけ動けば OK（valid は後で）

### 2.5 ここで必ずやる検証

- teacher forcing の損失は悪化してよい（むしろ気にしない）
- rollout 損失が下がり、rollout 波形が改善しているかを見る

---

## 手順 3：GRU の入力スケーリング（標準化）を追加（GRU 学習自体は同じで OK）

GRU でまず事故るのは **dt の桁**なので、最低限の無次元化だけでも効果がある。

### 3.1 まずは“無次元化”を採用（実装が簡単）

- `vs_scaled = vs / Vin`（だいたい 1 付近）
- `vC_scaled = vC / Vref`（だいたい 1 付近）
- `iL_scaled = iL / I_base`（I_base は平均電流やリップル幅で適当に）
- `dt_scaled = dt * f_sw`（ほぼ dt/T なので 0〜1 付近）
- `u` はそのまま（0/1）

ノイズも同様にスケールする（`iL_noise/I_base`, `vC_noise/Vref`）。

### 3.2 標準化（平均 0 分散 1）は次の段階で OK

無次元化で学習が安定してきたら、train 統計で標準化を入れるのが理想。

- 平均・標準偏差は train だけから計算
- valid/test にも同じ変換を適用

実装は「transform 関数」を作って、GRU 用特徴量生成の直前に一括適用が安全。

---

## 最短で成功しやすい“実装順”

1. **rollout 推論関数を作って波形が回ることを確認**
2. **Buck 学習を win_len rollout 損失に置換（win_len=50）**
3. **Buck の rollout 波形が改善するまで調整（学習率/epoch）**
4. **GRU 入力を無次元化（特に dt）**

---

## すぐにやるべき注意点（地雷回避）

- rollout 学習にすると勾配が長くなるので、**win_len は短く開始**する
- 学習率は今のままだと発散する場合がある

  - まず `lr_l, lr_c, lr_r` を **1 桁下げる**（例：1e-2→1e-3）と安定しやすい

- valid も rollout で評価する（teacher forcing valid は意味が薄い）
