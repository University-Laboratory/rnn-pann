# note17.py で行っていること

- オイラー法を用いたシミュレーションにより、u(t), vs, iL(t), vC(t) を計算
- シミュレーションデータを使ってモデルを学習させ、L, C, R を推定
- 結果を results ディレクトリに保存

# note17.py の実行方法

### 1. 回路のパラメータ等を変更

note17.py の 27 ~ 55 行目 を変更

### 2. 環境を構築

requires-python = ">=3.12"

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 実行

```bash
python notebooks/note17.py
```

### 4. 結果を確認

実行時に表示される保存先にログが保存される

```text
ログ保存先: /rnn-pann/results/20251203_160816_352_note17/log.md
実行中です...
finish
ログ保存先: /rnn-pann/results/20251203_160816_352_note17/log.md
実行が完了しました
```
