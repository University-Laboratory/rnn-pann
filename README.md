# PANN 頑張るぞ！

## 環境構築

- Python の仮想環境管理ツール [uv](https://github.com/astral-sh/uv) を使用しています

### セットアップ手順

1. [uv](https://github.com/astral-sh/uv) をインストール
   - 推奨: `pip install uv`
2. プロジェクトルートで仮想環境と依存パッケージをセットアップ

```
uv venv
uv sync
source .venv/bin/activate
```

3. ruff でコードチェック

```
uv run ruff check . --fix
```

- 念の為に`requirements.txt`に使用しているパッケージを記載しておきます
- 新しいノートや開発物は notebooks フォルダ内に保存してください

## 概要

- PANN のコードを参考に、降圧型 DC-DC コンバータのシミュレーションを行う
- シミュレーションデータを作成し、回路パラメータの推論を行う
- 推論した回路パラメータを使ってシミュレーションを行い、波形を比較する

## シミュレーション波形での動作確認

[完成版(note17.ipynb)](notebooks/note17.ipynb)

[note17.py に python ファイル作成](notebooks/note17.py)

[実行方法について](notebooks/note17.md)

## 実機データを使った学習

[actual_machine_notebooks](actual_machine_notebooks/note1/note1.ipynb)
