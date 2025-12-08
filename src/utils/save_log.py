import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def _get_project_root() -> Path:
    """
    プロジェクトのルートディレクトリを取得する

    Returns:
        プロジェクトルートのPathオブジェクト
    """
    # このファイル（save_log.py）の位置からプロジェクトルートを取得
    # src/utils/save_log.py -> プロジェクトルート（2階層上）
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def calculate_config_hash(config_dict: dict[str, float | int]) -> str:
    """
    実験設定の辞書からハッシュ値を計算する

    Args:
        config_dict: 実験設定の辞書
            - L_true, C_true, R_true（真値）
            - Vin, Vref（入力条件）
            - f_sw, points_per_cycle, cycles（シミュレーション設定）
            - tail_len（データ長）
            - train_ratio, valid_ratio（データ分割）
            - L_init, C_init, R_init（初期値）
            - lr_L, lr_C, lr_R（学習率）
            - epochs（エポック数）

    Returns:
        ハッシュ値（16進数文字列）
    """
    # パラメータをソートして正規化
    sorted_items = sorted(config_dict.items())
    # 浮動小数点数の精度を考慮して文字列化
    config_str = json.dumps(sorted_items, sort_keys=True, ensure_ascii=False)
    # SHA256でハッシュ化
    hash_obj = hashlib.sha256(config_str.encode("utf-8"))
    return hash_obj.hexdigest()


def find_existing_result_dir(
    config_hash: str, notebook_name: str, base_dir: str = "results"
) -> Path | None:
    """
    既存の結果ディレクトリを検索する

    Args:
        config_hash: 設定のハッシュ値
        notebook_name: ノートブック名
        base_dir: ベースディレクトリ（デフォルト: "results"）

    Returns:
        一致するディレクトリが見つかった場合はそのPath、見つからない場合はNone
    """
    project_root = _get_project_root()
    base_path = project_root / base_dir

    # ベースディレクトリが存在しない場合はNoneを返す
    if not base_path.exists():
        return None

    # 既存のディレクトリを走査
    for result_dir in base_path.iterdir():
        if not result_dir.is_dir():
            continue
        # ノートブック名が一致するか確認
        if not result_dir.name.endswith(f"_{notebook_name}"):
            continue

        # config_hash.jsonファイルを読み込む
        config_hash_path = result_dir / "config_hash.json"
        if not config_hash_path.exists():
            continue

        try:
            with open(config_hash_path, encoding="utf-8") as f:
                saved_hash = json.load(f).get("hash")
                if saved_hash == config_hash:
                    return result_dir
        except (json.JSONDecodeError, KeyError):
            continue

    return None


def create_result_dir(
    notebook_name: str,
    base_dir: str = "results",
    config_dict: dict[str, float | int] | None = None,
) -> Path:
    """
    タイムスタンプ付きの結果ディレクトリを作成する
    設定辞書が提供された場合、既存の同じ設定のディレクトリがあればそれを使用する

    Args:
        notebook_name: ノートブック名（例: "note17"）
        base_dir: ベースディレクトリ（デフォルト: "results"）
        config_dict: 実験設定の辞書（オプション）
            提供された場合、同じ設定の既存ディレクトリを検索する

    Returns:
        作成された、または既存の結果ディレクトリのPathオブジェクト
    """
    # プロジェクトルートを取得
    project_root = _get_project_root()

    # 設定辞書が提供された場合、既存ディレクトリを検索
    if config_dict is not None:
        config_hash = calculate_config_hash(config_dict)
        existing_dir = find_existing_result_dir(config_hash, notebook_name, base_dir)
        if existing_dir is not None:
            return existing_dir

    # タイムスタンプを生成（yyyyMMdd_HHMMSS_mmm 形式，ミリ秒単位まで含める）
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_") + f"{int(now.microsecond / 1000):03d}"

    # ディレクトリ名を生成（例: 20251203_145714_note17）
    dir_name = f"{timestamp}_{notebook_name}"

    # ベースディレクトリが存在しない場合は作成（プロジェクトルート基準）
    base_path = project_root / base_dir
    base_path.mkdir(exist_ok=True)

    # 結果ディレクトリを作成
    result_dir = base_path / dir_name
    result_dir.mkdir(exist_ok=True)

    # imagesディレクトリを作成
    images_dir = result_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # 設定辞書が提供された場合、ハッシュ値を保存
    if config_dict is not None:
        config_hash = calculate_config_hash(config_dict)
        config_hash_path = result_dir / "config_hash.json"
        with open(config_hash_path, "w", encoding="utf-8") as f:
            json.dump({"hash": config_hash, "config": config_dict}, f, indent=2)

    return result_dir


def save_image(
    fig: plt.Figure,
    filename: str,
    result_dir: Path,
    dpi: int = 300,
) -> Path:
    """
    画像を保存する

    Args:
        fig: matplotlibのFigureオブジェクト
        filename: 保存するファイル名（拡張子なしでも可）
        result_dir: 結果ディレクトリのPathオブジェクト
        dpi: 画像の解像度（デフォルト: 300）

    Returns:
        保存された画像ファイルのPathオブジェクト
    """
    # 拡張子がない場合は.pngを追加
    if not filename.endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
        filename = f"{filename}.png"

    # imagesディレクトリに保存
    images_dir = result_dir / "images"
    image_path = images_dir / filename

    fig.savefig(image_path, dpi=dpi, bbox_inches="tight")
    return image_path


def init_log(result_dir: Path, notebook_name: str) -> Path:
    """
    ログファイルを初期化する
    既存のログファイルが存在する場合は追記モードで新しい実行日時を追加する

    Args:
        result_dir: 結果ディレクトリのPathオブジェクト
        notebook_name: ノートブック名

    Returns:
        ログファイルのPathオブジェクト
    """
    log_path = result_dir / "log.md"

    # 既存のログファイルが存在するか確認
    if log_path.exists():
        # 追記モードで新しい実行日時を追加
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = "\n---\n\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(separator)
            f.write(f"実行日時: {timestamp}\n\n")
    else:
        # 新規作成
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""# {notebook_name} 実行ログ

実行日時: {timestamp}

---

"""
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header)

    return log_path


def append_log(
    result_dir: Path,
    content: str,
    image_paths: list[Path] | None = None,
) -> None:
    """
    ログファイルに内容を追加する

    Args:
        result_dir: 結果ディレクトリのPathオブジェクト
        content: 追加するテキスト内容
        image_paths: 追加する画像ファイルのパスリスト（相対パスで保存）
    """
    log_path = result_dir / "log.md"

    with open(log_path, "a", encoding="utf-8") as f:
        # テキスト内容を追加
        f.write(content)
        f.write("\n\n")

        # 画像を追加
        if image_paths:
            for img_path in image_paths:
                # 相対パスに変換（log.mdからの相対パス）
                relative_path = img_path.relative_to(result_dir)
                f.write(f"![{img_path.stem}]({relative_path})\n\n")


def save_text_output(
    result_dir: Path,
    text: str,
    title: str | None = None,
) -> None:
    """
    テキスト出力をログに保存する

    Args:
        result_dir: 結果ディレクトリのPathオブジェクト
        text: 保存するテキスト
        title: セクションタイトル（オプション）
    """
    log_path = result_dir / "log.md"

    with open(log_path, "a", encoding="utf-8") as f:
        if title:
            f.write(f"## {title}\n\n")
        # コードブロックとして保存
        f.write("```\n")
        f.write(text)
        f.write("\n```\n\n")


def save_figure_to_log(
    fig: plt.Figure,
    filename: str,
    result_dir: Path,
    title: str | None = None,
    dpi: int = 300,
) -> None:
    """
    図を保存してログに追加する

    Args:
        fig: matplotlibのFigureオブジェクト
        filename: 保存するファイル名
        result_dir: 結果ディレクトリのPathオブジェクト
        title: セクションタイトル（オプション）
        dpi: 画像の解像度（デフォルト: 300）
    """
    # 画像を保存
    image_path = save_image(fig, filename, result_dir, dpi)

    # ログに追加
    log_path = result_dir / "log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        if title:
            f.write(f"## {title}\n\n")
        relative_path = image_path.relative_to(result_dir)
        f.write(f"![{filename}]({relative_path})\n\n")


def save_graphviz_to_log(
    graphviz_graph: object,
    filename: str,
    result_dir: Path,
    title: str | None = None,
    format: str = "png",
) -> Path:
    """
    Graphvizのグラフを保存してログに追加する
    torchviewのdraw_graphで生成されたグラフなどに対応

    Args:
        graphviz_graph: Graphvizのグラフオブジェクト（render()メソッドを持つ）
        filename: 保存するファイル名（拡張子なし）
        result_dir: 結果ディレクトリのPathオブジェクト
        title: セクションタイトル（オプション）
        format: 画像フォーマット（デフォルト: "png"）

    Returns:
        保存された画像ファイルのPathオブジェクト
    """
    # imagesディレクトリに保存
    images_dir = result_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # ファイルパスを構築
    image_path = images_dir / f"{filename}.{format}"

    # Graphvizのグラフをレンダリングして保存
    graphviz_graph.render(
        filename=str(image_path.with_suffix("")),  # 拡張子なしで渡す
        format=format,
    )

    # ログに追加
    log_path = result_dir / "log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        if title:
            f.write(f"## {title}\n\n")
        relative_path = image_path.relative_to(result_dir)
        f.write(f"![{filename}]({relative_path})\n\n")

    return image_path
