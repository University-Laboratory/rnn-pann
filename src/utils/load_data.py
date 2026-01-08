from types import SimpleNamespace

import numpy as np
import pandas as pd

path_list = SimpleNamespace(
    path0="../../data/csv/tek0000ALL.csv",
    path1="../../data/csv/tek0001ALL.csv",
    path2="../../data/csv/tek0002ALL.csv",
    path3="../../data/csv/tek0003ALL.csv",
)


def load_data(
    path: str = path_list.path0,
    skiprows: int = 20,  # ヘッダーの行数
    downsample_step: int = 125,
    T: float = 1e-6,  # 1周期の実時間[s]
    cycles: int | None = 10,  # 末尾から何周期分切り出すか。Noneの場合は全体
    start_time_offset: float = 0.0000005,  # ズレ補正値[s]
    time_label: str = "TIME",
    iL_label: str = "CH1",
    vC_label: str = "CH2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, skiprows=skiprows)

    t_raw: np.ndarray = df[time_label].to_numpy(dtype=np.float64)  # 秒
    iL_raw: np.ndarray = df[iL_label].to_numpy(dtype=np.float64)  # A
    vC_raw: np.ndarray = df[vC_label].to_numpy(dtype=np.float64)  # V

    if cycles is not None:
        if cycles <= 0:
            raise ValueError("cycles must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if downsample_step <= 0:
        raise ValueError("downsample_step must be positive.")

    # データが空の場合は空配列を返す
    if t_raw.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    if cycles is None:
        # cyclesがNoneのとき、start_time_offset分だけ後ろから切り出す
        mask = t_raw >= (t_raw[0] + start_time_offset)
        if not np.any(mask):
            # マスクが空の場合は全データ返す
            t_all = t_raw
            iL_all = iL_raw
            vC_all = vC_raw
        else:
            t_all = t_raw[mask]
            iL_all = iL_raw[mask]
            vC_all = vC_raw[mask]
    else:
        # 末尾から「cycles*T 秒」分を切り出す
        t_end = float(t_raw[-1])
        start_time = t_end - float(cycles + 1) * float(T) + start_time_offset
        end_time = start_time + float(cycles) * float(T)
        mask = (t_raw >= start_time) & (t_raw <= end_time)

        # windowが長すぎる等で空になった場合は全体を返す
        if not np.any(mask):
            t_all = t_raw
            iL_all = iL_raw
            vC_all = vC_raw
        else:
            t_all = t_raw[mask]
            iL_all = iL_raw[mask]
            vC_all = vC_raw[mask]

    # 先頭を0秒にそろえる
    if t_all.size:
        t_all = t_all - t_all[0]

    t_downsampled: np.ndarray = t_all[::downsample_step]
    iL_downsampled: np.ndarray = iL_all[::downsample_step]
    vC_downsampled: np.ndarray = vC_all[::downsample_step]

    return t_downsampled, iL_downsampled, vC_downsampled
