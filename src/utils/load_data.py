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
    cycles: int = 10,  # 末尾から何周期分切り出すか
    start_time_offset: float = 0.0000005,  # ズレ補正値[s]
    time_label: str = "TIME",
    iL_label: str = "CH1",
    vC_label: str = "CH2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, skiprows=skiprows)

    t_raw: np.ndarray = df[time_label].to_numpy(dtype=np.float32)  # 秒
    iL_raw: np.ndarray = df[iL_label].to_numpy(dtype=np.float32)  # A
    vC_raw: np.ndarray = df[vC_label].to_numpy(dtype=np.float32)  # V

    if cycles <= 0:
        raise ValueError("cycles must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if downsample_step <= 0:
        raise ValueError("downsample_step must be positive.")

    # 末尾から「cycles*T 秒」分を切り出す
    if t_raw.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    t_end = float(t_raw[-1])
    start_time = t_end - float(cycles + 1) * float(T) + start_time_offset
    end_time = start_time + float(cycles) * float(T)
    mask = (t_raw >= start_time) & (t_raw <= end_time)

    # window が長すぎる等で空になった場合は全体を返す
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
