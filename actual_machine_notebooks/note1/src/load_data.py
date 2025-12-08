from types import SimpleNamespace

import numpy as np
import pandas as pd

path_list = SimpleNamespace(
    path0="../../data/csv/tek0000ALL.csv",
    path1="../../data/csv/tek0001ALL.csv",
)


def _load_data(
    path: str = path_list.path0,
    skiprows: int = 20,
    downsample_step: int = 125,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, skiprows=skiprows)

    t_raw: np.ndarray = df["TIME"].to_numpy(dtype=np.float32)  # 秒
    iL_raw: np.ndarray = df["CH1"].to_numpy(dtype=np.float32)  # A
    vC_raw: np.ndarray = df["CH2"].to_numpy(dtype=np.float32)  # V

    # Time 軸が -0.00005 から始まっているので、0.00005秒加算し、0以上部分のみ抽出
    t_all: np.ndarray = t_raw + 0.00005
    mask: np.ndarray = t_all >= 0
    t_all = t_all[mask]
    iL_all = iL_raw[mask]
    vC_all = vC_raw[mask]

    t_downsampled: np.ndarray = t_all[::downsample_step]
    iL_downsampled: np.ndarray = iL_all[::downsample_step]
    vC_downsampled: np.ndarray = vC_all[::downsample_step]

    return t_downsampled, iL_downsampled, vC_downsampled


def load_data_path0() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _load_data(path=path_list.path0)


def load_data_path1() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _load_data(path=path_list.path1)
