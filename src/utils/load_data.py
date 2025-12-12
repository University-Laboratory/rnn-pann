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
    start_time: float = 0.00005,  # シミュレーションの開始時間
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, skiprows=skiprows)

    t_raw: np.ndarray = df["TIME"].to_numpy(dtype=np.float32)  # 秒
    iL_raw: np.ndarray = df["CH1"].to_numpy(dtype=np.float32)  # A
    vC_raw: np.ndarray = df["CH2"].to_numpy(dtype=np.float32)  # V

    t_all: np.ndarray = t_raw + start_time
    mask: np.ndarray = t_all >= 0
    t_all = t_all[mask]
    iL_all = iL_raw[mask]
    vC_all = vC_raw[mask]

    t_downsampled: np.ndarray = t_all[::downsample_step]
    iL_downsampled: np.ndarray = iL_all[::downsample_step]
    vC_downsampled: np.ndarray = vC_all[::downsample_step]

    return t_downsampled, iL_downsampled, vC_downsampled


def load_data_path0() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return load_data(path=path_list.path0)


def load_data_path1() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return load_data(path=path_list.path1)


def load_data_path2() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return load_data(path=path_list.path2)


def load_data_path3() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return load_data(path=path_list.path3)
