import torch
from torch import nn


class BuckConverterCell(nn.Module):
    def __init__(self, L_init: float, C_init: float, R_init: float) -> None:
        super().__init__()
        # パラメータを対数空間で学習（正の値を保証）
        self.log_L = nn.Parameter(torch.log(torch.tensor(L_init)))
        self.log_C = nn.Parameter(torch.log(torch.tensor(C_init)))
        self.log_R = nn.Parameter(torch.log(torch.tensor(R_init)))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        h: [iL, vC]
        x: [vs, u, dt]
        """

        iL = h[:, 0]
        vC = h[:, 1]
        vs = x[:, 0]
        u = x[:, 1]
        dt = x[:, 2]
        vp = vs * u

        # パラメータを指数関数で変換（正の値を保証）
        L = torch.exp(self.log_L)
        C = torch.exp(self.log_C)
        R = torch.exp(self.log_R)

        # オイラー法
        iL_next = iL + (dt / L) * (vp - vC)
        vC_next = vC + (dt / C) * (iL - vC / R)

        return torch.stack([iL_next, vC_next], dim=1)


class BuckConverterCellILOnly(nn.Module):
    def __init__(self, L_init: float) -> None:
        super().__init__()
        # パラメータを対数空間で学習（正の値を保証）
        self.log_L = nn.Parameter(torch.log(torch.tensor(L_init)))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        h: [iL, vC]
        x: [vs, u, dt]
        """

        iL = h[:, 0]
        vC = h[:, 1]
        vs = x[:, 0]
        u = x[:, 1]
        dt = x[:, 2]
        vp = vs * u

        # パラメータを指数関数で変換（正の値を保証）
        L = torch.exp(self.log_L)

        # オイラー法
        iL_next = iL + (dt / L) * (vp - vC)

        return torch.stack([iL_next], dim=1)


# GRUモデルの定義（ノイズ予測用）
class GRUModel(nn.Module):
    def __init__(
        self,
        input_size: int = 7,  # [iL, vC, vs, u, dt, iL_noise, vC_noise]
        hidden_size: int = 32,
        num_layers: int = 2,
        output_size: int = 2,  # [iL_noise, vC_noise]
        seq_length: int = 10,  # 時系列の長さ
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_length, input_size]
        出力: [batch_size, output_size]
        """
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # 最後の時刻の出力を使用
        return out
