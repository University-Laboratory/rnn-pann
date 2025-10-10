import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# sinカーブのデータを作成
x = np.linspace(0, 100, 1000)
y = np.sin(x)

df = pd.DataFrame({"x": x, "y": y})

# CSVとして保存
csv_path = "data/sin_curve.csv"
df.to_csv(csv_path, index=False)


data = df["y"].values.astype("float32")


# --- Dataset定義 ---
class SinDataset(Dataset):
    def __init__(
        self, data: torch.Tensor, look_back: int = 3, horizon: int = 7
    ) -> None:
        """
        data: 時系列データ
        look_back: 入力に使うステップ数
        horizon: 何ステップ先を予測するか (例: 7なら T+7 をラベルにする)
        初期値のままなら、X[1,2,3] => y[10] という予測になる
        """
        self.data = data
        self.look_back = look_back
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.data) - self.look_back - self.horizon

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.look_back]
        y = self.data[idx + self.look_back + self.horizon - 1]
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y, dtype=torch.float32),
        )


# --- モデル定義 ---
class RNNModel(nn.Module):
    def __init__(
        self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 最後のステップの出力のみ使用
        return out


# --- パラメータ ---
look_back = 3  # 何ステップ過去を見るか
horizon = 7  # 何ステップ先を予測するか
batch_size = 32
epochs = 20

# --- Dataset / DataLoader ---
dataset = SinDataset(data, look_back=look_back, horizon=horizon)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- モデル・学習 ---
model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    for X_batch, y_batch in dataloader:
        pred = model(X_batch)
        loss = criterion(pred, y_batch.unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")


model.eval()
preds = np.full_like(data, fill_value=np.nan, dtype=np.float32)  # 予測値を格納する配列
targets = np.full_like(data, fill_value=np.nan, dtype=np.float32)

with torch.no_grad():
    for i in range(len(dataset)):
        X, y = dataset[i]
        X = X.unsqueeze(0)  # (1, look_back, 1)
        pred = model(X).item()

        target_idx = i + look_back + horizon - 1
        preds[target_idx] = pred
        targets[target_idx] = y.item()

# --- 1周期 (0〜2π) の範囲だけを抽出 ---
x_values = df["x"].values
mask = x_values <= 2 * np.pi

# --- 推論結果をCSVファイルに保存 ---
results_df = pd.DataFrame({"x": x_values, "true_y": targets, "predicted_y": preds})

# NaNを除外（学習に使用できなかった部分を除外）
results_df = results_df.dropna()

# CSVファイルとして保存
output_csv_path = "data/prediction_results.csv"
results_df.to_csv(output_csv_path, index=False)
print(f"推論結果を {output_csv_path} に保存しました")

# --- 可視化 ---
plt.figure(figsize=(10, 5))
plt.plot(x_values[mask], targets[mask], label="True (sin)")
plt.scatter(
    x_values[mask], preds[mask], label="Predicted", color="orange", s=10
)  # ドットのみ表示
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.title(f"RNN Prediction vs True (look_back={look_back}, horizon={horizon})")
plt.savefig("data/prediction_results.png")
# plt.show()
