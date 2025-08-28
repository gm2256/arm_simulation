import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import pandas as pd
#모델 만들기 
class PoseToJointDataset(Dataset):
    def __init__(self, csv_path, seq_len=6):
        df = pd.read_csv(csv_path)
        data = df.values

        self.seq_len = seq_len
        self.inputs = []
        self.outputs = []

        for i in range(len(data) - seq_len + 1):
            x_seq = data[i:i+seq_len, 0:6]   # X, Y, Z, Roll, Pitch, Yaw
            y_seq = data[i:i+seq_len, 6:12]  # theta1~theta6

            self.inputs.append(torch.tensor(x_seq, dtype=torch.float32))   # [6, 6]
            self.outputs.append(torch.tensor(y_seq, dtype=torch.float32))  # [6, 6]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class PoseToJointModel(nn.Module):
    def __init__(self, input_dim=6, model_dim=128, output_dim=6, num_heads=4, ff_dim=128, num_layers=4):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, model_dim, num_heads, ff_dim, num_layers)
        self.predictor = nn.Linear(model_dim, output_dim)

    def forward(self, x):  # x: [batch, seq_len, input_dim]
        encoded = self.encoder(x)        # [batch, seq_len, model_dim]
        output = self.predictor(encoded) # [batch, seq_len, output_dim]
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = self.create_positional_encoding(500, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def create_positional_encoding(self, max_len, model_dim):
        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(model_dim).unsqueeze(0)
        angle = pos / (10000 ** (2 * (i // 2) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        return pe.unsqueeze(0)

    def forward(self, x):  # x: [batch, seq_len, input_dim]
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :].to(x.device)
        x = self.transformer_encoder(x.transpose(0, 1))  # → [seq_len, batch, model_dim]
        return x.transpose(0, 1)  # → [batch, seq_len, model_dim]

#학습실행
if __name__ == "__main__":
    dataset = PoseToJointDataset("robot_dataset.csv", seq_len=6)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PoseToJointModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            pred = model(x_batch)  # [batch, 6, 6]
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    torch.save(model.state_dict(), "pose_to_joint_model5.pth")
    print("저장완료")