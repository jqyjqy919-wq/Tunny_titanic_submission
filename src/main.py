import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

#导入数据
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
df = pd.read_csv("train.csv")

seed = 42

np.random.seed(seed)
torch.manual_seed(seed)

#数据预处理和特征工程
known_ages = df["Age"].dropna()
missing_age = df[df["Age"].isna()].index
df.loc[missing_age, "Age"] = np.random.choice(known_ages, size=len(missing_age))

df["Embarked"]=df['Embarked'].fillna(df['Embarked'].mode()[0])

df["Sex"] = df["Sex"].map({"male": 0,"female": 1})

df=df.drop(['PassengerID','Cabin','Ticket','Name'], axis=1)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

df=pd.get_dummies(df,columns=["Embarked"],drop_first=True,dtype=int)
df=pd.get_dummies(df,columns=["Pclass"],drop_first=True,dtype=int)



train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Survived"]
)

#标准化
num_cols =['Age','FamilySize','Fare']
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])


feature_names = train.drop("Survived", axis=1).columns.tolist()

# 定义 X 和 y
X_train = train[feature_names].values
y_train = train["Survived"].values

X_test = test[feature_names].values
y_test = test["Survived"].values

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)



class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.mlp(x)


class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#设置Dataset，DataLoader

train_dataset = TitanicDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TitanicDataset(
    X_test_tensor,
    y_test_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

input_dim = X_train.shape[1]

model = MLP(
    input_dim=input_dim
).to(device)

#损失函数
criterion = nn.BCEWithLogitsLoss()
#优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0003
)

#训练
epochs = 1000

train_losses = []
train_accs = []
test_accs = []


for epoch in range(epochs):

    model.train()

    total_loss = 0.0
    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()#每次清零梯度

        logits = model(X_batch).squeeze(1)

        loss = criterion(logits, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        correct1 += (preds == y_batch).sum().item()
        total1 += y_batch.size(0)
    train_loss = total_loss / len(train_loader)
    train_acc = correct1 / total1

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch).squeeze(1)
            probs = torch.sigmoid(logits)

            preds = (probs >= 0.5).float()

            correct2 += (preds == y_batch).sum().item()
            total2 += y_batch.size(0)


    test_acc = correct2 / total2

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}"
            f"Test Acc: {test_acc:.4f}"
        )

model.eval()

epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(
    epochs_range,
    train_accs,
    label="Train Accuracy"
)
plt.plot(
    epochs_range,
    test_accs,
    label="Test Accuracy"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train and Test Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    epochs_range,
    train_losses,
    label="Train Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.show()

#保存模型
torch.save({
    "input_dim": input_dim,
    "model_state_dict": model.state_dict()
}, "titanic_checkpoint.pth")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")
