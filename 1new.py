
import pandas as pd
import torch
import torch.nn as nn
import joblib

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim

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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

#导入模型
checkpoint = torch.load(
    "titanic_checkpoint.pth",
    map_location=device
)

model = MLP(
    input_dim=checkpoint["input_dim"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

model.eval()


def predict_survival(
    Pclass,
    Sex,
    Age,
    SibSp,
    Parch,
    Fare,
    Embarked
):

    passenger = pd.DataFrame([{
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": Embarked
    }])

    passenger["Sex"] = passenger["Sex"].map({
        "male": 0,
        "female": 1
    })

    if passenger["Sex"].isna().any():
        raise ValueError("Sex 只能输入 male 或 female")

    if Embarked not in ["C", "Q", "S"]:
        raise ValueError("Embarked 只能输入 C、Q 或 S")

    passenger["FamilySize"] = (
        passenger["SibSp"]
        + passenger["Parch"]
        + 1
    )
    passenger["IsAlone"] = (
            passenger["FamilySize"] == 1
    ).astype(int)

    passenger["Embarked_Q"] = (passenger["Embarked"] == "Q").astype(int)
    passenger["Embarked_S"] = (passenger["Embarked"] == "S").astype(int)

    passenger["Pclass_2"] = (passenger["Pclass"] == 2).astype(int)
    passenger["Pclass_3"] = (passenger["Pclass"] == 3).astype(int)

    passenger = passenger.drop(
        columns=["Embarked", "Pclass"]
    )


    passenger = passenger.reindex(
        columns=feature_names,
        fill_value=0
    )


    num_cols = [ "Age", "FamilySize", "Fare"]

    passenger[num_cols] = scaler.transform(
        passenger[num_cols]
    )

    X_new = torch.tensor(
        passenger.values,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        logits = model(X_new).squeeze(1)

        probability = torch.sigmoid(logits).item()

        prediction = 1 if probability >= 0.5 else 0

    return prediction, probability



if __name__ == "__main__":

    print("=== Titanic 新乘客生存预测 ===")

    Pclass = int(input("请输入船票等级 Pclass（1、2、3）："))
    Sex = input("请输入性别 Sex（male/female）：").strip().lower()
    Age = float(input("请输入年龄 Age："))
    SibSp = int(input("请输入同行兄弟姐妹或配偶人数 SibSp："))
    Parch = int(input("请输入同行父母或子女人数 Parch："))
    Fare = float(input("请输入船票价格 Fare："))
    Embarked = input("请输入登船港口 Embarked（C/Q/S）：").strip().upper()

    prediction, probability = predict_survival(
        Pclass=Pclass,
        Sex=Sex,
        Age=Age,
        SibSp=SibSp,
        Parch=Parch,
        Fare=Fare,
        Embarked=Embarked
    )

    print("\n=== 预测结果 ===")
    print(f"幸存概率：{probability:.2%}")
    #print(f"幸存概率：{probability:.2%}")
    print(f"预测标签：{prediction}")

    if prediction == 1:
        print("模型预测：该乘客可能幸存。")
    else:
        print("模型预测：该乘客可能未幸存。")