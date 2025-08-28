import torch
import torch.nn as nn
import project2_makemodel as Model

model = Model.PoseToJointModel()
model.load_state_dict(torch.load("pose_to_joint_model.pth"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ 너가 직접 넣을 pose 시퀀스 6줄
custom_pose = [
    [0.550, -0.250, 0.1, 90.0, 0.0, 0.0],
    [0.5, -0.2, 0.15, 90.0, 1.0, 0.0],
    [0.5, -0.2, 0.2, 90.0, 2.0, 0.0],
    [0.5, -0.2, 0.25, 90.0, 3.0, 0.0],
    [0.5, -0.2, 0.3, 90.0, 4.0, 0.0],
    [0.5, -0.2, 0.35, 90.0, 5.0, 0.0],
]

# 👉 텐서로 변환하고 모델 입력 형태에 맞게 [1, 6, 6] shape로
x = torch.tensor(custom_pose, dtype=torch.float32).unsqueeze(0).to(device)

# 🧠 예측 실행
with torch.no_grad():
    y_pred = model(x)* (180/torch.pi)  # [1, 6, 6]

# 출력
print("예측된 관절 각도 시퀀스:")
print(y_pred.squeeze(0).cpu())  # [6, 6]



