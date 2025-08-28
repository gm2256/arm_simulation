import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,TextBox, Button
from scipy.spatial.transform import Rotation as R
import pandas as pd
import torch
import torch.nn as nn
import project2_makemodel as Model


features = []
button_clicked = False
model_joint = [] 
# ---- DH 행렬 계산 ----
def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

# ---- 정기구학 ----
def forward_kinematics(joint_angles):
    dh_params = [
        [joint_angles[0],  0.100,   0,   np.deg2rad(90)],
        [joint_angles[1],    0, 0.300,   0],
        [joint_angles[2],    0, 0.250,   0],
        [joint_angles[3],  0.200,   0,   np.deg2rad(90)],
        [joint_angles[4],    0,   0,   np.deg2rad(-90)],
        [joint_angles[5],   0.050,   0,   0]
    ]
    
    T = np.identity(4)
    positions = [T[:3, 3]]  # 시작점
    for theta, d, a, alpha in dh_params:
        T = T @ dh_matrix(theta, d, a, alpha)
        positions.append(T[:3, 3])
    return np.array(positions), T

# ---- 버튼클릭함수
def apply_pose(event):
    global button_clicked, model_joint
    # 입력값 읽기
    vals = [float(text_boxes[l].text) for l in labels]
    x,y,z,roll,pitch,yaw = vals
    custom_pose = [
        [x,y,z,roll,pitch,yaw]]

    # 👉 텐서로 변환하고 모델 입력 형태에 맞게 [1, 6, 6] shape로
    x = torch.tensor(custom_pose, dtype=torch.float32).unsqueeze(0).to(device)

    # 🧠 예측 실행
    with torch.no_grad():
        y_pred = model(x)* (180/torch.pi)  # [1, 6, 6]
    
    # 전역 변수에 저장
    button_clicked = True
    model_joint = y_pred.squeeze(0)[-1].cpu().numpy().tolist()  # [6]
    print("예측된 관절 각도:", model_joint)

    for i, s in enumerate(sliders[:len(model_joint)]):
        s.set_val(int(model_joint[i]))   # 예측값을 슬라이더에 반영

# ---- 초기 값 ----
init_angles = [np.deg2rad(0)] * 6
# ---- 모델 불러오기 ----
model = Model.PoseToJointModel()
model.load_state_dict(torch.load("pose_to_joint_model5.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# ---- Figure 설정 ----
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.35)

line, = ax.plot([], [], [], 'o-', lw=4, color='blue')
text_pos = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

# ---- 슬라이더 생성 ----
sliders = []
slider_axes = []
slider_labels = ["theta1", "theta2", "theta3", "theta4", "theta5", "theta6"]

# ---- 입력칸 (X, Y, Z, R, P, Y) ----
text_boxes = {}
labels = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
default_vals = ["0","0","0.5","0","0","0"]

# ---- 버튼 ----
ax_button = plt.axes([0.2, 0.018, 0.1, 0.03])
btn = Button(ax_button, "Set Pose")
btn.on_clicked(apply_pose) 

#---- 입력칸 위치,길이
for i, label in enumerate(labels):
    ax_box = plt.axes([0.2, 0.25 - i*0.04, 0.2, 0.03])  
    text_box = TextBox(ax_box, label+" ", initial=default_vals[i])
    text_boxes[label] = text_box

# ---- theta 위치,길이
for i in range(6):
    ax_slider = plt.axes([0.55, 0.25 - i*0.035, 0.3, 0.03])
    slider = Slider(ax_slider, slider_labels[i], -180, 180, valinit=0)
    sliders.append(slider)
    
# ---- 업데이트 함수 ----
def update(val):
    global button_clicked, model_joint
    if button_clicked == True:
        angles_deg = model_joint
        #reset
        button_clicked = False
    else:    
        angles_deg = [s.val for s in sliders]
    
    angles_rad = [np.deg2rad(a) for a in angles_deg]
    positions, T = forward_kinematics(angles_rad)

    # 3D 로봇 라인 업데이트
    line.set_data(positions[:, 0], positions[:, 1])
    line.set_3d_properties(positions[:, 2])

    # 축 범위
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1.4)

    # 위치 + 방향 출력
    tip = T[:3, 3]
    rotation_matrix = T[:3, :3]
    rpy = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)  # radians → degrees
    roll, pitch, yaw = rpy      
    
    #데이터 저장
    save_data(tip[0],tip[1],tip[2],roll,pitch,yaw,angles_rad[0],angles_rad[1],angles_rad[2],angles_rad[3],angles_rad[4],angles_rad[5])
    text_pos.set_text(
        f"End Effector:\n"
        f"X={tip[0]:.3f}, Y={tip[1]:.3f}, Z={tip[2]:.3f}\n"
        f"Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°"
    )

    fig.canvas.draw_idle()

# ---- 슬라이더 연결 ----
for s in sliders:
    s.on_changed(update)

# ---- 데이터저장 ----
data_log = []

def save_data(x, y, z, Roll, Pitch, Yaw, theta1,theta2,theta3,theta4,theta5,theta6):
    row = [x, y, z, Roll, Pitch, Yaw, theta1,theta2,theta3,theta4,theta5,theta6]
    data_log.append(row)

    if len(data_log) >= 4000:
        df = pd.DataFrame(data_log, columns=[
            "X", "Y", "Z", "Roll", "Pitch", "Yaw",
            "theta1", "theta2", "theta3", "theta4", "theta5", "theta6"
        ])
        df.to_csv("robot_dataset.csv", index=False)
        print("✅ 저장 완료")
        data_log.clear()


# ---- 초기 호출 ----
update(None)
plt.show()


