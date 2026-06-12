# 屏蔽MediaPipe冗余GL/EGL日志（适配WSL，消除刷屏）
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader

# MediaPipe 人体姿态
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 可视化 & 评估
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import math
import json

# ==========================================
# 全局超参数配置
# ==========================================
CONFIG = {
    'input_dim': 132,
    'target_frames': 30,
    'd_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'num_classes': 6,
    'dropout': 0.2,
    'batch_size': 16,
    'epochs': 40,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'test_size': 0.2,
    'random_seed': 42,
    'data_dir': './archive',
    'output_dir': './output_images',
    'npy_save_dir': './dataset_npy',
    'class_names': [
        'forehand_drive',
        'forehand_lift',
        'forehand_net_shot',
        'forehand_clear',
        'backhand_drive',
        'backhand_net_shot'
    ],
    'detect_conf': 0.5
}

# 创建目录
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['npy_save_dir'], exist_ok=True)

# 固定随机种子（实验可复现）
np.random.seed(CONFIG['random_seed'])
torch.manual_seed(CONFIG['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG['random_seed'])

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用设备: {device}")

# 全局初始化Pose（仅创建一次，提升效率）
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=CONFIG['detect_conf'],
    model_complexity=1
)

# 保存标签映射
label_map = {idx: name for idx, name in enumerate(CONFIG['class_names'])}
with open(os.path.join(CONFIG['npy_save_dir'], "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

# ==========================================
# 1. 骨架归一化（实验要求：髋中心 + 肩宽归一）
# ==========================================
def normalize_skeleton(skel_data):
    T, feat_dim = skel_data.shape
    data = skel_data.copy()
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    for t in range(T):
        lh_x = data[t, LEFT_HIP * 4]
        lh_y = data[t, LEFT_HIP * 4 + 1]
        rh_x = data[t, RIGHT_HIP * 4]
        rh_y = data[t, RIGHT_HIP * 4 + 1]
        hip_cx = (lh_x + rh_x) / 2.0
        hip_cy = (lh_y + rh_y) / 2.0

        ls_x = data[t, LEFT_SHOULDER * 4]
        rs_x = data[t, RIGHT_SHOULDER * 4]
        shoulder_w = abs(ls_x - rs_x)
        scale = shoulder_w if shoulder_w > 1e-6 else 1.0

        for k in range(33):
            x_idx = k * 4
            y_idx = k * 4 + 1
            data[t, x_idx] = (data[t, x_idx] - hip_cx) / scale
            data[t, y_idx] = (data[t, y_idx] - hip_cy) / scale
    return data

# ==========================================
# 2. 视频提取骨架序列
# ==========================================
def extract_skeleton_from_video(video_path, target_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            frame_feat = []
            for lm in results.pose_landmarks.landmark:
                frame_feat.extend([lm.x, lm.y, lm.z, lm.visibility])
            frames_data.append(frame_feat)
    cap.release()

    if len(frames_data) == 0:
        return None
    frames_data = np.array(frames_data, dtype=np.float32)
    T_orig = frames_data.shape
    indices = np.linspace(0, T_orig[0] - 1, target_frames)
    resampled = np.zeros((target_frames, CONFIG['input_dim']), dtype=np.float32)
    for dim in range(CONFIG['input_dim']):
        resampled[:, dim] = np.interp(indices, np.arange(T_orig[0]), frames_data[:, dim])
    resampled = normalize_skeleton(resampled)
    return resampled

# ==========================================
# 3. 数据集加载（优先读取npy，加速重复运行）
# ==========================================
def prepare_dataset(data_dir):
    npy_root = CONFIG['npy_save_dir']
    try:
        X_train = np.load(os.path.join(npy_root, "X_train.npy"))
        X_test = np.load(os.path.join(npy_root, "X_test.npy"))
        y_train = np.load(os.path.join(npy_root, "y_train.npy"))
        y_test = np.load(os.path.join(npy_root, "y_test.npy"))
        print("[INFO] 加载已有NPY数据集，跳过视频解析")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("[INFO] 未检测NPY文件，开始解析视频...")

    X, y = [], []
    class_to_idx = {name: idx for idx, name in enumerate(CONFIG['class_names'])}
    if not os.path.exists(data_dir):
        print(f"[ERROR] 数据集目录不存在: {data_dir}")
        return None, None, None, None

    for cls_name in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_path) or cls_name not in class_to_idx:
            continue
        label = class_to_idx[cls_name]
        print(f"  处理类别: {cls_name} (标签={label})")
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                vid_path = os.path.join(cls_path, fname)
                skel = extract_skeleton_from_video(vid_path, CONFIG['target_frames'])
                if skel is not None:
                    X.append(skel)
                    y.append(label)

    if len(X) == 0:
        print("[ERROR] 未提取有效样本！")
        return None, None, None, None

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"[INFO] 有效样本总数: {X.shape[0]}, 样本形状: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'],
        random_state=CONFIG['random_seed'], stratify=y
    )
    np.save(os.path.join(npy_root, "X_train.npy"), X_train)
    np.save(os.path.join(npy_root, "X_test.npy"), X_test)
    np.save(os.path.join(npy_root, "y_train.npy"), y_train)
    np.save(os.path.join(npy_root, "y_test.npy"), y_test)
    print(f"[INFO] 数据集已保存至 {npy_root}")
    return X_train, X_test, y_train, y_test

# ==========================================
# 4. 位置编码
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ==========================================
# 5. 自定义 Transformer Encoder 层（支持返回注意力权重）
# ==========================================
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力层，返回注意力权重
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 前馈层
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

# ==========================================
# 6. 自定义 Transformer Encoder（兼容自定义层）
# ==========================================
class CustomTransformerEncoder(nn.Module):
    def __init__(self, layers, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attn_weights_list = []
        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weights_list.append(attn_weights)
        return output, attn_weights_list

# ==========================================
# 7. 模型（兼容自定义层，支持注意力权重提取）
# ==========================================
class SkeletonTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Linear(cfg['input_dim'], cfg['d_model'])
        self.pos_enc = PositionalEncoding(cfg['d_model'], cfg['dropout'], cfg['target_frames'])
        self.d_model = cfg['d_model']
        self.nhead = cfg['nhead']

        # 使用自定义层替代原生层
        enc_layer = CustomTransformerEncoderLayer(
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            dim_feedforward=cfg['dim_feedforward'],
            dropout=cfg['dropout'],
            activation="relu"
        )
        self.encoder = CustomTransformerEncoder([enc_layer]*cfg['num_layers'], cfg['num_layers'])

        self.classifier = nn.Sequential(
            nn.Linear(cfg['d_model'], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg['dropout']),
            nn.Linear(64, cfg['num_classes'])
        )
        self.attn_weights = None  # 存储注意力权重

    def forward(self, x):
        # x: [B, T, D]
        x = self.embed(x)
        x = self.pos_enc(x)
        x, attn_weights_list = self.encoder(x)
        self.attn_weights = attn_weights_list[-1].detach().cpu().numpy()  # 取最后一层注意力权重
        x = torch.mean(x, dim=1)
        logits = self.classifier(x)
        return logits

# ==========================================
# 8. 训练 & 评估主流程
# ==========================================
def train_and_evaluate():
    X_train, X_test, y_train, y_test = prepare_dataset(CONFIG['data_dir'])
    if X_train is None:
        return None

    train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    model = SkeletonTransformer(CONFIG).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    train_loss_list = []
    test_acc_list = []
    print("\n===== 开始模型训练 =====")
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0.0
        for batch_in, batch_label in train_loader:
            batch_in = batch_in.to(device)
            batch_label = batch_label.to(device)
            optimizer.zero_grad()
            out = model(batch_in)
            loss = criterion(out, batch_label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_in.size(0)
        epoch_loss = total_loss / len(train_set)
        train_loss_list.append(epoch_loss)

        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch_in, batch_label in test_loader:
                batch_in = batch_in.to(device)
                batch_label = batch_label.to(device)
                out = model(batch_in)
                _, pred = torch.max(out, dim=1)
                all_pred.extend(pred.cpu().numpy())
                all_true.extend(batch_label.cpu().numpy())
        epoch_acc = accuracy_score(all_true, all_pred) * 100
        test_acc_list.append(epoch_acc)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:2d}/{CONFIG['epochs']}] | Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.2f}%")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(train_loss_list, color='royalblue', linewidth=2)
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.subplot(1,2,2)
    plt.plot(test_acc_list, color='forestgreen', linewidth=2)
    plt.title("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], "training_curves.png"), dpi=150)
    plt.close()

    # 分类报告 & 混淆矩阵
    print("\n===== 分类评估报告 =====")
    print(classification_report(all_true, all_pred, target_names=CONFIG['class_names'], digits=4))
    cm = confusion_matrix(all_true, all_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(CONFIG['num_classes']))
    plt.figure(figsize=(8,8))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], "confusion_matrix.png"), dpi=150)
    plt.close()

    torch.save(model.state_dict(), os.path.join(CONFIG['npy_save_dir'], "badminton_transformer.pth"))
    print(f"[INFO] 模型已保存")
    return model

# ==========================================
# 9. 单视频推理
# ==========================================
def inference_single_video(model, video_path):
    print("\n===== 单视频推理 =====")
    if not os.path.exists(video_path):
        print(f"[WARN] 视频不存在: {video_path}")
        return None
    feat = extract_skeleton_from_video(video_path, CONFIG['target_frames'])
    if feat is None:
        print("[ERROR] 骨架提取失败")
        return None
    input_tensor = torch.from_numpy(feat).float().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_cls = CONFIG['class_names'][pred_idx]
    conf = float(probs[pred_idx])
    print(f"Predicted class: {pred_cls}")
    print(f"Confidence: {conf:.2f}")
    return input_tensor

# ==========================================
# 10. 拓展1：骨架可视化（输出视频，报告可用）
# ==========================================
def draw_skeleton_video(src_video, dst_name="skeleton_vis.mp4"):
    cap = cv2.VideoCapture(src_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(CONFIG['output_dir'], dst_name), fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
            )
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"[INFO] 骨架可视化视频已保存至 {CONFIG['output_dir']}/{dst_name}")

# ==========================================
# 11. 拓展2：Attention注意力热力图（报告可用）
# ==========================================
def plot_attention(model, input_tensor):
    """绘制Transformer时序注意力热力图"""
    if model.attn_weights is None:
        print("[WARN] 未获取到注意力权重")
        return
    attn = model.attn_weights[0]
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap="hot", aspect="auto")
    plt.colorbar()
    plt.title("Transformer Attention Weights")
    plt.xlabel("Frame Index")
    plt.ylabel("Frame Index")
    plt.tight_layout()
    save_path = os.path.join(CONFIG['output_dir'], "attention_heatmap.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] 注意力热力图已保存: {save_path}")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 训练模型
    trained_model = train_and_evaluate()

    if trained_model is not None:
        # ========== 1. 选择2个视频做可视化（修改这里为你的视频路径） ==========
        # 请将下面的路径替换为你数据集中真实存在的视频文件
        vid1 = os.path.join(CONFIG['data_dir'], "forehand_clear", "002.mp4")
        vid2 = os.path.join(CONFIG['data_dir'], "backhand_net_shot", "003.mp4")

        # 2. 骨架可视化（2个视频，满足要求）
        if os.path.exists(vid1):
            draw_skeleton_video(vid1, "skeleton_video_1.mp4")
        if os.path.exists(vid2):
            draw_skeleton_video(vid2, "skeleton_video_2.mp4")

        # 3. 推理 + 注意力可视化
        if os.path.exists(vid1):
            input_tensor = inference_single_video(trained_model, vid1)
            if input_tensor is not None:
                plot_attention(trained_model, input_tensor)

    print("\n[INFO] 所有流程执行完毕！")