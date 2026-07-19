# ================================
# LSTM(Long-Short Time Memory) 적용
# =================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ── 1. LSTM에 들어갈 Feature 선택 ──
# (과거 요약본인 PREV_ 컬럼들은 LSTM이 스스로 기억하므로 제외합니다)
lstm_features = [
    'RELEASE_SPEED', 'AX', 'AZ', 
    'X_AT_DECISION', 'Z_AT_DECISION',
    'MOVE_RATE', 'DROP_RATE', 
    'STRIKES', 'BALLS', 'STAND',
    'BATTER_ZONE_SWING', 'BATTER_CHASE_RATE'
]

# 결측치 방어 코드 (혹시 남아있을 수 있는 NaN 0으로 처리)
rhp[lstm_features] = rhp[lstm_features].fillna(0)
rhp['STAND'] = (rhp['STAND'] == 'R').astype(int)
# ── 2. 타석 단위(At-bat)로 그룹화하여 시퀀스 생성 ──
sequences = []
labels = []
seq_lengths = []

# GAME_PK와 AT_BAT_NUMBER 기준으로 그룹화 (이미 시간순 정렬되어 있다고 가정)
grouped = rhp.groupby(['GAME_PK', 'AT_BAT_NUMBER'])

for _, group in grouped:
    # 해당 타석의 투구 feature 시퀀스 추출
    seq = group[lstm_features].values
    # 해당 타석의 스윙 여부 레이블 추출
    label = group['DID_SWING'].values
    
    sequences.append(torch.tensor(seq, dtype=torch.float32))
    labels.append(torch.tensor(label, dtype=torch.float32))
    seq_lengths.append(len(seq))

# ── 3. 패딩(Padding) 적용 ──
# 타석마다 투구 수가 다르므로(어떤 타석은 1구, 어떤 타석은 8구), 
# 가장 긴 타석에 맞춰 0으로 채워 길이를 통일합니다.
# batch_first=True 로 설정하여 (Batch, Seq, Feature) 형태 유지
X_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

# 레이블 패딩 시, 실제 타석이 아닌 패딩 부분은 -1 등으로 채워 Loss 계산 시 무시하도록 합니다.
y_padded = pad_sequence(labels, batch_first=True, padding_value=-1.0)

# print(f"변환된 X 텐서 형태: {X_padded.shape}") 
# # 예: (총 타석 수, 최대 투구 수, 피처 수) -> (N_at_bats, max_pitches, len(lstm_features))
# print(f"변환된 y 텐서 형태: {y_padded.shape}")


# =================================
# =================================

from torch.utils.data import TensorDataset, DataLoader

# ── 1. Train / Validation Split (80% / 20%) ──
# (시간순 정렬되어 있다고 가정하고 순서대로 자릅니다. 야구 데이터는 미래 데이터로 과거를 예측하면 안 되기 때문입니다.)
dataset_size = len(X_padded)
train_size = int(dataset_size * 0.7)

X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, train_size=train_size, random_state=42)

# DataLoader 구성 (GPU 메모리를 고려해 Batch Size는 256 또는 512 정도가 적당합니다)
batch_size = 256
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"학습 타석 수: {len(X_train)}, 검증 타석 수: {len(X_test)}")

# ── 2. LSTM 모델 구조 정의 ──
class SwingPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SwingPredictionLSTM, self).__init__()
        
        # LSTM 계층: batch_first=True로 설정해야 (Batch, Seq, Feature) 형태를 받아들입니다.
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # 출력층: LSTM이 요약한 정보를 바탕으로 '해당 투구'의 스윙 확률(로짓)을 계산
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # lstm_out 형태: (Batch_size, Seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # out 형태: (Batch_size, Seq_len, 1)
        out = self.fc(lstm_out)
        
        # 예측값과 정답(y_padded)의 형태를 맞추기 위해 마지막 1차원 제거 
        # 최종 형태: (Batch_size, Seq_len)
        return out.squeeze(-1)

# ── 3. 모델 초기화 및 디바이스(GPU/CPU) 할당 ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 12       # 12개의 피처
hidden_size = 64      # LSTM 내부에서 기억을 유지하는 벡터의 크기 (크면 더 복잡한 패턴 기억 가능)
num_layers = 1        # LSTM 층의 개수

model = SwingPredictionLSTM(input_size, hidden_size, num_layers).to(device)

print("\n[모델 구조]")
print(model)

# ===========================
# ===========================

import torch.optim as optim

# ── 1. Loss 함수 및 Optimizer 세팅 ──
# BCEWithLogitsLoss: 이진 분류(0 or 1)에 특화된 Loss 함수입니다.
# reduction='none'으로 설정해야 타석 길이마다 마스킹을 씌우고 우리가 직접 평균을 낼 수 있습니다.
criterion = nn.BCEWithLogitsLoss(reduction='none')

# 학습률(Learning Rate)은 0.001로 시작
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10  # 우선 10번만 돌려보며 Loss가 잘 떨어지는지 확인합니다.

print("🚀 학습 시작...\n")

for epoch in range(epochs):
    
    # --- [Train Phase] ---
    model.train()
    train_loss_sum = 0.0
    
    for X_batch, y_batch in train_loader:
        # 데이터를 GPU/CPU로 이동
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad() # 기울기 초기화
        
        # 1. 모델 예측 (결과 형태: Batch x 14)
        predictions = model(X_batch)
        
        # 2. 마스킹(Masking) 생성: 정답이 -1이 아닌(즉, 실제 투구인) 위치만 True(1)
        mask = (y_batch != -1.0)
        
        # 3. 전체 Loss 계산 
        loss = criterion(predictions, y_batch)
        
        # 4. 빈 공간(패딩)의 Loss 지우기
        # 마스크를 곱해주면, 실제 투구가 없는 부분의 오차는 0이 됩니다.
        masked_loss = loss * mask
        
        # 유효한(실제 투구) 데이터의 개수만큼만 나누어서 최종 평균 Loss 산출
        final_loss = masked_loss.sum() / mask.sum()
        
        # 역전파 및 가중치 업데이트
        final_loss.backward()
        optimizer.step()
        
        train_loss_sum += final_loss.item()
        
    avg_train_loss = train_loss_sum / len(train_loader)
    
    
    # --- [Validation Phase] ---
    model.eval()
    val_loss_sum = 0.0
    
    with torch.no_grad(): # 평가할 때는 기울기 계산을 하지 않음
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            predictions = model(X_batch)
            mask = (y_batch != -1.0)
            loss = criterion(predictions, y_batch)
            masked_loss = loss * mask
            final_loss = masked_loss.sum() / mask.sum()
            
            val_loss_sum += final_loss.item()
            
    avg_val_loss = val_loss_sum / len(val_loader)
    
    # 결과 출력
    print(f"Epoch [{epoch+1:2d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
print("\n✅ 학습 완료!")

from sklearn.metrics import accuracy_score, roc_auc_score

# 평가 모드 전환
model.eval()

all_predictions = []
all_true_labels = []

print("📊 성능 평가 데이터 추출 중...")

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        
        # 모델 예측 (결과는 Logit 값)
        logits = model(X_batch)
        
        # Sigmoid를 통과시켜 0~1 사이의 확률(Probability) 값으로 변환
        probs = torch.sigmoid(logits)
        
        # CPU로 옮기고 NumPy 배열로 변환
        probs_np = probs.cpu().numpy()
        y_batch_np = y_batch.cpu().numpy()
        
        # 🚨 마스킹: y값이 -1.0이 아닌(즉, 실제 투구인) 위치의 데이터만 골라냅니다.
        # flatten()을 써서 (Batch x Seq) 2차원 배열을 1차원 리스트로 쫙 폅니다.
        mask = (y_batch_np != -1.0)
        
        valid_probs = probs_np[mask]
        valid_labels = y_batch_np[mask]
        
        all_predictions.extend(valid_probs)
        all_true_labels.extend(valid_labels)

# ── 평가 지표 계산 ──
all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)

# 확률이 0.5 이상이면 스윙(1), 아니면 참음(0)으로 간주
pred_classes = (all_predictions >= 0.5).astype(int)

# Accuracy (정확도)
accuracy = accuracy_score(all_true_labels, pred_classes)

# ROC-AUC (1에 가까울수록 모델이 스윙/비스윙을 잘 구분함)
roc_auc = roc_auc_score(all_true_labels, all_predictions)

print("\n🏆 [Validation 성능 결과]")
print(f"Accuracy (정확도): {accuracy:.4f}")
print(f"ROC-AUC 점수   : {roc_auc:.4f}")