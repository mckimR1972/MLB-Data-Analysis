import pandas as pd 
import oracledb as DB
import numpy as np
import seaborn as sns
import traceback
import matplotlib.pyplot as plt
import folium
import missingno as msno

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

non_ab_events = [
    "walk", "intent_walk", "hit_by_pitch",
    "sac_fly", "sac_bunt", "sac_fly_double_play",
    "catcher_interf" 
]
hit_for_run_events = ["single", "double", "triple", "home_run"]


def calc_decision_point(df:pd.DataFrame):
    """
    스윙 결정 시점(t=0.25초)에서의 보정된 x, y, z 좌표 계산
    
    Parameters:
        df: PITCH_DESCRIPTION, VX0, VY0, VZ0, AX, AY, AZ, 
            RELEASE_POS_X, RELEASE_POS_Z, PLATE_X, PLATE_Z 컬럼 필요
        t_decision: 스윙 결정 시점 (기본 0.25초)
    
    Returns:
        DataFrame with columns: X_AT_DECISION, Y_AT_DECISION, Z_AT_DECISION
    """
    t_decision=0.25

    # ── 홈플레이트 도달 시간 계산 ──
    # y(t) = RELEASE_POS_Y + VY0*t + 0.5*AY*t² = 0
    a = 0.5 * df['AY']
    b = df['VY0']
    c = df['RELEASE_POS_Y'] - 1.41

    discriminant = b**2 - 4*a*c
    t_plate = (-b - np.sqrt(discriminant)) / (2 * a)

    # ── 궤적 공식으로 예측한 홈플레이트 좌표 ──
    pred_plate_x = df['RELEASE_POS_X'] + df['VX0'] * t_plate + 0.5 * df['AX'] * t_plate**2
    pred_plate_z = df['RELEASE_POS_Z'] + df['VZ0'] * t_plate + 0.5 * df['AZ'] * t_plate**2

    # ── 보정 계수 ──
    error_x = df['PLATE_X'] - pred_plate_x
    error_z = df['PLATE_Z'] - pred_plate_z
    correction_ratio = t_decision / t_plate

    # ── 결정 시점 원시 좌표 ──
    raw_x = df['RELEASE_POS_X'] + df['VX0'] * t_decision + 0.5 * df['AX'] * t_decision**2
    raw_z = df['RELEASE_POS_Z'] + df['VZ0'] * t_decision + 0.5 * df['AZ'] * t_decision**2
    raw_y = df['RELEASE_POS_Y'] + df['VY0'] * t_decision + 0.5 * df['AY'] * t_decision**2

    # ── 보정 적용 ──
    result = pd.DataFrame({
        'X_AT_DECISION': raw_x + error_x * correction_ratio,
        'Y_AT_DECISION': raw_y,
        'Z_AT_DECISION': raw_z + error_z * correction_ratio,
    }, index=df.index)

    return result

def my_model(df: pd.DataFrame):
    # 데이터 전처리 
    pitch_features = [
        'RELEASE_SPEED', 'AX', 'AZ',
        'PFX_X', 'PFX_Z',
        'RELEASE_SPIN_RATE', 
        'X_AT_DECISION', 'Y_AT_DECISION', 'Z_AT_DECISION'
    ]

    rhp = df[(df['P_THROWS'] == 'R') & (df['PITCH_TYPE'].isin(['FF', 'SL']))]
    rhp = rhp.sort_values(['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])
    rhp['DID_SWING'] = np.where(
        rhp['PITCH_DESCRIPTION'].isin(
            ['swinging_strike', 'swinging_strike_blocked', 'foul',
             'foul_tip', 'foul_bunt', 'hit_into_play']
        ), 1, 0
    )

    col = calc_decision_point(df)
    rhp = pd.concat([rhp, col], axis=1)
    rhp = rhp.dropna(subset=pitch_features)

    # ── PREV 원시값 ──
    group = rhp.groupby(['GAME_PK', 'AT_BAT_NUMBER'])
    prev_cols = ['RELEASE_SPEED', 'AX', 'AZ', 'X_AT_DECISION', 'Z_AT_DECISION', 'PLATE_X', 'PLATE_Z']
    prev_df = group[prev_cols].shift(1).add_prefix('PREV_')
    rhp = pd.concat([rhp, prev_df], axis=1)

    # ── 궤적 괴리 비율 ──
    rhp['MOVE_RATE'] = (rhp['PLATE_X'] - rhp['X_AT_DECISION']) / rhp['X_AT_DECISION']
    rhp['DROP_RATE'] = (rhp['Z_AT_DECISION'] - rhp['PLATE_Z']) / rhp['Z_AT_DECISION']

    # ── 직전 대비 변화량 ──
    rhp['X_AT_DECISION_DIFF'] = rhp['X_AT_DECISION'] - rhp['PREV_X_AT_DECISION']
    rhp['Z_AT_DECISION_DIFF'] = rhp['Z_AT_DECISION'] - rhp['PREV_Z_AT_DECISION']

    # ── NaN 처리: PREV는 전체 평균으로, DIFF는 0으로 ──
    prev_fill_cols = ['PREV_RELEASE_SPEED', 'PREV_AX', 'PREV_AZ',
                      'PREV_X_AT_DECISION', 'PREV_Z_AT_DECISION',
                      'PREV_PLATE_X', 'PREV_PLATE_Z']
    for c in prev_fill_cols:
        original = c.replace('PREV_', '')
        rhp[c] = rhp[c].fillna(rhp[original].mean())

    rhp['X_AT_DECISION_DIFF'] = rhp['X_AT_DECISION_DIFF'].fillna(0)
    rhp['Z_AT_DECISION_DIFF'] = rhp['Z_AT_DECISION_DIFF'].fillna(0)
    rhp['MOVE_RATE'] = rhp['MOVE_RATE'].fillna(0)
    rhp['DROP_RATE'] = rhp['DROP_RATE'].fillna(0)

    # ── 존 안/밖 판별 (타자 성향 계산용) ──
    DZ_TOP = 4.69
    DZ_BOT = 3.76
    DZ_EDGE = 0.63

    rhp['IN_ZONE_AT_DECISION'] = (
        (abs(rhp['X_AT_DECISION']) <= DZ_EDGE) &
        (rhp['Z_AT_DECISION'] >= DZ_BOT) &
        (rhp['Z_AT_DECISION'] <= DZ_TOP)
    ).astype(int)
    
    # 정렬 (타자별 시간순)
    rhp = rhp.sort_values(['BATTER', 'GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])

    # ── 1. 타자 누적 스윙율 (과거 이력만) ──
    rhp['BATTER_CUM_SWING'] = (
        rhp.groupby('BATTER')['DID_SWING']
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # ── 2. 존 안/밖 별 스윙율 ──
    # 존 안 공에 대한 스윙율 (공격성)
    rhp['IN_ZONE_SWING'] = rhp['IN_ZONE_AT_DECISION'] * rhp['DID_SWING']
    rhp['BATTER_ZONE_SWING'] = (
        rhp.groupby('BATTER')['IN_ZONE_SWING']
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # 존 밖 공에 대한 스윙율 (선구안)
    rhp['OUT_ZONE_SWING'] = (1 - rhp['IN_ZONE_AT_DECISION']) * rhp['DID_SWING']
    rhp['BATTER_CHASE_RATE'] = (
        rhp.groupby('BATTER')['OUT_ZONE_SWING']
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # ── 3. 해당 타석 내 스윙율 (현재 타석의 적극성) ──
    rhp = rhp.sort_values(['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])
    rhp['AB_CUM_SWING'] = (
        rhp.groupby(['GAME_PK', 'AT_BAT_NUMBER'])['DID_SWING']
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # ── 4. 타자의 누적 투구 수 (경험치) ──
    rhp = rhp.sort_values(['BATTER', 'GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])
    rhp['BATTER_PITCH_COUNT'] = rhp.groupby('BATTER').cumcount()

    # ── NaN 처리 ──
    global_swing = rhp['DID_SWING'].mean()
    rhp['BATTER_CUM_SWING'] = rhp['BATTER_CUM_SWING'].fillna(global_swing)
    rhp['BATTER_ZONE_SWING'] = rhp['BATTER_ZONE_SWING'].fillna(global_swing)
    rhp['BATTER_CHASE_RATE'] = rhp['BATTER_CHASE_RATE'].fillna(global_swing)
    rhp['AB_CUM_SWING'] = rhp['AB_CUM_SWING'].fillna(global_swing)

    # 중간 계산용 컬럼 제거
    rhp = rhp.drop(columns=['IN_ZONE_SWING', 'OUT_ZONE_SWING', 'PLATE_X', 'PLATE_Z'])

    # =====================================================
    # =====================================================
    # # ── 최종 feature 리스트 ──
    # model_features = [
    #     'RELEASE_SPEED', 'AX', 'AZ',
    #     'X_AT_DECISION', 'Z_AT_DECISION',
    #     'MOVE_RATE', 'DROP_RATE',
    #     'X_AT_DECISION_DIFF', 'Z_AT_DECISION_DIFF',
    #     'PREV_RELEASE_SPEED', 'PREV_AX', 'PREV_AZ',
    #     'PREV_X_AT_DECISION', 'PREV_Z_AT_DECISION',
    #     'STRIKES', 'BALLS',
    #     'PREV_PLATE_X', 'PREV_PLATE_Z',
    #     # 타자 관련 features 
    #     'BATTER_ZONE_SWING', 'BATTER_CHASE_RATE'
    # ]

    # non_miller_df = rhp[rhp['PITCHER'] != 695243].copy()
    # miller_df = rhp[rhp['PITCHER'] == 695243].copy()

    # # 밀러 데이터 분리 
    # X_miller = miller_df[model_features].copy()
    # X_miller['STAND'] = (miller_df['STAND'] == 'R').astype(int)
    # y_miller = miller_df['DID_SWING']

    # X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    #     X_miller, y_miller, test_size=.3, random_state=42
    # )

    # # 베이스 학습 : non-밀러 전체 + 밀러 train 데이터
    # base_df = pd.concat([non_miller_df, miller_df.loc[X_m_train.index]])

    # X = base_df[model_features].copy()
    # X['STAND'] = (base_df['STAND'] == 'R').astype(int)
    # y = base_df['DID_SWING']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    # from lightgbm import LGBMClassifier

    # lgbm = LGBMClassifier(
    #     n_estimators=500,
    #     max_depth=4,
    #     learning_rate=.1,
    #     num_leaves=31,
    #     min_child_samples=50,
    #     subsample=.7,
    #     reg_alpha=1.0,
    #     reg_lambda=.1,
    #     colsample_bytree=.9,
    #     random_state=42,
    #     verbose=-1
    # )

    # lgbm.fit(X_train, y_train)
    # y_hat = lgbm.predict(X_test)

    # print(" === Base 모델 평가 지표 === ")
    # print(f"Train score: {lgbm.score(X_train, y_train):.4f}")
    # print(f"Test score:  {lgbm.score(X_test, y_test):.4f}")
    # print(metrics.classification_report(y_test, y_hat))

    # # ===================================== 
    # # 전이 학습 (base → Miller fine-tune)   
    # # =====================================

    # fine_model = LGBMClassifier(
    #     n_estimators=50,        # 적은 트리로 미세 조정
    #     max_depth=4,
    #     learning_rate=0.01,     # 작은 보폭
    #     num_leaves=31,
    #     min_child_samples=10,   # Miller 데이터가 적으니 완화
    #     subsample=0.8,
    #     colsample_bytree=0.9,
    #     random_state=42,
    #     verbose=-1
    # )
    # fine_model.fit(X_m_train, y_m_train, init_model=lgbm)
    # y_hat_fine = fine_model.predict(X_m_test)
    # print("=== 전이 학습 (fine-tune) ===")
    # print(f"Score: {fine_model.score(X_m_test, y_m_test):.4f}")
    # print(metrics.classification_report(y_m_test, y_hat_fine))
    # print(f"ROC-AUC 점수 : {metrics.roc_auc_score(y_m_test, y_hat_fine):>.4f}")
    # ================================================================================
    # =================================================================================

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

    # ── 2. LSTM 모델 구조 정의 (개선 ver)──
    class SwingPredictionLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=2, dropout=.2):
            super(SwingPredictionLSTM, self).__init__()
            
            # LSTM 계층: batch_first=True로 설정해야 (Batch, Seq, Feature) 형태를 받아들입니다.
            self.lstm = nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True,
                dropout = dropout
            )
            
            # 출력층: LSTM이 요약한 정보를 바탕으로 '해당 투구'의 스윙 확률(로짓)을 계산
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
            
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
    hidden_size = 128      # LSTM 내부에서 기억을 유지하는 벡터의 크기 (크면 더 복잡한 패턴 기억 가능)
    num_layers = 2        # LSTM 층의 개수

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

    # 학습률 스케줄러: 3 epoch 동안 개선 없으면 lr을 절반으로
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=.5
    )

    epochs = 30  
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 7  # 7 epoch 동안 개선 없으면 조기 종료

    print("🚀 학습 시작...\n")

    for epoch in range(epochs):
        
        # 학습 모드 전환
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
        
        
        # 검증 모드 전환 
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
        
        # 학습률 스케줄러 업데이트
        scheduler.step(avg_val_loss)
        
        # Early Stopping 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 최고 성능 모델 저장
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # 현재 학습률 확인
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:2d}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"{'★ Best' if patience_counter == 0 else ''}")
        
        if patience_counter >= early_stop_patience:
            print(f"\n⚠ Early Stopping: {early_stop_patience} epoch 동안 개선 없음")
            break

        
    model.load_state_dict(best_state)
    print("\n학습 완료! (Best Val Loss로 모델 복원)")

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
    print(metrics.classification_report(all_true_labels, pred_classes))
    print(f"Accuracy (정확도): {accuracy:.4f}")
    print(f"ROC-AUC 점수   : {roc_auc:.4f}")

if __name__ == "__main__":
    try:
        conn = DB.connect(user="C##AKSEN", password="1234", dsn="localhost:1521/xe")
        cursor = conn.cursor()

        sql = """
            SELECT *
            FROM TMP_TABLE
        """

        cursor.execute(sql)
        df = pd.DataFrame(
            data= cursor.fetchall(), 
            columns= [d[0] for d in cursor.description]
        )

        # hit_distance_analysis(df)
        # pfx_z_analysis(df)
        # pitch_type_classification(df)
        # Masson_Miller(df)    
        my_model(df)


    except Exception as e:
        traceback.print_exc()
    finally:
        if "conn" in locals():
            conn.close()