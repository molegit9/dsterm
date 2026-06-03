import pandas as pd
import numpy as np
import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lightgbm import LGBMRegressor

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Workspace directories
workspace_dir = "c:/Dev/open"
train_path = os.path.join(workspace_dir, "train/train.csv")
test_dir = os.path.join(workspace_dir, "test")
sample_submission_path = os.path.join(workspace_dir, "sample_submission.csv")
submission_path = os.path.join(workspace_dir, "submission.csv")

# 1. Test T calendar date mappings
test_t_mappings = {
    0: (10, 1),
    1: (11, 1),
    2: (4, 1),
    3: (5, 3),
    4: (5, 1),
    5: (6, 2),
    6: (8, 3),
    7: (10, 2),
    8: (8, 2),
    9: (4, 2),
    10: (3, 3),
    11: (6, 1),
    12: (7, 1),
    13: (7, 2),
    14: (9, 3),
    15: (10, 3),
    16: (9, 2),
    17: (11, 2),
    18: (9, 1),
    19: (4, 3),
    20: (8, 1),
    21: (6, 3),
    22: (7, 3),
    23: (11, 3),
    24: (5, 2),
}

# Helper to convert flat index back to year, month, soon, season
def flat_idx_to_date(flat_idx):
    year = 2018 + (flat_idx // 36)
    rem = flat_idx % 36
    month = 1 + (rem // 3)
    soon = 1 + (rem % 3)
    
    if month in [3, 4, 5]:
        season = 1
    elif month in [6, 7, 8]:
        season = 2
    elif month in [9, 10, 11]:
        season = 3
    else:
        season = 4
        
    return year, month, soon, season

# Helper to encode calendar features
def get_calendar_features(year, month, soon, season):
    y_norm = (year - 2018) / 4.0
    m_sin = np.sin(2 * np.pi * month / 12)
    m_cos = np.cos(2 * np.pi * month / 12)
    s_sin = np.sin(2 * np.pi * soon / 3)
    s_cos = np.cos(2 * np.pi * soon / 3)
    season_oh = [0.0] * 4
    season_oh[season - 1] = 1.0
    
    return [y_norm, m_sin, m_cos, s_sin, s_cos] + season_oh

# Item specifications and filters
items_conditions = {
    '감자': lambda df: df[(df['품종명'] == '감자 수미') & (df['거래단위'] == '20키로상자') & (df['등급'] == '상')],
    '건고추': lambda df: df[(df['품종명'] == '화건') & (df['거래단위'] == '30 kg') & (df['등급'] == '상품')],
    '깐마늘(국산)': lambda df: df[(df['품목명'] == '깐마늘(국산)') & (df['거래단위'] == '20 kg') & (df['등급'] == '상품')],
    '대파': lambda df: df[(df['품종명'] == '대파(일반)') & (df['거래단위'] == '1키로단') & (df['등급'] == '상')],
    '무': lambda df: df[(df['품목명'] == '무') & (df['거래단위'] == '20키로상자') & (df['등급'] == '상')],
    '배추': lambda df: df[(df['품목명'] == '배추') & (df['거래단위'] == '10키로망대') & (df['등급'] == '상')],
    '사과': lambda df: df[(df['품목명'] == '사과') & (df['품종명'].isin(['홍로', '후지'])) & (df['거래단위'] == '10 개') & (df['등급'] == '상품')],
    '상추': lambda df: df[(df['품목명'] == '상추') & (df['품종명'] == '청') & (df['거래단위'] == '100 g') & (df['등급'] == '상품')],
    '양파': lambda df: df[(df['품목명'] == '양파') & (df['품종명'] == '양파') & (df['거래단위'] == '1키로') & (df['등급'] == '상')],
    '배': lambda df: df[(df['품목명'] == '배') & (df['품종명'] == '신고') & (df['거래단위'] == '10 개') & (df['등급'] == '상품')]
}

# Dataset Definition for PyTorch
class SequenceDataset(Dataset):
    def __init__(self, prices, seasonal_mean, start_idx_list):
        self.prices = prices
        self.seasonal_mean = seasonal_mean
        self.start_indices = start_idx_list
        
    def __len__(self):
        return len(self.start_indices)
        
    def __getitem__(self, idx):
        start = self.start_indices[idx]
        in_prices = self.prices[start : start + 9]
        scale = in_prices[-1]
        if scale == 0:
            scale = np.mean(self.prices) if np.mean(self.prices) > 0 else 1.0
            
        target_prices = self.prices[start + 9 : start + 12]
        
        in_features = []
        for i in range(9):
            step_idx = start + i
            yr, m, sn, seas = flat_idx_to_date(step_idx)
            cal_feats = get_calendar_features(yr, m, sn, seas)
            
            p_scaled = in_prices[i] / scale
            sm_scaled = self.seasonal_mean[step_idx % 36] / scale
            
            if i == 0:
                pct = 0.0
            else:
                pct = (in_prices[i] - in_prices[i-1]) / (in_prices[i-1] + 1e-8)
                
            in_features.append([p_scaled, sm_scaled, pct] + cal_feats)
            
        target_cal_features = []
        for i in range(3):
            step_idx = start + 9 + i
            yr, m, sn, seas = flat_idx_to_date(step_idx)
            cal_feats = get_calendar_features(yr, m, sn, seas)
            sm_scaled = self.seasonal_mean[step_idx % 36] / scale
            target_cal_features.extend([sm_scaled] + cal_feats)
            
        return (
            torch.tensor(in_features, dtype=torch.float32),
            torch.tensor(target_cal_features, dtype=torch.float32),
            torch.tensor(target_prices / scale, dtype=torch.float32),
            torch.tensor(scale, dtype=torch.float32)
        )

# Custom NMAE Loss
class CustomNMAELoss(nn.Module):
    def __init__(self):
        super(CustomNMAELoss, self).__init__()
        
    def forward(self, pred_scaled, y_target, scale):
        if scale.dim() == 1:
            scale = scale.unsqueeze(1)
        pred_actual = pred_scaled * scale
        y_actual = y_target * scale
        loss = torch.sum(torch.abs(pred_actual - y_actual)) / (torch.sum(torch.abs(y_actual)) + 1e-8)
        return loss

# PyTorch GRU-MLP 모델 아키텍처
class PriceGRUMLP(nn.Module):
    def __init__(self, input_dim=12, target_cal_dim=30, hidden_dim=64):
        super(PriceGRUMLP, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + target_cal_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 3)
        )
        
    def forward(self, x_seq, x_target_cal):
        _, h_n = self.gru(x_seq)
        h_n = h_n.squeeze(0)
        concat = torch.cat([h_n, x_target_cal], dim=-1)
        out = self.mlp(concat)
        return out

# LightGBM 37차원 스케일 정규화 피처 생성 함수
def get_lgbm_features_scaled(prices_series, seasonal_mean, start_idx, horizon, scale):
    in_p = prices_series[start_idx : start_idx + 9]
    in_p_scaled = in_p / scale
    
    yr_t, m_t, sn_t, seas_t = flat_idx_to_date(start_idx + 8)
    cal_feats_t = get_calendar_features(yr_t, m_t, sn_t, seas_t)
    sm_t_scaled = seasonal_mean[(start_idx + 8) % 36] / scale
    
    pct_changes = []
    for i in range(1, 9):
        pct = (in_p[i] - in_p[i-1]) / (in_p[i-1] + 1e-8)
        pct_changes.append(pct)
        
    yr_th, m_th, sn_th, seas_th = flat_idx_to_date(start_idx + 8 + horizon)
    cal_feats_th = get_calendar_features(yr_th, m_th, sn_th, seas_th)
    sm_th_scaled = seasonal_mean[(start_idx + 8 + horizon) % 36] / scale
    
    feat = list(in_p_scaled) + cal_feats_t + [sm_t_scaled] + pct_changes + cal_feats_th + [sm_th_scaled]
    return feat

def get_test_lgbm_features_scaled(test_prices, seasonal_mean, t_flat_idx, horizon, scale):
    in_p_scaled = test_prices / scale
    yr_t, m_t, sn_t, seas_t = flat_idx_to_date(t_flat_idx)
    cal_feats_t = get_calendar_features(yr_t, m_t, sn_t, seas_t)
    sm_t_scaled = seasonal_mean[t_flat_idx % 36] / scale
    
    pct_changes = []
    for i in range(1, 9):
        pct = (test_prices[i] - test_prices[i-1]) / (test_prices[i-1] + 1e-8)
        pct_changes.append(pct)
        
    yr_th, m_th, sn_th, seas_th = flat_idx_to_date(t_flat_idx + horizon)
    cal_feats_th = get_calendar_features(yr_th, m_th, sn_th, seas_th)
    sm_th_scaled = seasonal_mean[(t_flat_idx + horizon) % 36] / scale
    
    feat = list(in_p_scaled) + cal_feats_t + [sm_t_scaled] + pct_changes + cal_feats_th + [sm_th_scaled]
    return feat

def main():
    print(f"Using device: {device}")
    
    # 2. Chronological periods from 201801상순 to 202112하순 (144 periods)
    months_list = [f"{m:02d}" for m in range(1, 13)]
    soons_list = ["상순", "중순", "하순"]
    years_list = ["2018", "2019", "2020", "2021"]
    all_train_periods = [f"{y}{m}{s}" for y in years_list for m in months_list for s in soons_list]
    
    # Load train.csv and clean/align data
    print("Loading train.csv...")
    train_df = pd.read_csv(train_path, encoding='utf-8-sig')
    
    train_series = {}
    train_seasonal_means = {}
    
    for item, cond in items_conditions.items():
        sub_df = cond(train_df).copy()
        grouped = sub_df.groupby('시점')['평균가격(원)'].max().reset_index()
        grouped = grouped.set_index('시점').reindex(all_train_periods)
        prices = grouped['평균가격(원)'].replace(0, np.nan).values
        
        mask = np.isnan(prices)
        if mask.all():
            prices = np.zeros_like(prices)
        else:
            idx = np.where(~mask)[0]
            prices[:idx[0]] = prices[idx[0]]
            prices[idx[-1]:] = prices[idx[-1]]
            for i in range(len(idx) - 1):
                prices[idx[i]:idx[i+1]] = prices[idx[i]]
                
        train_series[item] = prices
        
        # Calculate seasonal means
        seasonal_sum = np.zeros(36)
        seasonal_cnt = np.zeros(36)
        for idx, p in enumerate(prices):
            p_idx = idx % 36
            seasonal_sum[p_idx] += p
            seasonal_cnt[p_idx] += 1
        train_seasonal_means[item] = seasonal_sum / seasonal_cnt
        
    seeds = [42, 100, 2021, 2022, 2026] # 5-seed ensembling
    models_dict = {}
    lgbm_models_dict = {}
    blended_val_nmaes = {}
    
    for item in items_conditions.keys():
        print(f"\n======================================")
        print(f"Training Models for: {item}")
        print(f"======================================")
        prices = train_series[item]
        seasonal_mean = train_seasonal_means[item]
        
        train_indices = list(range(110))
        val_indices = list(range(110, 133))
        
        models_dict[item] = []
        lgbm_models_dict[item] = []
        
        for seed_idx, seed in enumerate(seeds):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                
            print(f"[{item}] Training Seed {seed} ({seed_idx+1}/{len(seeds)})...")
            
            # --- PyTorch GRU 모델 학습 ---
            train_dataset = SequenceDataset(prices, seasonal_mean, train_indices)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            
            model = PriceGRUMLP(input_dim=12, target_cal_dim=30, hidden_dim=64).to(device)
            criterion = CustomNMAELoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=180)
            
            val_dataset = SequenceDataset(prices, seasonal_mean, val_indices)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            best_val_loss = float('inf')
            best_model_state = None
            
            for epoch in range(180):
                model.train()
                for x_seq, x_target_cal, y_target, scale in train_loader:
                    x_seq, x_target_cal, y_target, scale = x_seq.to(device), x_target_cal.to(device), y_target.to(device), scale.to(device)
                    
                    optimizer.zero_grad()
                    pred = model(x_seq, x_target_cal)
                    loss = criterion(pred, y_target, scale)
                    loss.backward()
                    optimizer.step()
                    
                scheduler.step()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x_seq, x_target_cal, y_target, scale in val_loader:
                        x_seq, x_target_cal, y_target, scale = x_seq.to(device), x_target_cal.to(device), y_target.to(device), scale.to(device)
                        pred = model(x_seq, x_target_cal)
                        loss = criterion(pred, y_target, scale)
                        val_loss += loss.item() * x_seq.size(0)
                val_loss /= len(val_dataset)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    
            model.load_state_dict(best_model_state)
            models_dict[item].append(model)
            
            # --- LightGBM 모델 학습 (Scale Normalization 적용) ---
            lgbm_models = []
            for horizon in [1, 2, 3]:
                lgbm_features = []
                lgbm_targets_scaled = []
                for start_idx in train_indices:
                    scale = prices[start_idx + 8]
                    if scale == 0:
                        scale = np.mean(prices) if np.mean(prices) > 0 else 1.0
                    feat = get_lgbm_features_scaled(prices, seasonal_mean, start_idx, horizon, scale)
                    lgbm_features.append(feat)
                    lgbm_targets_scaled.append(prices[start_idx + 8 + horizon] / scale)
                    
                lgbm_features = np.array(lgbm_features)
                lgbm_targets_scaled = np.array(lgbm_targets_scaled)
                
                lgbm_model = LGBMRegressor(
                    random_state=seed,
                    n_estimators=300,
                    learning_rate=0.02,
                    max_depth=4,
                    min_child_samples=5,
                    colsample_bytree=0.8,
                    subsample=0.8,
                    verbosity=-1
                )
                lgbm_model.fit(lgbm_features, lgbm_targets_scaled)
                lgbm_models.append(lgbm_model)
            lgbm_models_dict[item].append(lgbm_models)
            
        # --- 검증 셋 평가: 5개 시드 앙상블 블렌딩 ---
        val_dataset = SequenceDataset(prices, seasonal_mean, val_indices)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        gru_val_preds_all = []
        val_actual_values = []
        val_scales = []
        
        for x_seq, x_target_cal, y_target, scale in val_loader:
            x_seq, x_target_cal = x_seq.to(device), x_target_cal.to(device)
            batch_preds = []
            for model in models_dict[item]:
                model.eval()
                with torch.no_grad():
                    pred_scaled = model(x_seq, x_target_cal).cpu().numpy()
                    pred_actual = pred_scaled * scale.numpy()[:, None]
                    batch_preds.append(pred_actual)
            gru_val_preds_all.extend(np.mean(batch_preds, axis=0))
            val_actual_values.extend(y_target.numpy() * scale.numpy()[:, None])
            val_scales.extend(scale.numpy())
            
        gru_val_preds_all = np.array(gru_val_preds_all)
        val_actual_values = np.array(val_actual_values)
        val_scales = np.array(val_scales)
        
        # LightGBM 검증셋 예측 (Scale-Normalized 5개 시드 평균)
        lgbm_val_preds_all = np.zeros_like(val_actual_values)
        for horizon_idx, horizon in enumerate([1, 2, 3]):
            val_lgbm_features = []
            for idx, start_idx in enumerate(val_indices):
                scale = val_scales[idx]
                feat = get_lgbm_features_scaled(prices, seasonal_mean, start_idx, horizon, scale)
                val_lgbm_features.append(feat)
            val_lgbm_features = np.array(val_lgbm_features)
            
            horizon_preds_scaled = []
            for lgbm_models in lgbm_models_dict[item]:
                pred_scaled = lgbm_models[horizon_idx].predict(val_lgbm_features)
                horizon_preds_scaled.append(pred_scaled)
            
            mean_pred_scaled = np.mean(horizon_preds_scaled, axis=0)
            lgbm_val_preds_all[:, horizon_idx] = mean_pred_scaled * val_scales
            
        # 가중 블렌딩
        blended_val_preds = 0.4 * gru_val_preds_all + 0.6 * lgbm_val_preds_all
        blended_val_preds = np.clip(blended_val_preds, 0.0, None)
        
        abs_errors = np.abs(val_actual_values - blended_val_preds)
        val_nmae = np.sum(abs_errors) / (np.sum(val_actual_values) + 1e-8)
        blended_val_nmaes[item] = val_nmae
        print(f"--> Blended 5-Seed Ensemble Validation NMAE for {item}: {val_nmae:.4f}")
        
    print("\n=== Mean Blended Validation NMAE (Seed Ensemble with Scale Normalization) ===")
    mean_nmae = np.mean(list(blended_val_nmaes.values()))
    print(f"Mean Blended NMAE: {mean_nmae:.4f}")
    
    # 8. Inference on TEST sets
    print("\nStarting inference on TEST datasets...")
    predictions = {}
    
    for test_idx in range(25):
        predictions[test_idx] = {}
        
        test_file_path = os.path.join(test_dir, f"TEST_{test_idx:02d}.csv")
        test_df = pd.read_csv(test_file_path, encoding='utf-8-sig')
        
        m_t, soon_t = test_t_mappings[test_idx]
        t_flat_idx = (2022 - 2018) * 36 + (m_t - 1) * 3 + (soon_t - 1)
        
        for item in items_conditions.keys():
            cond = items_conditions[item]
            sub_test = cond(test_df).copy()
            
            steps = [f"T-{i}순" for i in range(8, 0, -1)] + ["T"]
            
            grouped_test = sub_test.groupby('시점')['평균가격(원)'].max().reset_index()
            grouped_test = grouped_test.set_index('시점').reindex(steps)
            
            test_prices = grouped_test['평균가격(원)'].replace(0, np.nan).values
            mask = np.isnan(test_prices)
            if mask.all():
                test_prices = np.array([train_seasonal_means[item][(t_flat_idx - 8 + i) % 36] for i in range(9)])
            else:
                idx = np.where(~mask)[0]
                test_prices[:idx[0]] = test_prices[idx[0]]
                test_prices[idx[-1]:] = test_prices[idx[-1]]
                for i in range(len(idx) - 1):
                    test_prices[idx[i]:idx[i+1]] = test_prices[idx[i]]
                    
            scale = test_prices[-1]
            if scale == 0:
                scale = np.mean(train_series[item]) if np.mean(train_series[item]) > 0 else 1.0
                
            # [1] GRU 테스트 추론 (5개 시드 앙상블 평균)
            in_features = []
            for i in range(9):
                step_idx = t_flat_idx - 8 + i
                yr, m, sn, seas = flat_idx_to_date(step_idx)
                cal_feats = get_calendar_features(yr, m, sn, seas)
                
                p_scaled = test_prices[i] / scale
                sm_scaled = train_seasonal_means[item][step_idx % 36] / scale
                
                if i == 0:
                    pct = 0.0
                else:
                    pct = (test_prices[i] - test_prices[i-1]) / (test_prices[i-1] + 1e-8)
                    
                in_features.append([p_scaled, sm_scaled, pct] + cal_feats)
                
            target_cal_features = []
            for i in range(3):
                step_idx = t_flat_idx + 1 + i
                yr, m, sn, seas = flat_idx_to_date(step_idx)
                cal_feats = get_calendar_features(yr, m, sn, seas)
                sm_scaled = train_seasonal_means[item][step_idx % 36] / scale
                target_cal_features.extend([sm_scaled] + cal_feats)
                
            x_seq_tensor = torch.tensor([in_features], dtype=torch.float32).to(device)
            x_target_cal_tensor = torch.tensor([target_cal_features], dtype=torch.float32).to(device)
            
            gru_preds_list = []
            for model in models_dict[item]:
                model.eval()
                with torch.no_grad():
                    pred_scaled_gru = model(x_seq_tensor, x_target_cal_tensor).cpu().numpy()[0]
                gru_preds_list.append(pred_scaled_gru * scale)
            pred_actual_gru = np.mean(gru_preds_list, axis=0)
            
            # [2] LightGBM 테스트 추론 (Scale-Normalized 5개 시드 평균)
            pred_actual_lgbm = np.zeros(3)
            for horizon_idx, horizon in enumerate([1, 2, 3]):
                lgbm_feat = get_test_lgbm_features_scaled(test_prices, train_seasonal_means[item], t_flat_idx, horizon, scale)
                
                horizon_preds_scaled = []
                for lgbm_models in lgbm_models_dict[item]:
                    pred_scaled = lgbm_models[horizon_idx].predict([lgbm_feat])[0]
                    horizon_preds_scaled.append(pred_scaled)
                pred_actual_lgbm[horizon_idx] = np.mean(horizon_preds_scaled, axis=0) * scale
                
            # [3] 블렌딩 및 음수 차단
            pred_actual = 0.4 * pred_actual_gru + 0.6 * pred_actual_lgbm
            pred_actual = np.clip(pred_actual, 0.0, None)
            
            predictions[test_idx][item] = pred_actual
            
    # 9. Format submission.csv
    print("\nGenerating submission.csv...")
    sub_df = pd.read_csv(sample_submission_path, encoding='utf-8-sig')
    
    item_col_map = {
        '감자': '감자',
        '건고추': '건고추',
        '깐마늘(국산)': '깐마늘(국산)',
        '대파': '대파',
        '무': '무',
        '배추': '배추',
        '사과': '사과',
        '상추': '상추',
        '양파': '양파',
        '배': '배'
    }
    
    for i in range(len(sub_df)):
        row = sub_df.iloc[i]
        sijum = row['시점']
        test_idx = int(sijum.split('+')[0].split('_')[1])
        horizon = int(sijum.split('+')[1][0])
        
        for item, col in item_col_map.items():
            pred_val = predictions[test_idx][item][horizon - 1]
            sub_df.at[i, col] = pred_val
            
    # Save submission.csv
    sub_df.to_csv(submission_path, index=False, encoding='utf-8-sig')
    print("Successfully generated submission.csv!")

if __name__ == '__main__':
    main()
