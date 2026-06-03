import pandas as pd
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Workspace directories
workspace_dir = "c:/Dev/open"
train_path = os.path.join(workspace_dir, "train/train.csv")
test_dir = os.path.join(workspace_dir, "test")
sample_submission_path = os.path.join(workspace_dir, "sample_submission.csv")
submission_path = os.path.join(workspace_dir, "submission.csv")

# 1. Test T calendar date mappings
# Format: test_idx -> (month, soon_idx) where soon_idx: 1=상순, 2=중순, 3=하순
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

# 2. Chronological periods from 201801상순 to 202112하순 (144 periods)
months_list = [f"{m:02d}" for m in range(1, 13)]
soons_list = ["상순", "중순", "하순"]
years_list = ["2018", "2019", "2020", "2021"]
all_train_periods = [f"{y}{m}{s}" for y in years_list for m in months_list for s in soons_list]

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
    # Year normalized
    y_norm = (year - 2018) / 4.0
    # Month sin/cos
    m_sin = np.sin(2 * np.pi * month / 12)
    m_cos = np.cos(2 * np.pi * month / 12)
    # Soon sin/cos
    s_sin = np.sin(2 * np.pi * soon / 3)
    s_cos = np.cos(2 * np.pi * soon / 3)
    # Season one-hot
    season_oh = [0.0] * 4
    season_oh[season - 1] = 1.0
    
    return [y_norm, m_sin, m_cos, s_sin, s_cos] + season_oh

# 3. Item specifications and filters
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

# 4. Load train.csv and clean/align data
print("Loading train.csv...")
train_df = pd.read_csv(train_path, encoding='utf-8-sig')

# Build the cleaned price series dictionary: item -> prices array of length 144
train_series = {}
train_seasonal_means = {} # item -> array of length 36

for item, cond in items_conditions.items():
    sub_df = cond(train_df).copy()
    grouped = sub_df.groupby('시점')['평균가격(원)'].max().reset_index()
    # Reindex to all 144 periods
    grouped = grouped.set_index('시점').reindex(all_train_periods)
    # Clean zeros and NaNs
    prices = grouped['평균가격(원)'].replace(0, np.nan).values
    # Forward/backward fill
    mask = np.isnan(prices)
    if mask.all():
        prices = np.zeros_like(prices)
    else:
        # manual ffill/bfill to avoid pandas deprecation warning
        idx = np.where(~mask)[0]
        prices[:idx[0]] = prices[idx[0]]
        prices[idx[-1]:] = prices[idx[-1]]
        for i in range(len(idx) - 1):
            prices[idx[i]:idx[i+1]] = prices[idx[i]]
            
    train_series[item] = prices
    
    # Calculate seasonal means (average price for each of the 36 periods)
    seasonal_sum = np.zeros(36)
    seasonal_cnt = np.zeros(36)
    for idx, p in enumerate(prices):
        p_idx = idx % 36
        seasonal_sum[p_idx] += p
        seasonal_cnt[p_idx] += 1
    train_seasonal_means[item] = seasonal_sum / seasonal_cnt

# 5. Dataset Definition
class SequenceDataset(Dataset):
    def __init__(self, prices, seasonal_mean, start_idx_list):
        self.prices = prices
        self.seasonal_mean = seasonal_mean
        self.start_indices = start_idx_list
        
    def __len__(self):
        return len(self.start_indices)
        
    def __getitem__(self, idx):
        start = self.start_indices[idx]
        # Input sequence (9 steps)
        in_prices = self.prices[start : start + 9]
        # Target scale: price at T (index start + 8)
        scale = in_prices[-1]
        if scale == 0:
            scale = np.mean(self.prices) if np.mean(self.prices) > 0 else 1.0
            
        # Target prices (3 steps)
        target_prices = self.prices[start + 9 : start + 12]
        
        # Build features for 9 input steps
        in_features = []
        for i in range(9):
            step_idx = start + i
            # Flat calendar index to date details
            yr, m, sn, seas = flat_idx_to_date(step_idx)
            cal_feats = get_calendar_features(yr, m, sn, seas)
            
            # Scaled price and seasonal mean
            p_scaled = in_prices[i] / scale
            sm_scaled = self.seasonal_mean[step_idx % 36] / scale
            
            in_features.append([p_scaled, sm_scaled] + cal_feats)
            
        # Build calendar features for 3 target steps
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

# 6. PyTorch Model Architecture
class PriceGRUMLP(nn.Module):
    def __init__(self, input_dim=11, target_cal_dim=30, hidden_dim=64):
        super(PriceGRUMLP, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + target_cal_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )
        
    def forward(self, x_seq, x_target_cal):
        # GRU encode
        _, h_n = self.gru(x_seq) # h_n shape: (1, batch, hidden_dim)
        h_n = h_n.squeeze(0) # shape: (batch, hidden_dim)
        
        # Concatenate with target calendar features
        concat = torch.cat([h_n, x_target_cal], dim=-1)
        
        # Predict 3 steps
        out = self.mlp(concat)
        return out

# 7. Model Training Loop
models = {}
val_nmaes = {}

for item in items_conditions.keys():
    print(f"\n--- Training Model for: {item} ---")
    prices = train_series[item]
    seasonal_mean = train_seasonal_means[item]
    
    # Train / Val splits based on start indices
    # Total periods = 144
    # Sliding window needs 12 steps (9 input + 3 target)
    # Train indices: 0 to 110 (targets fall within first 122 periods)
    # Val indices: 111 to 132 (targets fall within final 24 periods of training)
    train_indices = list(range(110))
    val_indices = list(range(110, 133))
    
    train_dataset = SequenceDataset(prices, seasonal_mean, train_indices)
    val_dataset = SequenceDataset(prices, seasonal_mean, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Instantiate Model
    # input_dim: 1 (scaled price) + 1 (scaled seasonal mean) + 9 (calendar features) = 11
    # target_cal_dim: 3 * [1 (scaled seasonal mean) + 9 (calendar)] = 30
    model = PriceGRUMLP(input_dim=11, target_cal_dim=30, hidden_dim=64).to(device)
    criterion = nn.L1Loss() # MAE Loss
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)
    
    # Training Loop
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(120):
        model.train()
        train_loss = 0.0
        for x_seq, x_target_cal, y_target, scale in train_loader:
            x_seq, x_target_cal, y_target = x_seq.to(device), x_target_cal.to(device), y_target.to(device)
            
            optimizer.zero_grad()
            pred = model(x_seq, x_target_cal)
            loss = criterion(pred, y_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_seq.size(0)
            
        train_loss /= len(train_dataset)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_seq, x_target_cal, y_target, scale in val_loader:
                x_seq, x_target_cal, y_target = x_seq.to(device), x_target_cal.to(device), y_target.to(device)
                pred = model(x_seq, x_target_cal)
                loss = criterion(pred, y_target)
                val_loss += loss.item() * x_seq.size(0)
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    # Load best weights
    model.load_state_dict(best_model_state)
    models[item] = model
    
    # Calculate NMAE on Validation set
    model.eval()
    abs_errors = []
    actual_values = []
    with torch.no_grad():
        for x_seq, x_target_cal, y_target, scale in val_loader:
            x_seq, x_target_cal = x_seq.to(device), x_target_cal.to(device)
            pred_scaled = model(x_seq, x_target_cal).cpu().numpy()
            y_actual = (y_target.numpy() * scale.numpy()[:, None])
            pred_actual = (pred_scaled * scale.numpy()[:, None])
            
            abs_errors.extend(np.abs(y_actual - pred_actual).flatten())
            actual_values.extend(y_actual.flatten())
            
    val_nmae = np.sum(abs_errors) / np.sum(actual_values)
    val_nmaes[item] = val_nmae
    print(f"Validation NMAE for {item}: {val_nmae:.4f}")

print("\n=== Mean Validation NMAE ===")
mean_nmae = np.mean(list(val_nmaes.values()))
print(f"Mean NMAE: {mean_nmae:.4f}")

# 8. Inference on TEST sets
print("\nStarting inference on TEST datasets...")
predictions = {} # test_idx -> {item: [pred_t1, pred_t2, pred_t3]}

for test_idx in range(25):
    predictions[test_idx] = {}
    
    # Load TEST_xx.csv
    test_file_path = os.path.join(test_dir, f"TEST_{test_idx:02d}.csv")
    test_df = pd.read_csv(test_file_path, encoding='utf-8-sig')
    
    # Get T calendar date mapping
    m_t, soon_t = test_t_mappings[test_idx]
    # flat index for T in 2022
    t_flat_idx = (2022 - 2018) * 36 + (m_t - 1) * 3 + (soon_t - 1)
    
    for item in items_conditions.keys():
        # Filter test data for the item
        cond = items_conditions[item]
        sub_test = cond(test_df).copy()
        
        # We need a sequence of length 9 corresponding to T-8 to T
        # Match T-8 to T to chronological steps
        steps = [f"T-{i}순" for i in range(8, 0, -1)] + ["T"]
        
        # Group by 시점 and take max (to handle apple or duplicates)
        grouped_test = sub_test.groupby('시점')['평균가격(원)'].max().reset_index()
        grouped_test = grouped_test.set_index('시점').reindex(steps)
        
        # Clean price sequence
        test_prices = grouped_test['평균가격(원)'].replace(0, np.nan).values
        mask = np.isnan(test_prices)
        if mask.all():
            # If all are missing, backfill with overall seasonal mean or training mean
            test_prices = np.array([train_seasonal_means[item][(t_flat_idx - 8 + i) % 36] for i in range(9)])
        else:
            idx = np.where(~mask)[0]
            test_prices[:idx[0]] = test_prices[idx[0]]
            test_prices[idx[-1]:] = test_prices[idx[-1]]
            for i in range(len(idx) - 1):
                test_prices[idx[i]:idx[i+1]] = test_prices[idx[i]]
                
        # Scale factor: price at T (last step)
        scale = test_prices[-1]
        if scale == 0:
            scale = np.mean(train_series[item]) if np.mean(train_series[item]) > 0 else 1.0
            
        # Build features for T-8 to T (9 steps)
        in_features = []
        for i in range(9):
            step_idx = t_flat_idx - 8 + i
            yr, m, sn, seas = flat_idx_to_date(step_idx)
            cal_feats = get_calendar_features(yr, m, sn, seas)
            
            p_scaled = test_prices[i] / scale
            sm_scaled = train_seasonal_means[item][step_idx % 36] / scale
            
            in_features.append([p_scaled, sm_scaled] + cal_feats)
            
        # Build calendar features for T+1, T+2, T+3
        target_cal_features = []
        for i in range(3):
            step_idx = t_flat_idx + 1 + i
            yr, m, sn, seas = flat_idx_to_date(step_idx)
            cal_feats = get_calendar_features(yr, m, sn, seas)
            sm_scaled = train_seasonal_means[item][step_idx % 36] / scale
            target_cal_features.extend([sm_scaled] + cal_feats)
            
        # Predict using model
        model = models[item]
        model.eval()
        
        x_seq_tensor = torch.tensor([in_features], dtype=torch.float32).to(device)
        x_target_cal_tensor = torch.tensor([target_cal_features], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_scaled = model(x_seq_tensor, x_target_cal_tensor).cpu().numpy()[0]
            
        # Denormalize and clip to non-negative
        pred_actual = pred_scaled * scale
        pred_actual = np.clip(pred_actual, 0.0, None)
        
        predictions[test_idx][item] = pred_actual

# 9. Format submission.csv
print("\nGenerating submission.csv...")
sub_df = pd.read_csv(sample_submission_path, encoding='utf-8-sig')

# Map of items to columns in submission
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
    # 시점 format is "TEST_xx+y순" (e.g. "TEST_00+1순")
    sijum = row['시점']
    test_idx = int(sijum.split('+')[0].split('_')[1])
    horizon = int(sijum.split('+')[1][0]) # 1, 2, or 3
    
    for item, col in item_col_map.items():
        pred_val = predictions[test_idx][item][horizon - 1]
        sub_df.at[i, col] = pred_val

# Save submission.csv
sub_df.to_csv(submission_path, index=False, encoding='utf-8-sig')
print("Successfully generated submission.csv!")
