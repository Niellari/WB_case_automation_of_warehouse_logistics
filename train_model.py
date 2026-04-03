import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ─── Load data ───
train = pd.read_parquet('data/train_team_track.parquet')
test = pd.read_parquet('data/test_team_track.parquet')

# Route -> office mapping (test doesn't have it, take from train)
route_office = train[['route_id', 'office_from_id']].drop_duplicates().set_index('route_id')['office_from_id']
test['office_from_id'] = test['route_id'].map(route_office)

print(f"Train: {train.shape}, Test: {test.shape}")

# ─── Add time columns early (needed for aggregations) ───
for df in [train, test]:
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dow'] = df['timestamp'].dt.dayofweek
    df['day'] = df['timestamp'].dt.day
    df['week'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['slot'] = df['hour'] * 2 + df['minute'] // 30

# ─── Feature engineering ───
def make_features(df, train_agg):
    df = df.copy()
    # Merge aggregated historical stats per route
    for key, agg_df in train_agg.items():
        df = df.merge(agg_df, on=list(agg_df.index.names) if isinstance(agg_df.index, pd.MultiIndex) else [agg_df.index.name], how='left')
    return df


# ─── Aggregate stats from train ───
# Per route: overall stats
route_stats = train.groupby('route_id')['target_2h'].agg(['mean', 'std', 'median']).rename(
    columns={'mean': 'route_mean', 'std': 'route_std', 'median': 'route_median'}
)

# Per route + hour
route_hour_stats = train.groupby(['route_id', 'hour'])['target_2h'].agg(['mean', 'median']).rename(
    columns={'mean': 'route_hour_mean', 'median': 'route_hour_median'}
)

# Per route + dow
route_dow_stats = train.groupby(['route_id', 'dow'])['target_2h'].agg(['mean', 'median']).rename(
    columns={'mean': 'route_dow_mean', 'median': 'route_dow_median'}
)

# Per route + dow + hour
route_dow_hour_stats = train.groupby(['route_id', 'dow', 'hour'])['target_2h'].agg(['mean']).rename(
    columns={'mean': 'route_dow_hour_mean'}
)

# Per office: overall stats
office_stats = train.groupby('office_from_id')['target_2h'].agg(['mean', 'std']).rename(
    columns={'mean': 'office_mean', 'std': 'office_std'}
)

# Per office + hour
office_hour_stats = train.groupby(['office_from_id', 'hour'])['target_2h'].agg(['mean']).rename(
    columns={'mean': 'office_hour_mean'}
)

train_agg = {
    'route': route_stats,
    'route_hour': route_hour_stats,
    'route_dow': route_dow_stats,
    'route_dow_hour': route_dow_hour_stats,
    'office': office_stats,
    'office_hour': office_hour_stats,
}

# ─── Lag features (last known values per route before cutoff) ───
# For each route, get the last N target values from train
train_sorted = train.sort_values(['route_id', 'timestamp'])

# Last values per route (these will be used for both validation and test)
def get_lag_features(df_history, df_target, n_lags=8):
    """For each row in df_target, get last n_lags target values from df_history for same route."""
    lag_data = {}
    for route_id, group in df_history.groupby('route_id'):
        vals = group.sort_values('timestamp')['target_2h'].values
        lag_data[route_id] = vals

    lag_cols = {f'lag_{i}': [] for i in range(1, n_lags + 1)}
    lag_cols['lag_mean_4'] = []
    lag_cols['lag_mean_8'] = []

    for _, row in df_target.iterrows():
        rid = row['route_id']
        vals = lag_data.get(rid, np.array([]))
        for i in range(1, n_lags + 1):
            if len(vals) >= i:
                lag_cols[f'lag_{i}'].append(vals[-i])
            else:
                lag_cols[f'lag_{i}'].append(np.nan)
        lag_cols['lag_mean_4'].append(np.mean(vals[-4:]) if len(vals) >= 4 else np.nan)
        lag_cols['lag_mean_8'].append(np.mean(vals[-8:]) if len(vals) >= 8 else np.nan)

    for col, values in lag_cols.items():
        df_target[col] = values

    return df_target


# ─── Validation: hold out last 5 hours of train (same structure as test) ───
cutoff = train['timestamp'].max() - pd.Timedelta(hours=4, minutes=30)  # last 10 steps
print(f"Validation cutoff: {cutoff}")

val_mask = train['timestamp'] > cutoff
val_df = train[val_mask].copy()
train_df = train[~val_mask].copy()

print(f"Train split: {train_df.shape}, Val split: {val_df.shape}")

# Recompute aggregates on train_df only (to avoid leakage) — time cols already exist
route_stats_tr = train_df.groupby('route_id')['target_2h'].agg(['mean', 'std', 'median']).rename(
    columns={'mean': 'route_mean', 'std': 'route_std', 'median': 'route_median'}
)
route_hour_stats_tr = train_df.groupby(['route_id', 'hour'])['target_2h'].agg(['mean', 'median']).rename(
    columns={'mean': 'route_hour_mean', 'median': 'route_hour_median'}
)
route_dow_stats_tr = train_df.groupby(['route_id', 'dow'])['target_2h'].agg(['mean', 'median']).rename(
    columns={'mean': 'route_dow_mean', 'median': 'route_dow_median'}
)
route_dow_hour_stats_tr = train_df.groupby(['route_id', 'dow', 'hour'])['target_2h'].agg(['mean']).rename(
    columns={'mean': 'route_dow_hour_mean'}
)
office_stats_tr = train_df.groupby('office_from_id')['target_2h'].agg(['mean', 'std']).rename(
    columns={'mean': 'office_mean', 'std': 'office_std'}
)
office_hour_stats_tr = train_df.groupby(['office_from_id', 'hour'])['target_2h'].agg(['mean']).rename(
    columns={'mean': 'office_hour_mean'}
)

train_agg_tr = {
    'route': route_stats_tr,
    'route_hour': route_hour_stats_tr,
    'route_dow': route_dow_stats_tr,
    'route_dow_hour': route_dow_hour_stats_tr,
    'office': office_stats_tr,
    'office_hour': office_hour_stats_tr,
}

# Build features for validation
val_df = make_features(val_df, train_agg_tr)
val_df = get_lag_features(train_df, val_df, n_lags=8)

# Build features for train
train_feat = make_features(train_df, train_agg_tr)
# For train lags — use shifted values within train (per route)
# Simpler: compute lags per route using shift
train_feat = train_feat.sort_values(['route_id', 'timestamp'])
for i in range(1, 9):
    train_feat[f'lag_{i}'] = train_feat.groupby('route_id')['target_2h'].shift(i)
train_feat['lag_mean_4'] = train_feat[[f'lag_{i}' for i in range(1, 5)]].mean(axis=1)
train_feat['lag_mean_8'] = train_feat[[f'lag_{i}' for i in range(1, 9)]].mean(axis=1)

# Drop rows with NaN lags (first 8 per route)
train_feat = train_feat.dropna(subset=['lag_8'])

feature_cols = [
    'hour', 'minute', 'dow', 'day', 'week', 'is_weekend', 'slot',
    'office_from_id', 'route_id',
    'route_mean', 'route_std', 'route_median',
    'route_hour_mean', 'route_hour_median',
    'route_dow_mean', 'route_dow_median',
    'route_dow_hour_mean',
    'office_mean', 'office_std', 'office_hour_mean',
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8',
    'lag_mean_4', 'lag_mean_8',
]

X_train = train_feat[feature_cols]
y_train = train_feat['target_2h']
X_val = val_df[feature_cols]
y_val = val_df['target_2h']

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

# ─── Train LightGBM ───
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'min_child_samples': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
}

model = lgb.train(
    params, dtrain,
    num_boost_round=2000,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

# ─── Evaluate ───
val_pred = model.predict(X_val)
val_pred = np.clip(val_pred, 0, None)  # target >= 0

wape = np.abs(val_pred - y_val.values).sum() / y_val.values.sum()
rbias = np.abs(val_pred.sum() / y_val.values.sum() - 1)
score = wape + rbias
print(f"\n=== Validation ===")
print(f"WAPE: {wape:.6f}")
print(f"|Relative Bias|: {rbias:.6f}")
print(f"Score (WAPE + |RBias|): {score:.6f}")

# ─── Feature importance ───
importance = pd.Series(model.feature_importance(importance_type='gain'), index=feature_cols)
print(f"\nTop 15 features (gain):")
print(importance.sort_values(ascending=False).head(15))

# ─── Retrain on full data and predict test ───
print("\n=== Retraining on full train data ===")

# Full train features
full_train = make_features(train, train_agg)
full_train = full_train.sort_values(['route_id', 'timestamp'])
for i in range(1, 9):
    full_train[f'lag_{i}'] = full_train.groupby('route_id')['target_2h'].shift(i)
full_train['lag_mean_4'] = full_train[[f'lag_{i}' for i in range(1, 5)]].mean(axis=1)
full_train['lag_mean_8'] = full_train[[f'lag_{i}' for i in range(1, 9)]].mean(axis=1)
full_train = full_train.dropna(subset=['lag_8'])

X_full = full_train[feature_cols]
y_full = full_train['target_2h']

dfull = lgb.Dataset(X_full, label=y_full)
best_rounds = model.best_iteration
model_final = lgb.train(params, dfull, num_boost_round=best_rounds)

# Test features
test_feat = make_features(test, train_agg)
test_feat = get_lag_features(train, test_feat, n_lags=8)

X_test = test_feat[feature_cols]
test_pred = model_final.predict(X_test)
test_pred = np.clip(test_pred, 0, None)

# ─── Save submission ───
submission = pd.DataFrame({'id': test['id'], 'y_pred': test_pred})
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head(10))
print(f"\nPred stats: mean={test_pred.mean():.2f}, std={test_pred.std():.2f}, min={test_pred.min():.2f}, max={test_pred.max():.2f}")
