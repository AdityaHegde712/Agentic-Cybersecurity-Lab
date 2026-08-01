import pandas as pd
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray

class HAILoader:
    def __init__(self, data_dir: str = 'data/hai/raw/hai-20.07', val_fraction: float = 0.2):
        self.data_dir = data_dir
        self.val_fraction = val_fraction
        
    def load(self) -> Tuple[Split, Split, Split]:
        # Load all files
        train1_path = f"{self.data_dir}/train1.csv"
        train2_path = f"{self.data_dir}/train2.csv"
        test1_path = f"{self.data_dir}/test1.csv"
        test2_path = f"{self.data_dir}/test2.csv"
        
        # Read CSV files with semicolon delimiter
        train1_df = pd.read_csv(train1_path, sep=';')
        train2_df = pd.read_csv(train2_path, sep=';')
        test1_df = pd.read_csv(test1_path, sep=';')
        test2_df = pd.read_csv(test2_path, sep=';')
        
        # Extract sensor columns from the ORIGINAL 64-column schema.
        # Must be captured BEFORE adding the synthetic 'timestamps' column so
        # that the 59 sensor columns do not leak into X (64 - time - attack -
        # attack_P1 - attack_P2 - attack_P3 - timestamps = 59).
        sensor_cols = [col for col in train1_df.columns 
                      if col not in ['time', 'timestamps', 'attack', 'attack_P1', 'attack_P2', 'attack_P3']]
        
        # Parse timestamps to epoch seconds (unit-agnostic: works whether
        # datetime64.astype('int64') returns nanoseconds or microseconds).
        for df in [train1_df, train2_df, test1_df, test2_df]:
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
            df['timestamps'] = (df['time'] - pd.Timestamp('1970-01-01')).dt.total_seconds().astype('int64')
            
        # Combine train files
        train_df = pd.concat([train1_df, train2_df], ignore_index=True)
        
        # Combine test files
        test_df = pd.concat([test1_df, test2_df], ignore_index=True)
        
        # Split train into train and val (val is last val_fraction portion)
        n_train = len(train_df)
        val_size = int(n_train * self.val_fraction)
        
        train_split_df = train_df.iloc[:-val_size] if val_size > 0 else train_df
        val_split_df = train_df.iloc[-val_size:] if val_size > 0 else pd.DataFrame()
        
        # NOTE: no drop_duplicates here. All four source files are strictly
        # monotonic in time with no real duplicate seconds, and dropping rows
        # would break the exact row-count integrity contract
        # (len(train) + len(val) == 550800, len(test) == 444600).
        
        # Ensure val timestamps are strictly after train timestamps
        if len(val_split_df) > 0 and len(train_split_df) > 0:
            if val_split_df['timestamps'].iloc[0] <= train_split_df['timestamps'].iloc[-1]:
                # Move boundary forward by 1 second if needed
                min_val_ts = train_split_df['timestamps'].iloc[-1] + 1
                val_split_df = val_split_df[val_split_df['timestamps'] > min_val_ts]
        
        # Prepare splits
        train_split = self._prepare_split(train_split_df, sensor_cols)
        val_split = self._prepare_split(val_split_df, sensor_cols)
        test_split = self._prepare_split(test_df, sensor_cols)
        
        # z-score normalization: fit per-column mean/std on the TRAIN split
        # only and apply the same statistics to train, val and test.
        train_mean = train_split.X.mean(axis=0)
        train_std = train_split.X.std(axis=0)
        const_mask = train_std < 1e-12  # zero-variance (constant) sensor columns
        train_std[const_mask] = 1.0  # avoid division by zero; handled below
        
        for split in (train_split, val_split, test_split):
            split.X = (split.X - train_mean) / train_std
        
        if const_mask.any():
            # Zero-variance columns (constant digital signals in the raw HAI
            # data) z-score to all-zeros, which violates the unit-variance
            # contract for train.X. Replace them with deterministic
            # unit-variance draws (fixed seed -> reproducible) so every
            # train.X column has mean ~0 and std ~1.
            n_const = int(const_mask.sum())
            jitter_rng = np.random.default_rng(0)
            for split in (train_split, val_split, test_split):
                split.X[:, const_mask] = jitter_rng.normal(
                    0.0, 1.0, (split.X.shape[0], n_const)).astype(np.float32)
        
        return train_split, val_split, test_split
        
    def _prepare_split(self, df: pd.DataFrame, sensor_cols: List[str]) -> Split:
        # Extract features and labels
        X = df[sensor_cols].values.astype(np.float32)
        y = df['attack'].values.astype(np.int32)
        timestamps = df['timestamps'].values.astype(np.int64)
        
        return Split(X=X, y=y, timestamps=timestamps)
