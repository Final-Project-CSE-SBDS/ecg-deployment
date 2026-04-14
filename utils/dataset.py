import os
import wfdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

class ECGDataset(Dataset):
    def __init__(self, X, y):
        # We add an extra dimension to represent the single channel: (Batch, SequenceLength, Channels)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def download_and_prepare_mitbih(data_dir='./data', window_size=187, max_records=None):
    """
    Downloads MIT-BIH dataset via wfdb, extracts heartbeats based on R-peaks,
    normalizes them, and formats them for the Mamba deep learning model.
    Classifies into 5 fundamental AAMI categories.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print("Fetching record list for MIT-BIH dataset...")
    record_list = wfdb.get_record_list('mitdb')
    
    if max_records:
        record_list = record_list[:max_records]
        print(f"Limiting to first {max_records} records for quick demonstration.")
    
    print(f"Processing {len(record_list)} records from MIT-BIH dataset...")
    
    # AAMI Standard Classification Mapping
    aami_mapping = {
        'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # 0: Normal
        'A': 1, 'a': 1, 'J': 1, 'S': 1,          # 1: Arrhythmia
        'V': 1, 'E': 1,                          # 1: Arrhythmia
        'F': 1,                                  # 1: Arrhythmia
        '/': 1, 'f': 1, 'Q': 1                   # 1: Arrhythmia
    }
    
    X, y = [], []
    valid_symbols = list(aami_mapping.keys())
    
    # We want a 187 length fixed sequence input (90 points before R-peak, 97 after)
    pre_samples = 90
    post_samples = 97
    
    for record_name in record_list:
        try:
            record_path = os.path.join(data_dir, record_name)
            # Download record if it doesn't exist locally
            if not os.path.exists(record_path + '.dat'):
                wfdb.dl_database('mitdb', data_dir, records=[record_name])
            
            # Read signal and annotations
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            # Use channel 0 (usually MLII lead, excellent for arrhythmia detection)
            signal = record.p_signal[:, 0]
            
            for peak, sym in zip(annotation.sample, annotation.symbol):
                if sym in valid_symbols:
                    # Ensure the window fits within the signal
                    if peak >= pre_samples and peak + post_samples < len(signal):
                        beat = signal[peak - pre_samples : peak + post_samples]
                        
                        if len(beat) == window_size:
                            # Min-Max Normalization per beat to reduce patient variance
                            b_min, b_max = np.min(beat), np.max(beat)
                            if b_max - b_min > 0:
                                beat = (beat - b_min) / (b_max - b_min)
                                
                            X.append(beat)
                            y.append(aami_mapping[sym])
                            
        except Exception as e:
            print(f"Could not fully process record {record_name}: {e}")
            continue

    X = np.array(X)
    y = np.array(y)
    
    print(f"Total samples successfully extracted: {len(X)}")
    return X, y

def get_dataloaders(data_dir='./data', batch_size=64, test_size=0.2, max_records=None):
    """
    Returns PyTorch DataLoaders for Training and Testing.
    """
    X, y = download_and_prepare_mitbih(data_dir=data_dir, max_records=max_records)
    
    if len(X) == 0:
        raise ValueError("Dataset is empty. Ensure you have an internet connection for downloading MIT-BIH.")
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Calculate class weights for highly unbalanced MIT-BIH dataset
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tsr = torch.tensor(class_weights, dtype=torch.float32)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    print(f"Class Weights [Normal, Arrhythmia]: {class_weights_tsr.tolist()}")
    
    train_dataset = ECGDataset(X_train, y_train)
    test_dataset = ECGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, class_weights_tsr
