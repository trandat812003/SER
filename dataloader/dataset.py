import os
import json
import torch
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor

from util.mapping import (
    mapping_pleasure_output,
    mapping_arousal_output,
    detect_unsatisfied,
)


class EmotionDataset(Dataset):
    def __init__(
        self,
        data_dir,
        csv_file=None,
        sr=16000,
        max_length=5,
        feature_extractor=None,
        feature_extractor_kwargs=None,
    ):
        self.data_dir = data_dir
        self.sr = sr
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.sr,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )

        if csv_file and os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
            self.audio_files = []
            self.labels = []

            if csv_file.endswith(".csv"):
                self.df = pd.read_csv(csv_file)
        
                for _, row in self.df.iterrows():
                    file_path = os.path.join(data_dir, row["audio"])
                    if os.path.exists(file_path):
                        self.audio_files.append(self.preprocess_audio(file_path))
        
                        p_label = mapping_pleasure_output(row["pleasure"])
                        a_label = mapping_arousal_output(row["arousal"])
                        label = detect_unsatisfied(p_label, a_label)
                        self.labels.append(label)
        
            elif csv_file.endswith(".txt"):
                self.df = pd.read_csv(csv_file, sep="\t", header=None, names=["filename", "label"])
        
                for _, row in self.df.iterrows():
                    file_path = os.path.join(data_dir, row["filename"])
                    if os.path.exists(file_path):
                        self.audio_files.append(self.preprocess_audio(file_path))
        
                        # Convert label: negative -> 1, neutral -> 0
                        if row["label"].strip().lower() == "negative":
                            label = 1
                        elif row["label"].strip().lower() == "neutral":
                            label = 0
                        else:
                            label = -1  # hoặc raise error tùy use case
        
                        self.labels.append(label)
            else:
                raise ValueError("Unsupported file type. Please provide .csv or .txt file.")
        else:
            raise ValueError("CSV file not found")

    def preprocess_audio(self, file_path):
        try:
            audio, orig_sr = librosa.load(file_path, sr=self.sr, mono=True)

            target_length = self.sr * self.max_length
            if len(audio) > target_length:
                audio_tensor = audio[:target_length]
            else:
                padding = target_length - len(audio)
                audio_tensor = np.pad(audio, (0, padding), "constant")

        except Exception as e:
            print(f"Lỗi load file {file_path}: {e}")
            audio_tensor = torch.zeros(self.sr * self.max_length)

        if self.feature_extractor is not None:
            audio_tensor = self.feature_extractor(audio_tensor, sampling_rate=16000, return_tensors="pt")
            audio_tensor = audio_tensor["input_values"].to(self.device)

        return audio_tensor

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_tensor = self.audio_files[idx]
        if audio_tensor.shape[1] > self.sr * self.max_length:
            audio_tensor = audio_tensor[:, : self.sr * self.max_length]
        else:
            padding = self.sr * self.max_length - audio_tensor.shape[1]
            audio_tensor = torch.cat(
                [audio_tensor, torch.zeros((audio_tensor.shape[0], padding), device=audio_tensor.device)],
                dim=1
            )
        label = self.labels[idx]
        audio_tensor = audio_tensor.squeeze(0)

        return audio_tensor, label


def create_dataloader(
    data_dir,
    csv_file=None,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    sr=16000,
    max_length=5,
):
    """
    Tạo DataLoader cho emotion classification

    Args:
        data_dir: Thư mục chứa file audio
        csv_file: File CSV chứa metadata (optional)
        batch_size: Kích thước batch
        shuffle: Có shuffle data không
        num_workers: Số worker cho DataLoader
        sr: Sample rate
        max_length: Độ dài tối đa của audio (giây)
    """
    dataset = EmotionDataset(data_dir, csv_file, sr, max_length)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader, dataset


if __name__ == "__main__":
    data_dir = "/home/jovyan/datnt/dataset3/wav"
    csv_file = "/home/jovyan/datnt/dataset3/ser.txt"

    dataloader, dataset = create_dataloader(
        data_dir=data_dir,
        csv_file=csv_file,
        batch_size=16,
        shuffle=True,
        sr=16000,
        max_length=5,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    for batch_idx, (audio, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Labels: {labels}")
        print(f"  Label distribution: {torch.bincount(labels)}")
        break
