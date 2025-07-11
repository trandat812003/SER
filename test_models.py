#!/usr/bin/env python3
"""
Script test để đánh giá các models và tìm checkpoint có accuracy và F1 score cao nhất
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import glob
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Config, Wav2Vec2Model

from models.wav2vec2_xlsr import SERModelWav2Vec2
from models.hubert_xlsr import SERModelHuBERT
from models.wav2vec2 import SERModel

from util.mapping import mapping_pleasure_output, mapping_arousal_output, detect_unsatisfied


class ModelTester:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = LabelEncoder()
        print(f"Sử dụng device: {self.device}")
        
    def load_model(self, model_name, checkpoint_path=None):
        """Load model với checkpoint"""
        if model_name == "wav2vec2_xlsr":
            model = SERModelWav2Vec2()
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        elif model_name == "hubert_xlsr":
            model = SERModelHuBERT()
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        elif model_name == "wav2vec2":
            w2v_config = "/media/admin123/DataVoice/ckpt"
            w2v_config = Wav2Vec2Config.from_pretrained(w2v_config)
            w2v_model = Wav2Vec2Model(w2v_config)
            model = SERModel(wav2vec_model=w2v_model)
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=False,
            )
        else:
            raise ValueError(f"Không hỗ trợ model: {model_name}")
        
        # Load checkpoint nếu có
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model, feature_extractor
    
    def load_audio(self, audio_path, feature_extractor):
        """Load và preprocess audio"""
        try:
            audio, sr = sf.read("/media/admin123/DataVoice/SER/audio/audio_2500/"+audio_path)
            if sr != 16000:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
            
            # Normalize audio
            audio = audio.astype(np.float32)
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # Lấy channel đầu tiên nếu stereo
            
            # Extract features
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            input_values = inputs["input_values"].to(self.device)
            
            return input_values
        except Exception as e:
            print(f"Lỗi khi load audio {audio_path}: {e}")
            return None
    
    def predict_audio(self, model, audio_path, feature_extractor):
        """Dự đoán cảm xúc cho một file audio"""
        input_values = self.load_audio(audio_path, feature_extractor)
        if input_values is None:
            return None
        
        with torch.no_grad():
            logits = model(input_values)
            pred = torch.argmax(logits, dim=-1).item()
        
        return pred
    
    def evaluate_model(self, model, test_data, feature_extractor, model_name, checkpoint_name=""):
        """Đánh giá model trên test data"""
        print(f"\nĐang đánh giá {model_name} {checkpoint_name}...")
        
        predictions = []
        true_labels = []
        
        for idx, row in test_data.iterrows():
            audio_path = row['audio_path']
            true_label = row['emotion']

            # breakpoint()
            
            pred = self.predict_audio(model, audio_path, feature_extractor)
            if pred is not None:
                predictions.append(pred)
                true_labels.append(true_label)
        
        if len(predictions) == 0:
            print("Không có predictions nào!")
            return None
        
        # Tính metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # In classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        
        return {
            'model_name': model_name,
            'checkpoint_name': checkpoint_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'num_samples': len(predictions)
        }
    
    def find_best_checkpoint(self, model_name, checkpoint_dir, test_data):
        """Tìm checkpoint tốt nhất"""
        print(f"\nTìm checkpoint tốt nhất cho {model_name}...")
        
        # Tìm tất cả checkpoint files (không cần pattern, lấy hết file .pth, .pt)
        checkpoints = []
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pth') or file.endswith('.pt'):
                checkpoints.append(os.path.join(checkpoint_dir, file))
        
        if not checkpoints:
            print(f"Không tìm thấy checkpoint nào trong {checkpoint_dir}")
            return None
        
        print(f"Tìm thấy {len(checkpoints)} checkpoints:")
        for ckpt in checkpoints:
            print(f"  - {os.path.basename(ckpt)}")
        
        results = []
        
        for checkpoint_path in checkpoints:
            try:
                checkpoint_name = os.path.basename(checkpoint_path)
                print(f"\nTesting checkpoint: {checkpoint_name}")
                
                model, feature_extractor = self.load_model(model_name, checkpoint_path)
                result = self.evaluate_model(model, test_data, feature_extractor, model_name, checkpoint_name)
                
                if result:
                    results.append(result)
                    
            except Exception as e:
                print(f"Lỗi khi test checkpoint {checkpoint_path}: {e}")
                continue
        
        if not results:
            print("Không có kết quả nào!")
            return None
        
        # Tìm checkpoint tốt nhất
        best_acc = max(results, key=lambda x: x['accuracy'])
        best_f1 = max(results, key=lambda x: x['f1_score'])
        
        print(f"\n{'='*60}")
        print("KẾT QUẢ TỐT NHẤT:")
        print(f"{'='*60}")
        print(f"Checkpoint tốt nhất theo Accuracy: {best_acc['checkpoint_name']}")
        print(f"  Accuracy: {best_acc['accuracy']:.4f}")
        print(f"Checkpoint tốt nhất theo F1 Score: {best_f1['checkpoint_name']}")
        print(f"  F1 Score: {best_f1['f1_score']:.4f}")
        
        # In tất cả kết quả
        print(f"\n{'='*60}")
        print("TẤT CẢ KẾT QUẢ:")
        print(f"{'='*60}")
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('f1_score', ascending=False)
        print(df_results.to_string(index=False))
        
        return df_results


def load_test_data(data_path):
    """Load test data"""
    if not os.path.exists(data_path):
        print(f"File {data_path} không tồn tại!")
        return None
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Kiểm tra columns
    required_cols = ['audio', 'pleasure', 'arousal']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Thiếu columns: {missing_cols}")
        return None

    # Mapping thành nhãn cuối cùng (ví dụ dùng mapping từ util/mapping.py)
    df['pleasure_output'] = df['pleasure'].apply(mapping_pleasure_output)
    df['arousal_output'] = df['arousal'].apply(mapping_arousal_output)
    df['emotion'] = df.apply(lambda row: detect_unsatisfied(row['pleasure_output'], row['arousal_output']), axis=1)
    df = df.rename(columns={'audio': 'audio_path'})
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Test models và tìm checkpoint tốt nhất")
    parser.add_argument("--model_name", type=str, default="wav2vec2_xlsr",
                       choices=["wav2vec2_xlsr", "hubert_xlsr", "wav2vec2"],
                       help="Tên model để test")
    parser.add_argument("--checkpoint_dir", type=str, default="/media/admin123/DataVoice/ckpt",
                       help="Thư mục chứa checkpoints")
    parser.add_argument("--test_data", type=str, default="/media/admin123/DataVoice/valid.csv",
                       help="File CSV chứa test data")
    parser.add_argument("--output_file", type=str, default="test_results.csv",
                       help="File output cho kết quả")
    
    args = parser.parse_args()
    
    # Load test data
    test_data = load_test_data(args.test_data)
    if test_data is None:
        return
    
    # Khởi tạo tester
    tester = ModelTester()
    
    # Test model
    results = tester.find_best_checkpoint(args.model_name, args.checkpoint_dir, test_data)
    
    if results is not None:
        # Lưu kết quả
        results.to_csv(args.output_file, index=False)
        print(f"\nKết quả đã được lưu vào: {args.output_file}")
    
    print("\n✅ Test hoàn thành!")


if __name__ == "__main__":
    main() 