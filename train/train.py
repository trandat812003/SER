import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from dataloader.dataset import create_dataloader
from models.hubert_xlsr import SERModelHuBERT
from transformers import Wav2Vec2FeatureExtractor

def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

def train(
    train_data_dir,
    train_csv_file,
    valid_data_dir,
    valid_csv_file,
    num_epochs=30,
    batch_size=16,
    lr=2e-5,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Train dataloader
    train_loader, train_dataset = create_dataloader(
        data_dir=train_data_dir,
        csv_file=train_csv_file,
        batch_size=batch_size,
        shuffle=True,
        sr=16000,
        max_length=3,
    )
    # Valid dataloader
    valid_loader, valid_dataset = create_dataloader(
        data_dir=valid_data_dir,
        csv_file=valid_csv_file,
        batch_size=batch_size,
        shuffle=False,
        sr=16000,
        max_length=3,
    )

    # Lấy toàn bộ label để tính class weights
    all_labels = [label for _, label in train_dataset]
    num_classes = len(np.unique(all_labels))
    class_weights = compute_class_weights(np.array(all_labels), num_classes).to(device)

    checkpoint_path = "/home/jovyan/datnt/stream_ser/hubert_pretrain.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SERModelHuBERT()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = batch
            if isinstance(inputs, dict):
                inputs = inputs["input_values"]
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1} [Train]: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]"):
                inputs, labels = batch
                if isinstance(inputs, dict):
                    inputs = inputs["input_values"]
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_total += labels.size(0)
        val_loss = val_loss / val_total
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"Epoch {epoch+1} [Valid]: Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

        torch.save(model.state_dict(), f"emotion_model_epoch{epoch+1}.pth")
        print(f"✅ Đã lưu checkpoint emotion_model_epoch{epoch+1}.pth")

if __name__ == "__main__":
    # Thay đổi đường dẫn cho phù hợp
    train_data_dir = "/media/admin123/DataVoice/SER/audio/audio_2500"
    train_csv_file = "/media/admin123/DataVoice/train.csv"
    valid_data_dir = "/media/admin123/DataVoice/SER/audio/audio_valid"
    valid_csv_file = "/media/admin123/DataVoice/valid.csv"
    train(
        train_data_dir=train_data_dir,
        train_csv_file=train_csv_file,
        valid_data_dir=valid_data_dir,
        valid_csv_file=valid_csv_file,
        num_epochs=30,
        batch_size=16,
        lr=2e-5
    )
