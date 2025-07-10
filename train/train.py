import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataloader.dataset import create_dataloader
from models.wav2vec2 import SERModel
from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2FeatureExtractor

def compute_class_weights(labels, num_classes):
    """Tính class weights cho loss function"""
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

def train(
    data_dir,
    csv_file,
    num_epochs=30,
    batch_size=16,
    lr=2e-5,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Tạo dataloader
    dataloader, dataset = create_dataloader(
        data_dir=data_dir,
        csv_file=csv_file,
        batch_size=batch_size,
        shuffle=True,
        sr=16000,
        max_length=3,
    )

    # Lấy toàn bộ label để tính class weights
    all_labels = []
    for _, label in dataset:
        all_labels.append(label)
    num_classes = len(np.unique(all_labels))
    class_weights = compute_class_weights(np.array(all_labels), num_classes).to(device)

    # Khởi tạo model Wav2Vec2
    w2v_config = "/media/admin123/DataVoice/ckpt"
    checkpoint_path = "/media/admin123/DataVoice/ckpt/0-best.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    w2v_config = Wav2Vec2Config.from_pretrained(w2v_config)
    w2v_model = Wav2Vec2Model(w2v_config)
    model = SERModel(wav2vec_model=w2v_model)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    # Loss và optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Train loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = batch
            # Nếu inputs là dict (feature_extractor trả về), lấy input_values
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
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")

    # Lưu model cuối cùng
    torch.save(model.state_dict(), "emotion_model_final_wav2vec2.pth")
    print("✅ Đã lưu model emotion_model_final_wav2vec2.pth")

if __name__ == "__main__":
    # Thay đổi đường dẫn cho phù hợp
    data_dir = "/media/admin123/DataVoice/SER/audio/audio_2500"
    csv_file = "/media/admin123/DataVoice/train.csv"
    train(
        data_dir=data_dir,
        csv_file=csv_file,
        num_epochs=10,
        batch_size=16,
        lr=2e-5
    )
