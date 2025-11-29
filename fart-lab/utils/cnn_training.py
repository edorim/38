import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import random
import numpy as np
import warnings

# --- Global Labels Definition ---
LABELS = ['wet', 'squeaky', 'bass', 'ghost', 'not_fart']

# --- 1. CNN Model Definition ---
class FartClassifierCNN(nn.Module):
    def __init__(self, num_classes=len(LABELS)):
        super(FartClassifierCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# --- 2. Custom Audio Dataset ---
class FartAudioDataset(Dataset):
    def __init__(self, audio_dir, labels, sr=16000, n_mels=128, max_len_samples=16000):
        self.audio_dir = audio_dir
        self.labels_map = {label: i for i, label in enumerate(labels)}
        self.filepaths_labels = self._load_filepaths_labels(audio_dir, labels)
        self.sr = sr
        self.n_mels = n_mels
        self.max_len_samples = max_len_samples
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=400, hop_length=160)
        self.amplitude_to_db = T.AmplitudeToDB()

    def _load_filepaths_labels(self, audio_dir, labels):
        filepaths_labels = []
        for kind in labels:
            if kind == "not_fart":
                rejected_dir = os.path.join(audio_dir, '../rejected')
                if os.path.exists(rejected_dir):
                    for fname in os.listdir(rejected_dir):
                        if fname.endswith('.wav'):
                            filepaths_labels.append((os.path.join(rejected_dir, fname), "not_fart"))
                else:
                    warnings.warn(f"'rejected' directory not found at {rejected_dir}. Generating dummy 'not_fart' samples.")
                    num_dummy_not_farts = 50
                    for _ in range(num_dummy_not_farts):
                        filepaths_labels.append(("dummy_path_for_not_fart.wav", "not_fart"))

            else:
                kind_dir = os.path.join(audio_dir, kind)
                if not os.path.exists(kind_dir):
                    warnings.warn(f"Directory for '{kind}' not found at {kind_dir}. No samples will be loaded for this label.")
                    continue
                for fname in os.listdir(kind_dir):
                    if fname.endswith('.wav'):
                        filepaths_labels.append((os.path.join(kind_dir, fname), kind))
        return filepaths_labels

    def __len__(self):
        return len(self.filepaths_labels)

    def __getitem__(self, idx):
        audio_path, label_str = self.filepaths_labels[idx]
        class_id = self.labels_map[label_str]

        waveform = None
        sr = self.sr

        if label_str == "not_fart" and "dummy_path" in audio_path:
            waveform = torch.randn(1, self.max_len_samples)
        else:
            try:
                waveform, sr = torchaudio.load(audio_path)
            except Exception as e:
                warnings.warn(f"Error loading {audio_path}: {e}. Returning silent audio.")
                waveform = torch.zeros(1, self.max_len_samples)

            if sr != self.sr:
                resampler = T.Resample(orig_freq=sr, new_freq=self.sr)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if waveform.shape[1] < self.max_len_samples:
                padding = self.max_len_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif waveform.shape[1] > self.max_len_samples:
                waveform = waveform[:, :self.max_len_samples]

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

        return mel_spec_db, class_id

# --- 3. Data Loader Creation Function ---
def create_dataloaders(dataset, batch_size=32, val_split=0.2):
    if len(dataset) == 0:
        warnings.warn("Dataset is empty, cannot create dataloaders.")
        return None, None

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

# --- 4. Training Function ---
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, model_save_path="fart_cnn_model.pt"):
    if train_loader is None or val_loader is None:
        print("Training skipped: No data loaders available.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training on {device}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("Finished Training")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# --- 5. Live Fine-Tuning Function ---
def fine_tune_cnn_model(existing_model_path: str, new_data_dir: str, num_epochs_finetune: int = 3, model_save_path: str = "fart_cnn_model.pt", batch_size: int = 16):
    print(f"[Fine-tuning] Starting fine-tuning process with data from: {new_data_dir}")

    model = FartClassifierCNN(num_classes=len(LABELS))

    if os.path.exists(existing_model_path):
        model.load_state_dict(torch.load(existing_model_path, map_location=torch.device('cpu')))
        print(f"[Fine-tuning] Loaded existing model weights from {existing_model_path}")
    else:
        warnings.warn(f"[Fine-tuning] No existing model found at {existing_model_path}. Training from scratch instead of fine-tuning.")

    for label in LABELS:
        if label != 'not_fart':
            os.makedirs(os.path.join(new_data_dir, label), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(new_data_dir), 'rejected'), exist_ok=True)

    fine_tune_dataset = FartAudioDataset(audio_dir=new_data_dir, labels=LABELS)
    fine_tune_train_loader, fine_tune_val_loader = create_dataloaders(fine_tune_dataset, batch_size=batch_size)

    if fine_tune_train_loader is None or len(fine_tune_train_loader.dataset) == 0:
        print("[Fine-tuning] No new data found for fine-tuning. Skipping fine-tuning.")
        return

    print(f"[Fine-tuning] Fine-tuning model for {num_epochs_finetune} epochs on new data.")
    train_model(model, fine_tune_train_loader, fine_tune_val_loader,
                num_epochs=num_epochs_finetune, learning_rate=0.0001,
                model_save_path=model_save_path)

    print(f"[Fine-tuning] Fine-tuning complete. Updated model saved to {model_save_path}")

if __name__ == "__main__":
    print("Running initial CNN model training...")

    DATA_ROOT = "./fart_dataset_for_cnn"
    os.makedirs(DATA_ROOT, exist_ok=True)

    REJECTED_DIR = os.path.join(os.path.dirname(DATA_ROOT), 'rejected')
    os.makedirs(REJECTED_DIR, exist_ok=True)

    for label in LABELS:
        if label != 'not_fart':
            os.makedirs(os.path.join(DATA_ROOT, label), exist_ok=True)

    print("Generating dummy audio files for initial demonstration...")
    dummy_sr = 16000
    dummy_duration = 1
    for label in LABELS:
        if label == 'not_fart':
            target_dir = REJECTED_DIR
        else:
            target_dir = os.path.join(DATA_ROOT, label)

        num_dummy_files = 20
        for i in range(num_dummy_files):
            t = np.linspace(0, dummy_duration, int(dummy_sr * dummy_duration), endpoint=False)
            if label == 'wet':
                dummy_waveform = 0.3 * np.sin(2 * np.pi * random.uniform(50, 150) * t) + 0.1 * np.random.normal(0, 0.1, t.shape)
            elif label == 'squeaky':
                dummy_waveform = 0.2 * np.sin(2 * np.pi * random.uniform(800, 1500) * t)
            elif label == 'bass':
                dummy_waveform = 0.9 * np.sin(2 * np.pi * random.uniform(30, 80) * t)
            elif label == 'ghost':
                dummy_waveform = 0.05 * np.random.normal(0, 0.01, t.shape)
            else:
                dummy_waveform = 0.1 * np.sin(2 * np.pi * random.uniform(200, 1000) * t) + 0.05 * np.random.normal(0, 0.05, t.shape)

            dummy_waveform = dummy_waveform.astype(np.float32)
            dummy_waveform = dummy_waveform / (np.max(np.abs(dummy_waveform)) + 1e-9)

            dummy_path = os.path.join(target_dir, f"{label}_dummy_{i}.wav")
            torchaudio.save(dummy_path, torch.tensor(dummy_waveform).unsqueeze(0), dummy_sr)
    print("Dummy audio files generated for initial training.")

    fart_dataset = FartAudioDataset(audio_dir=DATA_ROOT, labels=LABELS)
    train_loader, val_loader = create_dataloaders(fart_dataset, batch_size=16)
    model = FartClassifierCNN(num_classes=len(LABELS))
    train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=0.001, model_save_path="fart_cnn_model.pt")

    print("\nDemonstrating fine-tuning with the same data for simplicity...")
    fine_tune_cnn_model(
        existing_model_path="fart_cnn_model.pt",
        new_data_dir=DATA_ROOT,
        num_epochs_finetune=2,
        model_save_path="fart_cnn_model_finetuned.pt"
    )