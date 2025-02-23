import streamlit as st
import torch
import torchaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from PIL import Image
import torch.nn as nn

# ----------------------------
# Helper Functions for Audio Processing
# ----------------------------
def load_audio(filepath, sr=192000):
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in [".wav", ".pkf"]:
        st.error("Unsupported file type. Please upload a .wav or .pkf file.")
        return None
    if ext == ".wav":
        try:
            waveform, _ = torchaudio.load(filepath)
            waveform = waveform.numpy()[0]  # Assuming single-channel
        except Exception as e:
            st.error(f"Error loading WAV file: {e}")
            return None
    elif ext == ".pkf":
        try:
            with open(filepath, "rb") as f:
                waveform = pickle.load(f)
            waveform = np.array(waveform, dtype=np.float32)
        except Exception as e:
            st.warning(f"Pickle loading failed for {filepath}: {e}")
            try:
                with open(filepath, "rb") as f:
                    header = f.read(20)  # Skip header bytes
                    st.write(f"Fallback: File header bytes: {header}")
                    raw_data = f.read()  # Read remaining bytes
                waveform = np.frombuffer(raw_data, dtype=np.float32)
            except Exception as e2:
                st.error(f"Fallback loading failed: {e2}")
                return None
    try:
        waveform = librosa.resample(waveform, orig_sr=48000, target_sr=sr)
    except Exception as e:
        st.error(f"Error during resampling: {e}")
        return None
    return waveform

def waveform_to_logmel(waveform, sr=192000, n_mels=128, n_fft=2048, hop_length=512):
    try:
        S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft,
                                           hop_length=hop_length, n_mels=n_mels)
        log_S = librosa.power_to_db(S, ref=np.max)
    except Exception as e:
        st.error(f"Error generating spectrogram: {e}")
        return None
    return log_S

# ----------------------------
# Model Definition (must match training)
# ----------------------------
class ConvAttentionNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvAttentionNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_block(x)  # (batch, 32, H, W)
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)   # (batch, 32, H*W)
        x = x.permute(0, 2, 1)  # (batch, H*W, 32)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.mean(dim=1)  # (batch, 32)
        out = self.classifier(attn_output)
        return out

# Load model checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAttentionNet(num_classes=2).to(device)
checkpoint_path = "conv_attention_model_checkpoint.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ----------------------------
# Mapping for Species Images (update paths as needed)
# ----------------------------
species_to_image = {
    "Pipistrellus ceylonicus": "Pipistrellus ceylonicus.jpg",
    "Rhinolophus indorouxii": "Rhinolophus_indorouxii.jpg"
}

# ----------------------------
# Streamlit UI Layout with Tabs
# ----------------------------
st.title("Bat Call Classification Dashboard")

tab1, tab2 = st.tabs(["Training History", "Test Prediction"])

# Tab 1: Training History Graphs
with tab1:
    st.header("Training History Metrics")
    st.write("Upload a CSV file with your training history (columns: epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, train_precision, val_precision, train_recall, val_recall).")
    uploaded_history = st.file_uploader("Upload CSV", type="csv", key="history")
    
    if uploaded_history is not None:
        history_df = pd.read_csv(uploaded_history)
    else:
        # Sample data for demonstration
        data = {
            "epoch": [1, 2, 3, 4, 5],
            "train_loss": [0.85, 0.65, 0.55, 0.45, 0.40],
            "val_loss": [0.80, 0.70, 0.60, 0.50, 0.47],
            "train_acc": [0.70, 0.78, 0.82, 0.85, 0.87],
            "val_acc": [0.68, 0.75, 0.80, 0.83, 0.82],
            "train_f1": [0.68, 0.75, 0.80, 0.83, 0.85],
            "val_f1": [0.66, 0.73, 0.78, 0.81, 0.80],
            "train_precision": [0.70, 0.77, 0.82, 0.84, 0.86],
            "val_precision": [0.68, 0.74, 0.79, 0.81, 0.80],
            "train_recall": [0.69, 0.76, 0.81, 0.83, 0.85],
            "val_recall": [0.67, 0.73, 0.78, 0.80, 0.79]
        }
        history_df = pd.DataFrame(data)
    
    st.dataframe(history_df)
    
    # Plot Loss and Accuracy over Epochs
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history_df["epoch"], history_df["train_loss"], "bo-", label="Train Loss")
    ax[0].plot(history_df["epoch"], history_df["val_loss"], "ro-", label="Val Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss over Epochs")
    ax[0].legend()
    
    ax[1].plot(history_df["epoch"], history_df["train_acc"], "bo-", label="Train Accuracy")
    ax[1].plot(history_df["epoch"], history_df["val_acc"], "ro-", label="Val Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy over Epochs")
    ax[1].legend()
    
    st.pyplot(fig)
    
    # Plot F1, Precision, and Recall over Epochs
    fig2, ax2 = plt.subplots(1, 3, figsize=(18, 4))
    ax2[0].plot(history_df["epoch"], history_df["train_f1"], "bo-", label="Train F1")
    ax2[0].plot(history_df["epoch"], history_df["val_f1"], "ro-", label="Val F1")
    ax2[0].set_xlabel("Epoch")
    ax2[0].set_ylabel("F1 Score")
    ax2[0].set_title("F1 Score")
    ax2[0].legend()
    
    ax2[1].plot(history_df["epoch"], history_df["train_precision"], "bo-", label="Train Precision")
    ax2[1].plot(history_df["epoch"], history_df["val_precision"], "ro-", label="Val Precision")
    ax2[1].set_xlabel("Epoch")
    ax2[1].set_ylabel("Precision")
    ax2[1].set_title("Precision")
    ax2[1].legend()
    
    ax2[2].plot(history_df["epoch"], history_df["train_recall"], "bo-", label="Train Recall")
    ax2[2].plot(history_df["epoch"], history_df["val_recall"], "ro-", label="Val Recall")
    ax2[2].set_xlabel("Epoch")
    ax2[2].set_ylabel("Recall")
    ax2[2].set_title("Recall")
    ax2[2].legend()
    
    st.pyplot(fig2)

# Tab 2: Test Prediction and Species Image
with tab2:
    st.header("Test Prediction and Species Image")
    st.write("Upload an audio file (.wav or .pkf) for classification. Then click 'Submit' to see the prediction and corresponding species image.")
    
    uploaded_test = st.file_uploader("Choose an audio file", type=["wav", "pkf"], key="test")
    
    if uploaded_test is not None:
        if st.button("Submit"):
            # Capture original extension
            original_ext = os.path.splitext(uploaded_test.name)[1].lower()
            # Create a temporary file with the correct extension
            temp_test_file = f"temp_test_file{original_ext}"
            with open(temp_test_file, "wb") as f:
                f.write(uploaded_test.getbuffer())
            
            waveform = load_audio(temp_test_file)
            if waveform is not None:
                spec = waveform_to_logmel(waveform)
                if spec is not None:
                    spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-6)
                    fixed_length = 300
                    if spec.shape[1] < fixed_length:
                        pad_width = fixed_length - spec.shape[1]
                        spec = np.pad(spec, ((0,0),(0, pad_width)), mode='constant')
                    elif spec.shape[1] > fixed_length:
                        start = (spec.shape[1] - fixed_length) // 2
                        spec = spec[:, start:start+fixed_length]
                    # Now spec shape is (n_mels, fixed_length)
                    # Expand dims twice to get shape (1, 1, n_mels, fixed_length)
                    spec = np.expand_dims(spec, axis=0)
                    spec = np.expand_dims(spec, axis=0)
                    spec = spec.astype(np.float32)
                    
                    fig3, ax3 = plt.subplots(figsize=(10, 4))
                    librosa.display.specshow(spec[0,0], sr=192000, hop_length=512, x_axis='time', y_axis='mel', ax=ax3)
                    ax3.set_title("Uploaded File Spectrogram")
                    st.pyplot(fig3)
                    
                    input_tensor = torch.tensor(spec).to(device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        _, predicted = torch.max(outputs, 1)
                    species = "Pipistrellus ceylonicus" if predicted.item() == 0 else "Rhinolophus indorouxii"
                    st.write(f"**Predicted Species:** {species}")
                    
                    if species in species_to_image:
                        image_path = species_to_image[species]
                        if os.path.exists(image_path):
                            st.image(Image.open(image_path), caption=species, use_column_width=True)
                        else:
                            st.write("Species image not found.")
                    else:
                        st.write("No image mapping available for this species.")
                else:
                    st.error("Error generating spectrogram.")
            else:
                st.error("Error loading the uploaded file.")

