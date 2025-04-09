import streamlit as st
import os
import tempfile
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from moviepy.editor import VideoFileClip
from ultralytics import YOLO
import soundfile as sf

# Define simple GAN models
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32000, 4096),
            nn.ReLU(),
            nn.Linear(4096, 32000),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32000, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Utility Functions
def extract_audio(video_path, audio_output_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_output_path)

def compute_energy(audio, sr=16000):
    frames = librosa.util.frame(audio, frame_length=1024, hop_length=512)
    energy = np.sum(frames**2, axis=0)
    return energy

# Main App
st.title("🚗 Vehicle Entry Detector via Audio-Guided GAN")

video_file = st.file_uploader("Upload a video", type=["mp4"])
if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.video(video_path)
    audio_path = video_path.replace(".mp4", ".wav")
    extract_audio(video_path, audio_path)

    # Load and segment audio
    sr = 16000
    y, _ = librosa.load(audio_path, sr=sr)
    segment_length = 2.0
    samples = librosa.util.frame(y, frame_length=int(sr*segment_length), hop_length=int(sr*segment_length)).T

    G = Generator()
    G.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G.to(device)

    # Denoise audio using untrained GAN for now
    enhanced_audio = []
    for segment in samples:
        noise = segment + 0.05 * np.random.randn(*segment.shape)
        noisy_tensor = torch.tensor(noise, dtype=torch.float32).to(device).view(1, -1)
        with torch.no_grad():
            enhanced = G(noisy_tensor).cpu().view(-1).numpy()
        enhanced_audio.append(enhanced)

    enhanced_audio = np.concatenate(enhanced_audio)
    enhanced_path = video_path.replace(".mp4", "_enhanced.wav")
    sf.write(enhanced_path, enhanced_audio, samplerate=sr)

    # Compute energy and find peaks
    energy = compute_energy(enhanced_audio)
    threshold = np.percentile(energy, 95)
    peaks = np.where(energy > threshold)[0]
    times = peaks * 512 / sr
    st.write("🔍 Likely vehicle entry timestamps (in seconds):", times[:5])

    # YOLO Detection
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    model = YOLO('yolov8n.pt')  # make sure the model file is available

    frame_indices = [int(t * fps) for t in times[:5]]
    frame_id = 0
    output_frames = []

    with st.spinner("🔎 Running YOLO on selected frames..."):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id in frame_indices:
                results = model(frame)[0]
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if label in ['car', 'truck']:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                output_frames.append(frame)
            frame_id += 1
        cap.release()

    st.success("✅ Detection Complete. Showing result frames:")

    for f in output_frames:
        st.image(f, channels="BGR")
