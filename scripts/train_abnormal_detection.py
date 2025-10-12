#!/usr/bin/env python3
"""
Simple training script for abnormal activity detection.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

class VideoDataset(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None, max_frames=16):
        self.video_dir = Path(video_dir)
        self.max_frames = max_frames
        self.transform = transform
        
        self.annotations = []
        with open(annotations_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(',')
                self.annotations.append({
                    'filename': parts[0],
                    'label': int(parts[2]),
                    'label_name': parts[1]
                })
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        video_path = self.video_dir / annotation['label_name'] / annotation['filename']
        
        frames = self._load_video_frames(video_path)
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
        
        return frames, annotation['label']
    
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= self.max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        while len(frames) < self.max_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        return [torch.from_numpy(frame).float() for frame in frames[:self.max_frames]]

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=3):  # Changed default to 3 classes
        super(Simple3DCNN, self).__init__()
        
        self.conv3d1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv3d2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 7, 7))
        self.fc = nn.Linear(128 * 7 * 7, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # (batch, channels, frames, height, width)
        x = self.relu(self.conv3d1(x))
        x = self.pool1(x)
        x = self.relu(self.conv3d2(x))
        x = self.pool2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(dataset_dir, model_save_dir, num_epochs=20, batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = VideoDataset(
        Path(dataset_dir) / 'annotations' / 'train_annotations.csv',
        Path(dataset_dir) / 'videos' / 'train',
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model, loss, optimizer
    model = Simple3DCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for videos, labels in pbar:
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
        
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={100.*correct/total:.2f}%')
    
    # Save model
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), Path(model_save_dir) / 'abnormal_detection_model.pth')
    print(f"Model saved to {model_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/home/nguyenhoang/Downloads/abnormal_detection_dataset')
    parser.add_argument('--output', default='/home/nguyenhoang/Downloads/abnormal_detection_models')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    
    args = parser.parse_args()
    train_model(args.dataset, args.output, args.epochs, args.batch_size)
