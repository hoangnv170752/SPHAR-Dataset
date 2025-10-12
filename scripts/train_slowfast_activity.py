#!/usr/bin/env python3
"""
SlowFast Training Script for 3-Class Activity Detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict

class SlowFastNet(nn.Module):
    def __init__(self, num_classes=3):
        super(SlowFastNet, self).__init__()
        
        # Slow pathway (8 frames)
        self.slow_conv1 = nn.Conv3d(3, 32, (1,7,7), stride=(1,2,2), padding=(0,3,3))
        self.slow_conv2 = nn.Conv3d(32, 64, (3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.slow_conv3 = nn.Conv3d(64, 128, (3,3,3), stride=(2,2,2), padding=(1,1,1))
        
        # Fast pathway (32 frames)
        self.fast_conv1 = nn.Conv3d(3, 8, (5,7,7), stride=(1,2,2), padding=(2,3,3))
        self.fast_conv2 = nn.Conv3d(8, 16, (3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.fast_conv3 = nn.Conv3d(16, 32, (3,3,3), stride=(2,2,2), padding=(1,1,1))
        
        # Lateral connections
        self.lateral1 = nn.Conv3d(8, 32, 1)
        self.lateral2 = nn.Conv3d(16, 64, 1)
        
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(160, num_classes)  # 128 + 32
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1))
        
    def forward(self, slow_input, fast_input):
        # Slow pathway
        slow = self.relu(self.slow_conv1(slow_input))
        slow = self.maxpool(slow)
        
        # Fast pathway
        fast = self.relu(self.fast_conv1(fast_input))
        fast = self.maxpool(fast)
        
        # Lateral connection 1
        fast_lateral = self.lateral1(fast)
        if fast_lateral.size(2) != slow.size(2):
            fast_lateral = torch.nn.functional.interpolate(
                fast_lateral, size=slow.shape[2:], mode='trilinear', align_corners=False)
        slow = slow + fast_lateral
        
        # Continue
        slow = self.relu(self.slow_conv2(slow))
        fast = self.relu(self.fast_conv2(fast))
        
        # Lateral connection 2
        fast_lateral = self.lateral2(fast)
        if fast_lateral.size(2) != slow.size(2):
            fast_lateral = torch.nn.functional.interpolate(
                fast_lateral, size=slow.shape[2:], mode='trilinear', align_corners=False)
        slow = slow + fast_lateral
        
        # Final
        slow = self.relu(self.slow_conv3(slow))
        fast = self.relu(self.fast_conv3(fast))
        
        # Pool and classify
        slow_pooled = self.global_pool(slow).view(slow.size(0), -1)
        fast_pooled = self.global_pool(fast).view(fast.size(0), -1)
        
        features = torch.cat([slow_pooled, fast_pooled], dim=1)
        return self.fc(features)

class ActivityDataset(Dataset):
    def __init__(self, dataset_path, split='train', slow_frames=8, fast_frames=32):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        
        self.class_to_idx = {'normal': 0, 'abnormal_physical': 1, 'abnormal_biological': 2}
        
        self.video_files = []
        self.labels = []
        
        split_path = self.dataset_path / 'videos' / split
        for class_name in self.class_to_idx.keys():
            class_path = split_path / class_name
            if class_path.exists():
                for ext in ['*.mp4', '*.avi']:
                    for video_file in class_path.glob(ext):
                        self.video_files.append(video_file)
                        self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.video_files)} videos for {split}")
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        
        slow_frames, fast_frames = self.load_video_frames(video_path)
        return slow_frames, fast_frames, label
    
    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 32
        
        # Sample frames
        slow_frames = self.sample_frames(frames, self.slow_frames, stride=4)
        fast_frames = self.sample_frames(frames, self.fast_frames, stride=1)
        
        # Convert to tensor (C, T, H, W)
        slow_frames = np.array(slow_frames).transpose(3,0,1,2).astype(np.float32) / 255.0
        fast_frames = np.array(fast_frames).transpose(3,0,1,2).astype(np.float32) / 255.0
        
        return torch.from_numpy(slow_frames), torch.from_numpy(fast_frames)
    
    def sample_frames(self, frames, num_frames, stride=1):
        total_frames = len(frames)
        if total_frames <= num_frames * stride:
            sampled = []
            for i in range(num_frames):
                idx = min(i * stride, total_frames - 1)
                sampled.append(frames[idx])
            return sampled
        else:
            indices = np.linspace(0, total_frames-1, num_frames*stride, dtype=int)
            indices = indices[::stride][:num_frames]
            return [frames[i] for i in indices]

def train_model(dataset_path, epochs=50, batch_size=4, device='cuda'):
    # Load datasets
    train_dataset = ActivityDataset(dataset_path, 'train')
    val_dataset = ActivityDataset(dataset_path, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = SlowFastNet(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for slow_data, fast_data, targets in train_loader:
            slow_data, fast_data, targets = slow_data.to(device), fast_data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(slow_data, fast_data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for slow_data, fast_data, targets in val_loader:
                slow_data, fast_data, targets = slow_data.to(device), fast_data.to(device), targets.to(device)
                outputs = model(slow_data, fast_data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_slowfast_model.pth')
    
    print(f'Best validation accuracy: {best_acc:.2f}%')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device')
    
    args = parser.parse_args()
    
    train_model(args.dataset, args.epochs, args.batch_size, args.device)

if __name__ == "__main__":
    main()
