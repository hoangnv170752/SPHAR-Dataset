#!/usr/bin/env python3
"""
Action Recognition Training with SlowFast Architecture
Fine-tuning from YOLO11s-detect.pt for human action classification

Target Actions:
- fall (ngã): falling, A43 (falling down), A42 (staggering)
- hitting (đánh): hitting, A50 (punch/slap), A51 (kicking), A106 (hit with object)
- running (chạy): running, panicking (emergency behavior)
- warning (cảnh báo): other suspicious activities
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from collections import defaultdict
import random
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionDatasetOrganizer:
    """Organize videos into action categories"""
    
    def __init__(self, videos_root):
        self.videos_root = Path(videos_root)
        self.action_mapping = self._create_action_mapping()
        
    def _create_action_mapping(self):
        """Map video folders to action categories"""
        return {
            # FALL actions
            'fall': {
                'folders': ['falling', 'URFD'],
                'ntu_actions': ['A42', 'A43'],  # staggering, falling down
                'description': 'Falling, staggering, loss of balance'
            },
            
            # HITTING actions  
            'hitting': {
                'folders': ['hitting', 'kicking', 'murdering', 'vandalizing'],
                'ntu_actions': ['A50', 'A51', 'A106', 'A107', 'A108', 'A110'],  # punch, kick, hit with object, wield knife, knock over, shoot
                'description': 'Aggressive actions: hitting, kicking, violence'
            },
            
            # RUNNING actions
            'running': {
                'folders': ['running', 'panicking'],
                'ntu_actions': [],  # No specific NTU running actions in the image
                'description': 'Fast movement, running, panic behavior'
            },
            
            # WARNING actions (suspicious but not immediately dangerous)
            'warning': {
                'folders': ['stealing', 'igniting', 'luggage', 'carcrash'],
                'ntu_actions': ['A109', 'A111', 'A112', 'A113', 'A114', 'A115', 'A116', 'A117', 'A118'],  # grab stuff, step on foot, high-five, cheers, carry object, take photo, follow, whisper, exchange things
                'description': 'Suspicious activities requiring attention'
            }
        }
    
    def organize_dataset(self, output_dir):
        """Organize videos into training structure"""
        output_dir = Path(output_dir)
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for action in self.action_mapping.keys():
                (output_dir / split / action).mkdir(parents=True, exist_ok=True)
        
        # Collect all videos by category
        video_collections = defaultdict(list)
        
        # Process direct action folders
        for action, config in self.action_mapping.items():
            for folder in config['folders']:
                folder_path = self.videos_root / folder
                if folder_path.exists():
                    videos = list(folder_path.glob('**/*.mp4')) + list(folder_path.glob('**/*.avi'))
                    video_collections[action].extend(videos)
                    logger.info(f"Found {len(videos)} videos in {folder} for action '{action}'")
        
        # Process URFD dataset (images organized by sequences)
        urfd_dir = self.videos_root / 'URFD'
        if urfd_dir.exists():
            for action, config in self.action_mapping.items():
                if 'URFD' in config['folders']:
                    # URFD contains images, not videos - create image sequences for fall detection
                    fall_images = []
                    for split in ['train', 'valid', 'test']:
                        images_dir = urfd_dir / split / 'images'
                        if images_dir.exists():
                            # Look for fall images
                            fall_imgs = list(images_dir.glob('fall-*.jpg'))
                            fall_images.extend(fall_imgs)
                    
                    # Group fall images by sequence (fall-01, fall-02, etc.)
                    fall_sequences = {}
                    for img_path in fall_images:
                        # Extract sequence ID from filename (e.g., fall-01-cam0-rgb-012)
                        filename = img_path.stem
                        if 'fall-' in filename:
                            seq_id = filename.split('-')[1]  # Get '01', '02', etc.
                            if seq_id not in fall_sequences:
                                fall_sequences[seq_id] = []
                            fall_sequences[seq_id].append(img_path)
                    
                    # Create pseudo-video paths for each sequence
                    urfd_sequences = []
                    for seq_id, images in fall_sequences.items():
                        if len(images) >= 8:  # Need at least 8 images for a sequence
                            # Sort images by frame number
                            images.sort(key=lambda x: int(x.stem.split('-')[-1].split('_')[0]))
                            # Create a pseudo video path that we'll handle specially
                            pseudo_video = urfd_dir / f"fall_sequence_{seq_id}.urfd"
                            urfd_sequences.append(pseudo_video)
                    
                    video_collections[action].extend(urfd_sequences)
                    logger.info(f"Found {len(urfd_sequences)} URFD fall sequences for action '{action}'")
        
        # Process NTU dataset
        ntu_dir = self.videos_root / 'NTU'
        if ntu_dir.exists():
            for action, config in self.action_mapping.items():
                for ntu_action in config['ntu_actions']:
                    action_dir = ntu_dir / ntu_action
                    if action_dir.exists():
                        videos = list(action_dir.glob('**/*.avi')) + list(action_dir.glob('**/*.mp4'))
                        video_collections[action].extend(videos)
                        logger.info(f"Found {len(videos)} NTU videos in {ntu_action} for action '{action}'")
        
        # Process IITB-Corridor (all as warning behavior - surveillance videos)
        iitb_dir = self.videos_root / 'IITB-Corridor'
        if iitb_dir.exists():
            videos = list(iitb_dir.glob('**/*.avi'))
            # Add all IITB videos to warning category (surveillance/suspicious behavior)
            video_collections['warning'].extend(videos)
            logger.info(f"Added {len(videos)} IITB-Corridor videos to warning category")
        
        # Split datasets (70% train, 15% val, 15% test)
        dataset_info = {}
        for action, videos in video_collections.items():
            if not videos:
                continue
                
            random.shuffle(videos)
            n_videos = len(videos)
            
            n_train = int(0.7 * n_videos)
            n_val = int(0.15 * n_videos)
            
            train_videos = videos[:n_train]
            val_videos = videos[n_train:n_train + n_val]
            test_videos = videos[n_train + n_val:]
            
            # Copy videos to organized structure
            for split, video_list in [('train', train_videos), ('val', val_videos), ('test', test_videos)]:
                split_dir = output_dir / split / action
                split_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                
                for i, video_path in enumerate(video_list):
                    # Skip if video doesn't exist
                    if not video_path.exists():
                        logger.warning(f"Video not found: {video_path}")
                        continue
                    
                    # Handle URFD pseudo videos
                    if video_path.suffix == '.urfd':
                        new_name = f"{action}_{i:04d}.urfd"
                    else:
                        new_name = f"{action}_{i:04d}{video_path.suffix}"
                    
                    new_path = split_dir / new_name
                    
                    # Create symlink instead of copying to save space
                    if not new_path.exists():
                        try:
                            new_path.symlink_to(video_path.absolute())
                        except OSError:
                            # If symlink fails, copy the file
                            try:
                                import shutil
                                shutil.copy2(video_path, new_path)
                            except Exception as e:
                                logger.error(f"Failed to copy {video_path} to {new_path}: {e}")
                                continue
            
            dataset_info[action] = {
                'total': n_videos,
                'train': len(train_videos),
                'val': len(val_videos),
                'test': len(test_videos),
                'description': self.action_mapping[action]['description']
            }
            
            logger.info(f"Action '{action}': {n_videos} total -> {len(train_videos)} train, {len(val_videos)} val, {len(test_videos)} test")
        
        # Save dataset info
        with open(output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Create class mapping
        class_mapping = {action: idx for idx, action in enumerate(sorted(dataset_info.keys()))}
        with open(output_dir / 'class_mapping.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        logger.info(f"Dataset organized in {output_dir}")
        logger.info(f"Classes: {list(class_mapping.keys())}")
        
        return dataset_info, class_mapping

class VideoActionDataset(Dataset):
    """Dataset for video action recognition"""
    
    def __init__(self, data_dir, split='train', sequence_length=16, transform=None, class_mapping=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Load class mapping
        if class_mapping is None:
            with open(self.data_dir / 'class_mapping.json', 'r') as f:
                self.class_mapping = json.load(f)
        else:
            self.class_mapping = class_mapping
        
        self.num_classes = len(self.class_mapping)
        
        # Collect video files
        self.video_files = []
        self.labels = []
        
        split_dir = self.data_dir / split
        for action, class_idx in self.class_mapping.items():
            action_dir = split_dir / action
            if action_dir.exists():
                videos = list(action_dir.glob('*.mp4')) + list(action_dir.glob('*.avi'))
                self.video_files.extend(videos)
                self.labels.extend([class_idx] * len(videos))
        
        logger.info(f"Loaded {len(self.video_files)} videos for {split} split")
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self._load_video_frames(video_path)
        
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label
    
    def _load_video_frames(self, video_path):
        """Load and sample frames from video or URFD image sequence"""
        video_path = Path(video_path)
        
        # Handle URFD image sequences
        if video_path.suffix == '.urfd':
            return self._load_urfd_sequence(video_path)
        
        # Handle regular video files
        cap = cv2.VideoCapture(str(video_path))
        
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count < self.sequence_length:
            # If video is too short, repeat frames
            indices = np.linspace(0, frame_count - 1, self.sequence_length, dtype=int)
        else:
            # Sample frames uniformly
            indices = np.linspace(0, frame_count - 1, self.sequence_length, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        # Convert to numpy array (T, H, W, C)
        frames = np.array(frames)
        
        # Resize frames to standard size
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (224, 224))
            resized_frames.append(resized_frame)
        
        return np.array(resized_frames)
    
    def _load_urfd_sequence(self, pseudo_video_path):
        """Load URFD image sequence"""
        # Extract sequence ID from pseudo path
        seq_id = pseudo_video_path.stem.split('_')[-1]  # Get sequence ID
        
        # Find all fall images for this sequence
        urfd_dir = pseudo_video_path.parent
        fall_images = []
        
        for split in ['train', 'valid', 'test']:
            images_dir = urfd_dir / split / 'images'
            if images_dir.exists():
                # Look for images from this sequence
                pattern = f"fall-{seq_id}-*.jpg"
                seq_images = list(images_dir.glob(pattern))
                fall_images.extend(seq_images)
        
        if not fall_images:
            # Fallback: create dummy frames
            dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            return np.array([dummy_frame] * self.sequence_length)
        
        # Sort images by frame number
        fall_images.sort(key=lambda x: int(x.stem.split('-')[-1].split('_')[0]))
        
        # Load images
        frames = []
        for img_path in fall_images:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        
        # Sample or repeat frames to match sequence_length
        if len(frames) < self.sequence_length:
            # Repeat frames if too few
            while len(frames) < self.sequence_length:
                frames.extend(frames[:self.sequence_length - len(frames)])
        elif len(frames) > self.sequence_length:
            # Sample frames if too many
            indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
            frames = [frames[i] for i in indices]
        
        return np.array(frames)

class SlowFastActionModel(nn.Module):
    """SlowFast architecture for action recognition"""
    
    def __init__(self, num_classes, sequence_length=16, pretrained_backbone=None):
        super(SlowFastActionModel, self).__init__()
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
        # Slow pathway (low frame rate, high spatial resolution)
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Slow pathway residual blocks
        self.slow_layer1 = self._make_layer(64, 64, 2, stride=(1, 1, 1))
        self.slow_layer2 = self._make_layer(64, 128, 2, stride=(1, 2, 2))
        self.slow_layer3 = self._make_layer(128, 256, 2, stride=(1, 2, 2))
        self.slow_layer4 = self._make_layer(256, 512, 2, stride=(1, 2, 2))
        
        # Fast pathway (high frame rate, low spatial resolution)
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Fast pathway residual blocks
        self.fast_layer1 = self._make_layer(8, 8, 2, stride=(1, 1, 1))
        self.fast_layer2 = self._make_layer(8, 16, 2, stride=(1, 2, 2))
        self.fast_layer3 = self._make_layer(16, 32, 2, stride=(1, 2, 2))
        self.fast_layer4 = self._make_layer(32, 64, 2, stride=(1, 2, 2))
        
        # Lateral connections (Fast -> Slow)
        self.lateral_conv1 = nn.Conv3d(8, 64 // 8, kernel_size=(5, 1, 1), stride=(4, 1, 1), padding=(2, 0, 0))
        self.lateral_conv2 = nn.Conv3d(16, 128 // 8, kernel_size=(5, 1, 1), stride=(4, 1, 1), padding=(2, 0, 0))
        self.lateral_conv3 = nn.Conv3d(32, 256 // 8, kernel_size=(5, 1, 1), stride=(4, 1, 1), padding=(2, 0, 0))
        self.lateral_conv4 = nn.Conv3d(64, 512 // 8, kernel_size=(5, 1, 1), stride=(4, 1, 1), padding=(2, 0, 0))
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 + 64, num_classes)  # Slow + Fast features
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=(1, 1, 1)):
        """Create residual layer"""
        layers = []
        layers.append(BasicBlock3D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input shape: (B, T, H, W, C) -> (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        
        # Create slow and fast pathways
        # Slow pathway: sample every 4th frame
        slow_x = x[:, :, ::4, :, :]  # (B, C, T//4, H, W)
        
        # Fast pathway: use all frames but downsample spatially
        fast_x = nn.functional.interpolate(x, scale_factor=(1, 0.5, 0.5), mode='trilinear', align_corners=False)
        
        # Slow pathway forward
        slow_x = self.slow_conv1(slow_x)
        slow_x = self.slow_bn1(slow_x)
        slow_x = self.slow_relu(slow_x)
        slow_x = self.slow_maxpool(slow_x)
        
        slow_x = self.slow_layer1(slow_x)
        slow_x = self.slow_layer2(slow_x)
        slow_x = self.slow_layer3(slow_x)
        slow_x = self.slow_layer4(slow_x)
        
        # Fast pathway forward
        fast_x = self.fast_conv1(fast_x)
        fast_x = self.fast_bn1(fast_x)
        fast_x = self.fast_relu(fast_x)
        fast_x = self.fast_maxpool(fast_x)
        
        fast_x1 = self.fast_layer1(fast_x)
        fast_x2 = self.fast_layer2(fast_x1)
        fast_x3 = self.fast_layer3(fast_x2)
        fast_x4 = self.fast_layer4(fast_x3)
        
        # Lateral connections (simplified)
        # In practice, you would add these to corresponding slow pathway features
        
        # Global pooling
        slow_features = self.global_pool(slow_x).flatten(1)  # (B, 512)
        fast_features = self.global_pool(fast_x4).flatten(1)  # (B, 64)
        
        # Concatenate features
        features = torch.cat([slow_features, fast_features], dim=1)  # (B, 576)
        
        # Classification
        features = self.dropout(features)
        output = self.fc(features)
        
        return output

class BasicBlock3D(nn.Module):
    """Basic 3D residual block"""
    
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1)):
        super(BasicBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class ActionRecognitionTrainer:
    """Trainer for action recognition model"""
    
    def __init__(self, model, train_loader, val_loader, device, num_classes):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Convert to float and normalize
            data = data.float() / 255.0
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                data = data.float() / 255.0
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, num_epochs, save_path):
        """Full training loop"""
        best_accuracy = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': val_accuracy,
                    'loss': val_loss
                }, save_path)
                logger.info(f"New best model saved with accuracy: {val_accuracy:.2f}%")
        
        return best_accuracy

class VideoTransform:
    """Custom transform class that can be pickled"""
    def __call__(self, x):
        return torch.tensor(x).float()

def create_transforms():
    """Create data transforms"""
    return VideoTransform()

def main():
    parser = argparse.ArgumentParser(description='Train Action Recognition Model')
    parser.add_argument('--videos-root', default=r'D:\SPHAR-Dataset\videos',
                       help='Root directory of videos')
    parser.add_argument('--output-dir', default=r'D:\SPHAR-Dataset\action_recognition',
                       help='Output directory for organized dataset')
    parser.add_argument('--model-save-path', default=r'D:\SPHAR-Dataset\models\action_recognition_slowfast.pt',
                       help='Path to save trained model')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--sequence-length', type=int, default=16,
                       help='Number of frames per video clip')
    parser.add_argument('--organize-only', action='store_true',
                       help='Only organize dataset, do not train')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Organize dataset
    organizer = ActionDatasetOrganizer(args.videos_root)
    dataset_info, class_mapping = organizer.organize_dataset(args.output_dir)
    
    if args.organize_only:
        logger.info("Dataset organization completed. Exiting.")
        return
    
    # Create datasets
    transform = create_transforms()
    
    train_dataset = VideoActionDataset(
        args.output_dir, 'train', args.sequence_length, transform, class_mapping
    )
    val_dataset = VideoActionDataset(
        args.output_dir, 'val', args.sequence_length, transform, class_mapping
    )
    
    # Create data loaders (num_workers=0 to avoid multiprocessing issues on Windows)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    num_classes = len(class_mapping)
    model = SlowFastActionModel(num_classes, args.sequence_length)
    
    logger.info(f"Model created with {num_classes} classes: {list(class_mapping.keys())}")
    
    # Create trainer
    trainer = ActionRecognitionTrainer(model, train_loader, val_loader, device, num_classes)
    
    # Train model
    logger.info("Starting training...")
    best_accuracy = trainer.train(args.epochs, args.model_save_path)
    
    logger.info(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    logger.info(f"Model saved to: {args.model_save_path}")

if __name__ == "__main__":
    main()
