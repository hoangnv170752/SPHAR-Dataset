#!/usr/bin/env python3
"""
Advanced Data Augmentation for Better Generalization
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import random
from torchvision import transforms
import albumentations as A

class AdvancedVideoAugmentation:
    """Advanced augmentation for video sequences"""
    
    def __init__(self, sequence_length=8):
        self.sequence_length = sequence_length
        
        # Spatial augmentations
        self.spatial_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        ])
        
        # Temporal augmentations
        self.temporal_dropout_prob = 0.1
        self.frame_skip_prob = 0.2
        
    def __call__(self, frames):
        """Apply augmentations to video frames"""
        augmented_frames = []
        
        for frame in frames:
            # Convert to uint8 for albumentations
            if frame.dtype == np.float32:
                frame_uint8 = (frame * 255).astype(np.uint8)
            else:
                frame_uint8 = frame
            
            # Apply spatial augmentation
            augmented = self.spatial_aug(image=frame_uint8)['image']
            
            # Convert back to float32
            augmented = augmented.astype(np.float32) / 255.0
            
            augmented_frames.append(augmented)
        
        # Temporal augmentations
        augmented_frames = self._temporal_augmentation(augmented_frames)
        
        return np.array(augmented_frames)
    
    def _temporal_augmentation(self, frames):
        """Apply temporal augmentations"""
        # Random frame dropout
        if random.random() < self.temporal_dropout_prob:
            dropout_indices = random.sample(range(len(frames)), 
                                          k=min(2, len(frames)//4))
            for idx in dropout_indices:
                if idx > 0:
                    frames[idx] = frames[idx-1]  # Replace with previous frame
        
        # Random frame skip/repeat
        if random.random() < self.frame_skip_prob and len(frames) > 4:
            skip_idx = random.randint(1, len(frames)-2)
            frames[skip_idx] = frames[skip_idx-1]
        
        return frames

class MixupAugmentation:
    """Mixup augmentation for video sequences"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch_frames, batch_labels):
        """Apply mixup to batch"""
        if self.alpha <= 0:
            return batch_frames, batch_labels
        
        batch_size = batch_frames.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Mix frames
        mixed_frames = lam * batch_frames + (1 - lam) * batch_frames[index]
        
        # Mix labels
        y_a, y_b = batch_labels, batch_labels[index]
        
        return mixed_frames, (y_a, y_b, lam)

class CutMixAugmentation:
    """CutMix augmentation for video sequences"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch_frames, batch_labels):
        """Apply CutMix to batch"""
        if self.alpha <= 0:
            return batch_frames, batch_labels
        
        batch_size = batch_frames.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Generate random box
        _, _, T, H, W = batch_frames.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cut_t = int(T * cut_rat)
        
        # Random position
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        ct = np.random.randint(T)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbt1 = np.clip(ct - cut_t // 2, 0, T)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        bbt2 = np.clip(ct + cut_t // 2, 0, T)
        
        # Apply CutMix
        mixed_frames = batch_frames.clone()
        mixed_frames[:, :, bbt1:bbt2, bby1:bby2, bbx1:bbx2] = \
            batch_frames[index, :, bbt1:bbt2, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbt2 - bbt1)) / (W * H * T)
        
        y_a, y_b = batch_labels, batch_labels[index]
        
        return mixed_frames, (y_a, y_b, lam)
