#!/usr/bin/env python3
"""
Improved Model Architecture with Better Regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ImprovedSlowFastModel(nn.Module):
    """Improved SlowFast with better regularization"""
    
    def __init__(self, num_classes, sequence_length=8, dropout_rate=0.3):
        super(ImprovedSlowFastModel, self).__init__()
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        
        # Improved channel sizes
        slow_channels = [32, 64, 128, 256]
        fast_channels = [8, 16, 32, 64]
        
        # Slow pathway with residual connections
        self.slow_pathway = self._make_pathway(3, slow_channels, 'slow')
        
        # Fast pathway with residual connections  
        self.fast_pathway = self._make_pathway(3, fast_channels, 'fast')
        
        # Lateral connections with attention
        self.lateral_connections = nn.ModuleList([
            LateralConnection(fast_channels[i], slow_channels[i]) 
            for i in range(len(fast_channels))
        ])
        
        # Global pooling with attention
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.attention_pool = SpatialAttention3D(slow_channels[-1] + fast_channels[-1])
        
        # Improved classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(slow_channels[-1] + fast_channels[-1]),
            nn.Dropout(dropout_rate),
            nn.Linear(slow_channels[-1] + fast_channels[-1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_pathway(self, in_channels, channels, pathway_type):
        """Create pathway with residual blocks"""
        layers = []
        
        # Initial conv
        if pathway_type == 'slow':
            layers.append(nn.Conv3d(in_channels, channels[0], 
                                  kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)))
        else:
            layers.append(nn.Conv3d(in_channels, channels[0], 
                                  kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)))
        
        layers.extend([
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        ])
        
        # Residual blocks
        for i in range(len(channels)-1):
            layers.append(ResidualBlock3D(channels[i], channels[i+1], 
                                        dropout_rate=self.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass with improved regularization"""
        # Input: (B, T, H, W, C) -> (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        
        # Slow pathway (every 2nd frame)
        slow_indices = torch.arange(0, x.size(2), 2, device=x.device)
        if len(slow_indices) == 0:
            slow_indices = torch.tensor([0], device=x.device)
        slow_x = x[:, :, slow_indices]
        
        # Fast pathway (all frames, half resolution)
        fast_x = F.interpolate(x, size=(x.size(2), x.size(3)//2, x.size(4)//2), 
                              mode='trilinear', align_corners=False)
        
        # Forward through pathways
        slow_out = self.slow_pathway(slow_x)
        fast_out = self.fast_pathway(fast_x)
        
        # Match temporal dimensions by pooling
        if slow_out.size(2) != fast_out.size(2):
            # Pool to match the smaller temporal dimension
            target_t = min(slow_out.size(2), fast_out.size(2))
            if slow_out.size(2) > target_t:
                slow_out = F.adaptive_avg_pool3d(slow_out, (target_t, slow_out.size(3), slow_out.size(4)))
            if fast_out.size(2) > target_t:
                fast_out = F.adaptive_avg_pool3d(fast_out, (target_t, fast_out.size(3), fast_out.size(4)))
        
        # Match spatial dimensions
        if slow_out.size(3) != fast_out.size(3) or slow_out.size(4) != fast_out.size(4):
            target_h, target_w = slow_out.size(3), slow_out.size(4)
            fast_out = F.interpolate(fast_out, size=(fast_out.size(2), target_h, target_w), 
                                   mode='trilinear', align_corners=False)
        
        # Apply attention pooling
        combined = torch.cat([slow_out, fast_out], dim=1)
        attended = self.attention_pool(combined)
        
        # Global pooling
        pooled = self.global_pool(attended)
        
        # Flatten
        features = pooled.view(pooled.size(0), -1)
        
        # Classification
        output = self.classifier(features)
        
        return output

class ResidualBlock3D(nn.Module):
    """3D Residual block with improved regularization"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.dropout1 = nn.Dropout3d(dropout_rate)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout2 = nn.Dropout3d(dropout_rate)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class LateralConnection(nn.Module):
    """Lateral connection with attention mechanism"""
    
    def __init__(self, fast_channels, slow_channels):
        super(LateralConnection, self).__init__()
        
        self.conv = nn.Conv3d(fast_channels, slow_channels, kernel_size=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(slow_channels, slow_channels//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(slow_channels//4, slow_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, fast_features, slow_features):
        # Convert fast to slow
        fast_to_slow = self.conv(fast_features)
        
        # Apply attention
        attention_weights = self.attention(slow_features)
        
        # Combine with attention
        combined = slow_features + attention_weights * fast_to_slow
        
        return combined

class SpatialAttention3D(nn.Module):
    """3D Spatial attention mechanism"""
    
    def __init__(self, channels):
        super(SpatialAttention3D, self).__init__()
        
        self.conv = nn.Conv3d(channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Compute attention map
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        
        # Apply attention
        return x * attention

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -torch.sum(smooth_targets * log_probs, dim=1)
        
        return loss.mean()
