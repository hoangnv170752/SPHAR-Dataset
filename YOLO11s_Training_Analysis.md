# YOLO11s-Detect Training Analysis

## Overview
This repository contains a complete pipeline for training YOLO11s models for human detection using the SPHAR dataset. The training process involves dataset creation, model fine-tuning, and evaluation.

## Dataset Information

### Source Dataset: SPHAR
- **Full Name**: SPHAR Human Detection Dataset
- **Purpose**: Human detection in surveillance videos
- **Total Frames**: 10,050 frames
- **Frame Extraction**: Every 60 frames (uniform sampling)
- **Videos Processed**: 294 videos

### Dataset Categories
The dataset includes multiple activity categories with varying human presence:

**High Human Presence Categories:**
- NTU: 580 frames (99.8% with humans)
- Neutral: 11 frames (100% with humans)
- Hitting: 17 frames (100% with humans)
- Kicking: 11 frames (100% with humans)

**Mixed Categories:**
- Murdering: 718 frames (52.5% with humans)
- Walking: 47 frames (66% with humans)
- Running: 35 frames (48.6% with humans)
- Sitting: 30 frames (56.7% with humans)

**Background/Low Human Categories:**
- Carcrash: 6,385 frames (46.9% with humans)
- Igniting: 1,779 frames (35.2% with humans)
- Vandalizing: 234 frames (53.8% with humans)

## Train/Test/Val Split

### Split Distribution (70/15/15):
- **Training Set**: 7,034 frames
  - With humans: 3,417 frames (48.6%)
  - Without humans: 3,617 frames (51.4%)

- **Validation Set**: 1,507 frames
  - With humans: 732 frames (48.6%)
  - Without humans: 775 frames (51.4%)

- **Test Set**: 1,509 frames
  - With humans: 733 frames (48.6%)
  - Without humans: 776 frames (51.4%)

### Split Strategy:
- **Balanced Distribution**: Maintains consistent human/non-human ratio across all splits
- **Stratified Sampling**: Ensures representative distribution of categories
- **Random Shuffling**: Applied to both human and non-human frames separately before splitting

## Model Training Configuration

### Base Model
- **Architecture**: YOLOv11s (Small variant)
- **Pre-trained**: Yes (COCO weights)
- **Task**: Object Detection (Human Detection)
- **Classes**: 1 class (person)

### Training Parameters
- **Epochs**: 50
- **Batch Size**: 8
- **Image Size**: 416x416 pixels
- **Optimizer**: AdamW
- **Learning Rate**: 0.001 (initial)
- **Final Learning Rate**: 0.01
- **Device**: CUDA (GPU)

### Optimization Features
- **Automatic Mixed Precision (AMP)**: Enabled for memory efficiency
- **Multi-scale Training**: Enabled for better generalization
- **Data Augmentation**: 
  - Horizontal flip: 50% probability
  - Rotation: ±10 degrees
  - Translation: 10%
  - Scale variation: 50%
  - HSV augmentation
  - Random augmentation
  - Random erasing: 40% probability

### Loss Configuration
- **Box Loss Weight**: 7.5
- **Classification Loss Weight**: 0.5
- **Distribution Focal Loss Weight**: 1.5
- **Label Smoothing**: 0.0

## Training Results

### Performance Metrics
- **mAP50**: 0.8274 (82.74%)
- **mAP50-95**: 0.6082 (60.82%)
- **Precision**: 0.7759 (77.59%)
- **Recall**: 0.7338 (73.38%)
- **F1-Score**: 0.7543 (75.43%)

### Model Output
- **Final Model**: `yolo11s-detect.pt` (19.1 MB)
- **Location**: `D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt`
- **Format**: PyTorch (.pt)

## Training Scripts

### Key Scripts:
1. **`create_human_detection_dataset.py`**: Dataset creation and preprocessing
2. **`train_human_detection.py`**: Basic YOLO training script
3. **`finetune_yolo11s_human.py`**: Advanced fine-tuning with optimizations
4. **`run_finetune_human.py`**: Preset training configurations

### Training Configurations Available:
- **Quick**: 50 epochs, batch=8, 416px (for testing)
- **Standard**: 100 epochs, batch=16, 416px (balanced)
- **High Quality**: 300 epochs, batch=8, 832px (high resolution)
- **Production**: 500 epochs, batch=32, 640px (production ready)

## Dataset Structure

```
human_focused_dataset/
├── images/
│   ├── train/          # 7,034 training images
│   ├── val/            # 1,507 validation images
│   └── test/           # 1,509 test images
├── labels/
│   ├── train/          # YOLO format labels
│   ├── val/            # YOLO format labels
│   └── test/           # YOLO format labels
├── annotations/
│   └── dataset_annotations.json
├── metadata/
│   └── dataset_stats.json
├── dataset.yaml        # YOLO configuration
└── dataset_info.json
```

## Model Performance Analysis

### Strengths:
- **High mAP50**: 82.74% indicates good detection at IoU=0.5
- **Balanced Dataset**: Equal distribution of positive/negative samples
- **Robust Training**: Advanced augmentation and optimization techniques

### Areas for Improvement:
- **mAP50-95**: 60.82% suggests room for improvement at higher IoU thresholds
- **Recall**: 73.38% indicates some humans are missed (false negatives)

## Usage Example

```python
from ultralytics import YOLO

# Load the fine-tuned model
model = YOLO('models/finetuned/yolo11s-detect.pt')

# Run inference
results = model('image.jpg', conf=0.25)

# Process results
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            print(f"Human detected with confidence: {box.conf[0]:.2f}")
```

## Conclusion

The YOLO11s-detect model has been successfully trained on the SPHAR dataset with good performance metrics. The training achieved:
- Strong detection capability (82.74% mAP50)
- Balanced precision-recall trade-off
- Efficient model size (19.1 MB)
- Optimized for real-time human detection in surveillance scenarios

The model is ready for deployment in human detection applications, particularly in surveillance and security contexts.
