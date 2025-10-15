# 🎬 Action Recognition Training Guide

## 📋 Overview

Hệ thống **Action Recognition** sử dụng **SlowFast architecture** để phân loại hành vi từ video:

### 🎯 Target Actions (4 classes)

1. **🔴 FALL** (Ngã) - **Emergency Priority**
   - `falling/` folder videos
   - `URFD/` dataset (fall detection)
   - NTU actions: A42 (staggering), A43 (falling down)

2. **🟠 HITTING** (Đánh) - **Emergency Priority**
   - `hitting/`, `kicking/`, `murdering/`, `vandalizing/` folders
   - NTU actions: A50 (punch/slap), A51 (kicking), A106 (hit with object), A107 (wield knife), A108 (knock over), A110 (shoot with gun)

3. **🟡 RUNNING** (Chạy) - **Alert Priority**
   - `running/`, `panicking/` folders
   - Fast movement, emergency behavior

4. **🟤 WARNING** (Cảnh báo) - **Warning Priority**
   - `stealing/`, `igniting/`, `luggage/`, `carcrash/` folders
   - NTU actions: A109 (grab stuff), A111-A118 (suspicious activities)
   - 50% of IITB-Corridor videos (surveillance)

## 🏗️ Architecture: SlowFast Network

### Slow Pathway
- **Low frame rate** (every 4th frame)
- **High spatial resolution** (224x224)
- Captures **spatial details** and **object appearance**

### Fast Pathway
- **High frame rate** (all frames)
- **Low spatial resolution** (112x112)
- Captures **temporal motion** and **dynamics**

### Lateral Connections
- **Fast → Slow** information flow
- Combines **spatial + temporal** features

## 🚀 Quick Start

### 1️⃣ Install Requirements
```bash
pip install -r requirements_action_recognition.txt
```

### 2️⃣ Organize Dataset
```bash
cd D:\SPHAR-Dataset\scripts
python action_recognition_trainer.py --organize-only
```

### 3️⃣ Train Model
```bash
python action_recognition_trainer.py --epochs 50 --batch-size 4
```

### 4️⃣ Test Model
```bash
# Test on webcam
python test_action_recognition.py --source webcam

# Test on video
python test_action_recognition.py --source video.mp4 --output result.mp4
```

### 5️⃣ Integrated System (YOLO + Action Recognition)
```bash
# Combined detection + action recognition
python integrated_detection_action.py --source webcam
```

## 📊 Dataset Organization

### Automatic Organization
Script tự động tổ chức dữ liệu từ:

```
videos/
├── falling/           → fall
├── URFD/             → fall  
├── hitting/          → hitting
├── kicking/          → hitting
├── running/          → running
├── panicking/        → running
├── stealing/         → warning
├── IITB-Corridor/    → warning (50%) + neutral (50%)
└── NTU/
    ├── A042/         → fall (staggering)
    ├── A043/         → fall (falling down)
    ├── A050/         → hitting (punch/slap)
    ├── A051/         → hitting (kicking)
    └── ...
```

### Output Structure
```
action_recognition/
├── train/
│   ├── fall/
│   ├── hitting/
│   ├── running/
│   └── warning/
├── val/
│   └── (same structure)
├── test/
│   └── (same structure)
├── dataset_info.json
└── class_mapping.json
```

### Split Ratios
- **70%** Training
- **15%** Validation  
- **15%** Testing

## 🎛️ Training Parameters

### Default Settings
```bash
python action_recognition_trainer.py \
    --epochs 50 \
    --batch-size 4 \
    --sequence-length 16 \
    --videos-root "D:\SPHAR-Dataset\videos" \
    --output-dir "D:\SPHAR-Dataset\action_recognition"
```

### Key Parameters
- **`--sequence-length 16`**: Number of frames per clip
- **`--batch-size 4`**: Adjust based on GPU memory
- **`--epochs 50`**: Training epochs
- **`--organize-only`**: Only organize dataset, don't train

## 📈 Expected Performance

### Training Metrics
- **Loss**: Should decrease to < 0.5
- **Accuracy**: Target > 85% on validation
- **Training time**: ~2-4 hours on GTX 1660 SUPER

### Action Priority Levels
```python
priorities = {
    'fall': 3,      # 🚨 EMERGENCY
    'hitting': 3,   # 🚨 EMERGENCY  
    'running': 2,   # ⚠️ ALERT
    'warning': 1,   # ⚡ WARNING
}
```

## 🎨 Visual Output

### Action Recognition Display
```
┌─────────────────────────────────┐
│ Action: HITTING                 │ ← Red color (emergency)
│ Confidence: 0.87                │
│ Raw Conf: 0.92                  │
│ 🚨 EMERGENCY                    │ ← Priority indicator
│                                 │
│ History: normal -> warning ->   │ ← Action sequence
│          hitting               │
└─────────────────────────────────┘
```

### Integrated System Display
```
Person #1 [HITTING] (0.87)  ← Bounding box + action
🚨 EMERGENCY                ← Priority warning

Person #2 [RUNNING] (0.74)
⚠️ ALERT

Tracking 2 people           ← System status
hitting: 1 | running: 1     ← Action summary
```

## 🔧 Advanced Usage

### Custom Training
```bash
# Longer sequences for better temporal understanding
python action_recognition_trainer.py --sequence-length 32 --epochs 100

# Smaller batch size for limited GPU memory
python action_recognition_trainer.py --batch-size 2

# Custom dataset location
python action_recognition_trainer.py \
    --videos-root "/path/to/videos" \
    --output-dir "/path/to/organized"
```

### Model Fine-tuning
```bash
# Resume training from checkpoint
python action_recognition_trainer.py \
    --resume "D:\SPHAR-Dataset\models\action_recognition_slowfast.pt" \
    --epochs 20
```

### Testing Options
```bash
# Test with different confidence threshold
python test_action_recognition.py \
    --source video.mp4 \
    --confidence 0.7

# Test integrated system
python integrated_detection_action.py \
    --source video.mp4 \
    --conf 0.5 \
    --output integrated_result.mp4
```

## 📁 Output Files

### Training Outputs
- **Model**: `D:\SPHAR-Dataset\models\action_recognition_slowfast.pt`
- **Dataset info**: `D:\SPHAR-Dataset\action_recognition\dataset_info.json`
- **Class mapping**: `D:\SPHAR-Dataset\action_recognition\class_mapping.json`

### Dataset Info Example
```json
{
  "fall": {
    "total": 150,
    "train": 105,
    "val": 22,
    "test": 23,
    "description": "Falling, staggering, loss of balance"
  },
  "hitting": {
    "total": 200,
    "train": 140,
    "val": 30,
    "test": 30,
    "description": "Aggressive actions: hitting, kicking, violence"
  }
}
```

### Class Mapping Example
```json
{
  "fall": 0,
  "hitting": 1,
  "running": 2,
  "warning": 3
}
```

## 🎯 Use Cases

### 1. Surveillance Security
```bash
# Monitor IITB-Corridor videos
python integrated_detection_action.py --source 000209
```

### 2. Fall Detection
```bash
# Test on fall detection dataset
python test_action_recognition.py --source "D:\SPHAR-Dataset\videos\falling\fall_video.mp4"
```

### 3. Violence Detection
```bash
# Test on hitting/violence videos
python test_action_recognition.py --source "D:\SPHAR-Dataset\videos\hitting\violence.mp4"
```

### 4. Real-time Monitoring
```bash
# Live webcam monitoring
python integrated_detection_action.py --source webcam
```

## 🆘 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python action_recognition_trainer.py --batch-size 2

# Reduce sequence length
python action_recognition_trainer.py --sequence-length 8
```

### Low Accuracy
```bash
# More training epochs
python action_recognition_trainer.py --epochs 100

# Check dataset balance
python action_recognition_trainer.py --organize-only
# Check dataset_info.json for class distribution
```

### Slow Training
```bash
# Reduce sequence length
python action_recognition_trainer.py --sequence-length 8

# Use CPU if GPU is slow
CUDA_VISIBLE_DEVICES="" python action_recognition_trainer.py
```

### Model Not Found
```bash
# Check model path
ls D:\SPHAR-Dataset\models\action_recognition_slowfast.pt

# Train model first
python action_recognition_trainer.py --epochs 50
```

## 📚 Technical Details

### SlowFast Architecture
- **Slow pathway**: 64→128→256→512 channels
- **Fast pathway**: 8→16→32→64 channels
- **Lateral connections**: Fast→Slow feature fusion
- **Global pooling**: Adaptive average pooling
- **Classifier**: Linear layer with dropout

### Data Processing
- **Frame sampling**: Uniform temporal sampling
- **Resize**: 224×224 for slow, 112×112 for fast
- **Normalization**: [0,1] range
- **Augmentation**: Can be added for better generalization

### Training Strategy
- **Optimizer**: Adam with weight decay
- **Learning rate**: 0.001 with step decay
- **Loss function**: Cross-entropy
- **Batch size**: 4 (adjustable based on GPU)

## 🎉 Example Commands

```bash
# Complete workflow
cd D:\SPHAR-Dataset\scripts

# 1. Organize dataset
python action_recognition_trainer.py --organize-only

# 2. Train model
python action_recognition_trainer.py --epochs 50

# 3. Test on webcam
python test_action_recognition.py --source webcam

# 4. Test integrated system
python integrated_detection_action.py --source webcam

# 5. Test on IITB video
python integrated_detection_action.py --source 000209 --output result.mp4
```

---

**Architecture**: SlowFast Network  
**Framework**: PyTorch  
**Target**: 4-class action recognition (fall, hitting, running, warning)  
**GPU**: NVIDIA GeForce GTX 1660 SUPER (6GB)
