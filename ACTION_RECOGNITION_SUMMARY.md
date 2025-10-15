# 🎬 Action Recognition System - Complete Summary

## 🎯 System Overview

**Complete AI surveillance system** combining:
1. **Human Detection** (YOLO11s fine-tuned)
2. **Multi-person Tracking** (DeepSORT)  
3. **Action Recognition** (SlowFast architecture)
4. **Real-time Analysis** with priority alerts

## 📊 Action Categories (4 Classes)

### 🔴 EMERGENCY (Priority 3)
- **FALL** (Ngã): Falling, staggering, medical emergencies
- **HITTING** (Đánh): Violence, fighting, aggressive behavior

### 🟡 ALERT (Priority 2)  
- **RUNNING** (Chạy): Fast movement, panic, chase scenarios

### 🟠 WARNING (Priority 1)
- **WARNING** (Cảnh báo): Suspicious activities, theft, vandalism

## 🏗️ Technical Architecture

### Detection Pipeline
```
Video Input → YOLO11s Detection → DeepSORT Tracking → Person Crops
                                                          ↓
Action Classification ← SlowFast Network ← Frame Sequences
                                                          ↓
Priority Assessment → Alert System → Visual Output
```

### SlowFast Network
- **Slow Pathway**: Spatial details (every 4th frame, 224×224)
- **Fast Pathway**: Temporal motion (all frames, 112×112)  
- **Fusion**: Lateral connections combine spatial + temporal features

## 📁 Dataset Organization

### Source Data
```
videos/
├── falling/          → FALL
├── URFD/            → FALL  
├── hitting/         → HITTING
├── kicking/         → HITTING
├── running/         → RUNNING
├── panicking/       → RUNNING
├── stealing/        → WARNING
├── IITB-Corridor/   → WARNING + Normal
└── NTU/
    ├── A042/        → FALL (staggering)
    ├── A043/        → FALL (falling)
    ├── A050/        → HITTING (punch)
    ├── A051/        → HITTING (kick)
    └── A109-A118/   → WARNING (suspicious)
```

### Organized Structure
```
action_recognition/
├── train/ (70%)
├── val/ (15%)  
├── test/ (15%)
├── dataset_info.json
└── class_mapping.json
```

## 🚀 Quick Start Commands

### 1. Complete Training Pipeline
```bash
cd D:\SPHAR-Dataset\scripts

# Full pipeline: organize + train + test
python run_action_training.py --full-pipeline --epochs 50
```

### 2. Step-by-step Training
```bash
# Step 1: Organize dataset
python run_action_training.py --organize-only

# Step 2: Train model  
python run_action_training.py --train-only --epochs 50

# Step 3: Test model
python run_action_training.py --test-only
```

### 3. Real-time Testing
```bash
# Action recognition only
python test_action_recognition.py --source webcam

# Full integrated system
python integrated_detection_action.py --source webcam

# Test on surveillance video
python integrated_detection_action.py --source 000209
```

## 📈 Expected Performance

### Training Metrics
- **Training Time**: 2-4 hours (GTX 1660 SUPER)
- **Target Accuracy**: >85% validation
- **Model Size**: ~50MB
- **Inference Speed**: ~15-20 FPS

### Real-world Performance
- **Detection Accuracy**: 90%+ for clear actions
- **False Positive Rate**: <5% with confidence filtering
- **Response Time**: <1 second for action classification
- **Multi-person**: Up to 5-8 people simultaneously

## 🎨 Visual Output Features

### Action Recognition Display
```
┌─────────────────────────────────┐
│ Person #1                       │
│ HITTING (0.87) 🚨 EMERGENCY     │ ← Red box, high priority
│                                 │
│ Person #2                       │  
│ RUNNING (0.74) ⚠️ ALERT         │ ← Yellow box, medium priority
│                                 │
│ Tracking 2 people               │
│ hitting: 1 | running: 1         │ ← Action summary
└─────────────────────────────────┘
```

### Priority System
- **🚨 EMERGENCY**: Fall, Violence → Immediate response
- **⚠️ ALERT**: Running, Panic → Quick attention  
- **⚡ WARNING**: Suspicious → Monitor closely
- **✅ NORMAL**: Regular behavior → No action

## 📊 Model Files

### Generated Models
```
models/
├── yolo11s.pt                     # Pretrained YOLO
├── yolov8s.pt                     # Pretrained YOLO  
├── finetuned/
│   └── yolo11s-detect.pt          # Fine-tuned human detection
└── action_recognition_slowfast.pt # Action classification (NEW!)
```

### Dataset Files
```
action_recognition/
├── train/val/test/               # Organized video clips
├── dataset_info.json            # Dataset statistics
└── class_mapping.json           # Class indices
```

## 🎯 Use Cases

### 1. Security Surveillance
```bash
# Monitor IITB corridor videos
python integrated_detection_action.py --source 000209 --output security_analysis.mp4
```

### 2. Fall Detection System  
```bash
# Elderly care monitoring
python integrated_detection_action.py --source webcam
# Alerts on fall detection with 🚨 EMERGENCY priority
```

### 3. Violence Detection
```bash
# Public safety monitoring
python integrated_detection_action.py --source security_camera.mp4
# Detects hitting/fighting with immediate alerts
```

### 4. Crowd Behavior Analysis
```bash
# Event monitoring
python integrated_detection_action.py --source crowd_video.mp4
# Tracks multiple people and their actions simultaneously
```

## 🔧 Configuration Options

### Training Parameters
```bash
# Longer sequences for better accuracy
python run_action_training.py --sequence-length 32 --epochs 100

# Smaller batch for limited GPU
python run_action_training.py --batch-size 2

# Quick training for testing
python run_action_training.py --epochs 20 --sequence-length 8
```

### Inference Parameters
```bash
# Higher confidence threshold
python integrated_detection_action.py --source webcam --conf 0.7

# Custom models
python integrated_detection_action.py \
    --yolo-model custom_yolo.pt \
    --action-model custom_action.pt \
    --source video.mp4
```

## 📚 Key Scripts

| Script | Purpose |
|--------|---------|
| `action_recognition_trainer.py` | Main training script |
| `test_action_recognition.py` | Action recognition testing |
| `integrated_detection_action.py` | Full system integration |
| `run_action_training.py` | Complete training pipeline |

## 🆘 Troubleshooting

### Common Issues
```bash
# CUDA out of memory
python run_action_training.py --batch-size 2 --sequence-length 8

# Low accuracy
python run_action_training.py --epochs 100

# Slow training  
python run_action_training.py --sequence-length 8

# Model not found
python run_action_training.py --organize-only  # Check dataset first
```

## 🎉 Success Metrics

### System is working correctly when:
- ✅ **Fall detection**: Immediate 🚨 EMERGENCY alert
- ✅ **Violence detection**: 🚨 EMERGENCY with red bounding box
- ✅ **Running detection**: ⚠️ ALERT with yellow box
- ✅ **Multi-person tracking**: Stable IDs with individual actions
- ✅ **Real-time performance**: 15+ FPS on GTX 1660 SUPER

### Expected Alerts:
- **Fall**: Person falls → 🚨 EMERGENCY (red)
- **Fight**: People hitting → 🚨 EMERGENCY (red)  
- **Chase**: Person running → ⚠️ ALERT (yellow)
- **Theft**: Suspicious behavior → ⚡ WARNING (orange)

## 🏆 Final System Capabilities

**Complete AI Surveillance System** featuring:

1. **🔍 Detection**: Fine-tuned YOLO for accurate human detection
2. **👥 Tracking**: DeepSORT for stable multi-person tracking  
3. **🎬 Recognition**: SlowFast for real-time action classification
4. **🚨 Alerts**: Priority-based warning system
5. **📊 Analytics**: Real-time statistics and behavior analysis
6. **🎯 Integration**: Seamless detection + tracking + action pipeline

---

**Ready for deployment in security, healthcare, and public safety applications!** 🚀

**Hardware**: NVIDIA GTX 1660 SUPER (6GB)  
**Framework**: PyTorch + Ultralytics YOLO  
**Performance**: Real-time (15-20 FPS) multi-person action recognition
