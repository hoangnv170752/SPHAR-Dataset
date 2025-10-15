# 🚀 Quick Start Guide - YOLO Human Detection

## 📦 What You Have

1. **✅ Fine-tuned Model**: `yolo11s-detect.pt` - Trained on SPHAR dataset
2. **✅ Tracking System**: YOLO + DeepSORT with ID tracking
3. **✅ Benchmark Tools**: Compare pretrained vs fine-tuned
4. **✅ Confidence Filter**: Only shows detections with confidence > 0.5 (high quality)
5. **✅ Action Recognition**: SlowFast network for behavior classification (NEW!)
6. **✅ Integrated System**: Detection + Tracking + Action Recognition (NEW!)

## 🎯 Common Tasks

### 1️⃣ Test Model on Webcam
```bash
cd D:\SPHAR-Dataset\scripts
python test_yolo_deepsort.py --source webcam
```

### 2️⃣ Test on Sample Video
```bash
python test_sample_videos.py 2people_meet
```

### 3️⃣ Benchmark Models

**Single comparison** (2 models, 1 video):
```bash
python run_benchmark.py quick
```

**Multi-model** (3 models, 5 videos):
```bash
python run_benchmark.py multi
```

### 4️⃣ Track Custom Video
```bash
python test_yolo_deepsort.py \
    --source "your_video.mp4" \
    --output "tracked_output.mp4" \
    --conf 0.3
```

### 5️⃣ Test IITB-Corridor Videos
```bash
# List 74 surveillance videos
python test_iitb_corridor.py --list

# Test specific video
python test_iitb_corridor.py 000209

# Or use shortcut
python test_yolo_deepsort.py --source 000220
```

### 6️⃣ Action Recognition Training (NEW!)
```bash
# Complete pipeline: organize + train + test
python run_action_training.py --full-pipeline --epochs 50

# Or step by step:
python run_action_training.py --organize-only
python run_action_training.py --train-only --epochs 50
python run_action_training.py --test-only
```

### 7️⃣ Integrated Detection + Action Recognition (NEW!)
```bash
# Real-time detection + action classification
python integrated_detection_action.py --source webcam

# Test on video with action recognition
python integrated_detection_action.py --source 000209 --output result.mp4
```

## 📚 Documentation

| File | Purpose |
|------|---------|
| `BENCHMARK_GUIDE.md` | Single model comparison |
| `MULTI_MODEL_BENCHMARK.md` | Multi-model benchmark |
| `ACTION_RECOGNITION_GUIDE.md` | SlowFast action training (NEW!) |
| `TRACKING_GUIDE.md` | DeepSORT tracking details |
| `CONFIDENCE_FILTER.md` | Confidence > 0.5 filter |
| `IITB_CORRIDOR_GUIDE.md` | 74 surveillance videos |
| `scripts/README_human_detection.md` | Complete documentation |

## 🎬 Sample Videos

```bash
# List available test videos
python test_sample_videos.py --list

# Test specific video
python test_sample_videos.py 2people_meet
python test_sample_videos.py 1person_walk
python test_sample_videos.py ntu_action
python test_sample_videos.py iitb_corridor_1

# List IITB-Corridor videos (74 surveillance videos)
python test_iitb_corridor.py --list
```

## 🏁 Benchmark Commands

**Single model comparison:**
```bash
# Quick test (100 frames)
python run_benchmark.py quick

# Full test (all frames)
python run_benchmark.py full
```

**Multi-model benchmark (NEW!):**
```bash
# Compare 3 models on 5 videos
python run_benchmark.py multi

# Quick multi-model (100 frames/video)
python run_benchmark.py multi --max-frames 100

# With custom confidence
python run_benchmark.py multi --conf 0.35
```

**List all configs:**
```bash
python run_benchmark.py --list
```

## 📊 What You Get

### Tracking Output
- ✅ Bounding boxes with unique colors
- ✅ Person ID (e.g., Person #1, Person #2)
- ✅ Confidence scores
- ✅ Trajectory lines
- ✅ Real-time FPS
- ✅ Tracking statistics

### Benchmark Results
- ✅ FPS comparison
- ✅ Detection accuracy
- ✅ Confidence scores
- ✅ Improvement percentage
- ✅ JSON output

## 💡 Tips

1. **Start with webcam** to see tracking in action
2. **Use 2people_meet video** to test ID switching
3. **Run quick benchmark** to compare models
4. **Adjust confidence** (--conf) between 0.2-0.4

## 🎨 Visual Features

```
┌─────────────────────────────┐
│ Person #1 [0.87]            │  ← ID + Confidence
│ Tracked: 150f               │  ← Frames tracked
├─────────────────────────────┤
│                             │
│     ┌──────────┐            │  ← Bounding box (unique color)
│     │ Person 1 │            │
│     │    ●     │            │  ← Center point
│     └──────────┘            │
│      ●●●●●●●               │  ← Trajectory trail
│                             │
└─────────────────────────────┘

=== TRACKING STATUS ===
Frame: 150
FPS: 28.5
Active Tracks: 2
Total People Seen: 5
DeepSORT: ACTIVE ✓
```

## 🎯 Model Paths

- **YOLOv8s**: `D:\SPHAR-Dataset\models\yolov8s.pt` (NEW!)
- **YOLOv11s**: `D:\SPHAR-Dataset\models\yolo11s.pt`
- **YOLOv11s-FT**: `D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt`
- **Output**: `D:\SPHAR-Dataset\output\`

## 📈 Expected Performance

| Metric | Pretrained | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| Detection Rate | ~78% | ~93% | +15% |
| Avg Confidence | 0.65 | 0.74 | +14% |
| FPS | ~28 | ~30 | +7% |

## 🆘 Quick Troubleshooting

### No detections or very few?
**Note**: Script now filters confidence > 0.5 for high quality tracking!

```bash
# If you want more detections, edit test_yolo_deepsort.py line ~116
# Change: if conf <= 0.5 to if conf <= 0.3

# Or lower YOLO confidence threshold
python test_yolo_deepsort.py --source webcam --conf 0.2
```

### DeepSORT not working?
```bash
pip install deep-sort-realtime
```

### Want simple detection (no tracking)?
```bash
python quick_test_model.py --source webcam
```

---

**Ready to start?** Try: `python test_sample_videos.py 2people_meet` 🚀
