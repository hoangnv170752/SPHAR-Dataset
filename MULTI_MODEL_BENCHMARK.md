# 🏆 Multi-Model Benchmark Guide

## 📋 Overview

So sánh **3 models** trên **5 videos** test:

### 🤖 Models
1. **YOLOv8s** - YOLOv8 small (pretrained COCO)
2. **YOLOv11s** - YOLOv11 small (pretrained COCO)
3. **YOLOv11s-FT** - YOLOv11 small fine-tuned trên SPHAR dataset

### 📹 Test Videos
1. **1person_walk** - 1 người đi bộ
2. **2people_meet** - 2 người gặp nhau
3. **2people_follow** - 2 người theo nhau
4. **2people_overtake** - 2 người vượt nhau
5. **ntu_action** - NTU action dataset

## 🚀 Quick Start

### Chạy benchmark đầy đủ
```bash
cd D:\SPHAR-Dataset\scripts
python run_benchmark.py multi
```

### Chạy nhanh (100 frames/video)
```bash
python run_benchmark.py multi --max-frames 100
```

### Với confidence threshold khác
```bash
python run_benchmark.py multi --conf 0.35
```

### Hoặc chạy trực tiếp
```bash
python benchmark_multi_models.py --conf 0.25
```

## 📊 Output Format

### 1. Comparison Tables

```
📈 Average FPS
--------------------------------------------------------------------------------
              YOLOv8s  YOLOv11s  YOLOv11s-FT
1person_walk     28.5      29.2         30.8
2people_meet     27.3      28.1         29.5
2people_follow   26.8      27.9         29.2
...

📈 Detections/Frame
--------------------------------------------------------------------------------
              YOLOv8s  YOLOv11s  YOLOv11s-FT
1person_walk     0.95      0.92         1.05
2people_meet     1.85      1.78         2.12
...
```

### 2. Overall Performance

```
🏆 OVERALL MODEL PERFORMANCE
--------------------------------------------------------------------------------
              avg_fps  avg_detections_per_frame  detection_rate  avg_confidence
YOLOv8s          27.5                      1.25            82.3           0.658
YOLOv11s         28.3                      1.18            79.8           0.671
YOLOv11s-FT      29.8                      1.42            89.5           0.745
```

### 3. Composite Scores

```
🎯 COMPOSITE SCORES (0-100)
--------------------------------------------------------------------------------
🥇 YOLOv11s-FT      : 100.0
🥈 YOLOv8s          : 67.3
🥉 YOLOv11s         : 58.9
```

## 📈 Metrics Explained

### Performance Metrics
- **avg_fps**: Tốc độ xử lý (frames/second)
- **avg_inference_time_ms**: Thời gian inference (milliseconds)

### Detection Metrics
- **avg_detections_per_frame**: Số người phát hiện trung bình/frame
- **detection_rate**: % frames có phát hiện người
- **total_detections**: Tổng số detections

### Quality Metrics
- **avg_confidence**: Độ tin cậy trung bình
- **min/max_confidence**: Độ tin cậy min/max

### Composite Score
Điểm tổng hợp (0-100) được tính từ:
- Speed (20%): FPS
- Detection (30%): Detections per frame
- Coverage (30%): Detection rate
- Quality (20%): Confidence

## 📁 Output Files

### JSON Result
Lưu tại: `D:\SPHAR-Dataset\multi_model_benchmark.json`

```json
{
  "timestamp": "2025-10-15 19:30:00",
  "device": "cuda",
  "models": ["YOLOv8s", "YOLOv11s", "YOLOv11s-FT"],
  "results": [
    {
      "model_name": "YOLOv11s-FT",
      "video_id": "2people_meet",
      "total_frames": 217,
      "avg_fps": 29.5,
      "avg_detections_per_frame": 2.12,
      "detection_rate": 94.3,
      "avg_confidence": 0.745
    },
    ...
  ]
}
```

## 🎯 Use Cases

### 1. Chọn model tốt nhất
```bash
python run_benchmark.py multi
# Xem composite scores để chọn model
```

### 2. So sánh speed vs accuracy
```bash
# Lower confidence = more detections but slower
python run_benchmark.py multi --conf 0.2

# Higher confidence = fewer detections but faster
python run_benchmark.py multi --conf 0.4
```

### 3. Quick test trên subset
```bash
python run_benchmark.py multi --max-frames 50
```

## 💡 Tips

1. **Full benchmark**: Chạy không giới hạn frames để kết quả chính xác nhất
2. **Quick test**: Dùng `--max-frames 100` để test nhanh
3. **Confidence**: 0.25-0.35 là khoảng tốt cho most cases
4. **GPU required**: Benchmark sẽ rất chậm trên CPU

## 📊 Expected Results

Based on initial tests:

| Model | Speed | Detection | Quality | Overall |
|-------|-------|-----------|---------|---------|
| YOLOv8s | 🟡 Medium | 🟢 Good | 🟢 Good | 🟢 Good |
| YOLOv11s | 🟢 Fast | 🟡 Medium | 🟢 Good | 🟡 Medium |
| YOLOv11s-FT | 🟢 Fast | 🟢 Excellent | 🟢 Excellent | 🥇 Best |

**Winner**: YOLOv11s-FT (fine-tuned) should win overall

## 🆘 Troubleshooting

### Missing model
```bash
# Check models exist
ls D:\SPHAR-Dataset\models\yolov8s.pt
ls D:\SPHAR-Dataset\models\yolo11s.pt
ls D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt
```

### pandas not found
```bash
pip install pandas tqdm
# Or install all requirements
pip install -r requirements_tracking.txt
```

### Out of memory
```bash
# Use fewer frames
python run_benchmark.py multi --max-frames 100
```

### Video not found
Check that videos exist:
```bash
ls D:\SPHAR-Dataset\videos\walking\*.mp4
ls D:\SPHAR-Dataset\videos\NTU\A001\*.avi
```

## 🎓 Understanding Results

### High composite score (>80)
✅ Model performs well across all metrics

### Medium score (50-80)
⚠️ Model good in some areas, needs improvement in others

### Low score (<50)
❌ Model needs significant improvement or more training

### Fine-tuned advantage
Fine-tuned model should show:
- ✅ Higher detection rate (better recall)
- ✅ Higher confidence (more certain)
- ✅ More detections per frame (fewer misses)
- ⚠️ Similar or slightly better speed

## 📚 Files

- `benchmark_multi_models.py`: Main benchmark script
- `run_benchmark.py`: Quick runner with multi support
- `multi_model_benchmark.json`: Output results

## 🎉 Example Commands

```bash
# Full benchmark (recommended)
python run_benchmark.py multi

# Quick test
python run_benchmark.py multi --max-frames 100

# High confidence
python run_benchmark.py multi --conf 0.35

# Low confidence (more detections)
python run_benchmark.py multi --conf 0.2

# See available configs
python run_benchmark.py --list
```

---

**Models Tested:**
- YOLOv8s (pretrained)
- YOLOv11s (pretrained)
- YOLOv11s-FT (fine-tuned on SPHAR)

**Test Dataset**: 5 videos from SPHAR-Dataset  
**GPU**: NVIDIA GeForce GTX 1660 SUPER (6GB)
