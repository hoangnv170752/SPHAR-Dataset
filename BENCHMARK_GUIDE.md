# 🏁 Model Benchmark Guide

## 📋 Mục đích

So sánh hiệu năng giữa:
- **Model 1**: `yolo11s.pt` (Pretrained - COCO dataset)
- **Model 2**: `yolo11s-detect.pt` (Fine-tuned - SPHAR dataset)

## 🚀 Quick Start

### Chạy benchmark nhanh (100 frames)
```bash
cd D:\SPHAR-Dataset\scripts
python run_benchmark.py quick
```

### Chạy benchmark đầy đủ
```bash
python run_benchmark.py full
```

### Xem danh sách cấu hình
```bash
python run_benchmark.py --list
```

## 📊 Các cấu hình có sẵn

| Config | Video | Frames | Mô tả |
|--------|-------|--------|-------|
| **quick** | casia_angleview_p01_walk_a1.mp4 | 100 | Test nhanh (1 người) |
| **2people** | casia_angleview_p01p02_meettogether_a1.mp4 | Full | 2 người gặp nhau |
| **ntu** | S001C001P001R001A001_rgb.avi | Full | NTU action dataset |
| **full** | casia_angleview_p01p02_meettogether_a1.mp4 | Full | Test đầy đủ |

## 📈 Metrics được đo

### 1. Performance Metrics
- **Average FPS**: Frames per second
- **Inference Time**: Thời gian xử lý mỗi frame (ms)

### 2. Detection Metrics
- **Total Detections**: Tổng số người phát hiện
- **Avg Detections/Frame**: Trung bình số người/frame
- **Detection Rate**: % frames có phát hiện người

### 3. Confidence Metrics
- **Average Confidence**: Độ tin cậy trung bình
- **Min/Max Confidence**: Độ tin cậy min/max

## 🎯 Cách sử dụng nâng cao

### Test với video tùy chỉnh
```bash
python benchmark_models.py \
    --video "path/to/video.mp4" \
    --conf 0.3 \
    --output "results.json"
```

### Test với số frames giới hạn
```bash
python benchmark_models.py \
    --video "video.mp4" \
    --max-frames 200 \
    --conf 0.25
```

### So sánh 2 model tùy chỉnh
```bash
python benchmark_models.py \
    --model1 "model1.pt" \
    --model2 "model2.pt" \
    --video "video.mp4"
```

## 📊 Kết quả mẫu

```
================================================================================
📊 BENCHMARK COMPARISON
================================================================================

Metric                         Model 1 (Pretrained)      Model 2 (Fine-tuned)      Improvement    
-----------------------------------------------------------------------------------------------
Average FPS                    28.50                     30.20                     🟢 +6.0%
Avg Inference Time (ms)        35.1                      33.1                      🟢 +5.7%
Total Detections               185                       238                       🟢 +28.6%
Avg Detections/Frame           0.85                      1.10                      🟢 +29.4%
Detection Rate (%)             78.3%                     92.6%                     🟢 +18.3%
Avg Confidence                 0.652                     0.743                     🟢 +14.0%
Min Confidence                 0.312                     0.421                     🟢 +34.9%
Frames w/ Detections           170                       201                       🟢 +18.2%
================================================================================

📋 SUMMARY:
   🚀 Speed: Fine-tuned model is 6.0% faster
   🎯 Detection: Fine-tuned model detects 29.4% more people
   💪 Confidence: Fine-tuned model is 14.0% more confident

🏆 OVERALL SCORE: +23.7%
   ✅ Fine-tuned model shows SIGNIFICANT improvement!
================================================================================
```

## 📁 Output Files

### JSON Results
```json
{
  "timestamp": "2025-10-15 19:30:00",
  "model1": {
    "model_name": "Pretrained",
    "total_frames": 217,
    "total_detections": 185,
    "avg_fps": 28.50,
    ...
  },
  "model2": {
    "model_name": "Fine-tuned",
    "total_frames": 217,
    "total_detections": 238,
    "avg_fps": 30.20,
    ...
  },
  "improvements": {
    "avg_fps": 6.0,
    "avg_detections_per_frame": 29.4,
    ...
  }
}
```

Lưu tại: `D:\SPHAR-Dataset\benchmark_<config>.json`

## 💡 Tips

1. **Quick test trước**: Dùng `quick` config để test nhanh
2. **Full test sau**: Dùng `full` hoặc `2people` cho kết quả chính xác
3. **Confidence threshold**: 
   - 0.25 (default) - balanced
   - 0.3 - fewer false positives
   - 0.2 - more detections
4. **GPU recommended**: Benchmark nhanh hơn nhiều với GPU

## 🎓 Giải thích metrics

### Detection Rate
- **>90%**: Excellent - Model phát hiện người trong hầu hết frames
- **70-90%**: Good - Model hoạt động tốt
- **<70%**: Poor - Model bỏ sót nhiều

### Average Confidence
- **>0.7**: High confidence - Model rất chắc chắn
- **0.5-0.7**: Medium confidence - Model khá tự tin
- **<0.5**: Low confidence - Model không chắc chắn

### Improvement Percentage
- **>20%**: Significant improvement
- **10-20%**: Good improvement
- **0-10%**: Marginal improvement
- **<0%**: No improvement (need more training)

## 🐛 Troubleshooting

### Model not found
```bash
# Kiểm tra model paths
ls D:\SPHAR-Dataset\models\yolo11s.pt
ls D:\SPHAR-Dataset\models\finetuned\yolo11s-detect.pt
```

### Video not found
```bash
# Kiểm tra video path
python run_benchmark.py --list
```

### Out of memory
```bash
# Giảm số frames
python run_benchmark.py quick  # chỉ 100 frames
```

## 📚 Files

- `benchmark_models.py`: Main benchmark script
- `run_benchmark.py`: Quick runner với presets
- `benchmark_<config>.json`: Results output

## 🎉 Example Commands

```bash
# Test nhanh
python run_benchmark.py quick

# Test đầy đủ với 2 người
python run_benchmark.py 2people

# Test với confidence cao hơn
python run_benchmark.py full --conf 0.35

# Custom benchmark
python benchmark_models.py \
    --video "my_video.mp4" \
    --max-frames 500 \
    --conf 0.3 \
    --output "my_results.json"
```

---

**Models:**
- Pretrained: `yolo11s.pt` (COCO 80 classes)
- Fine-tuned: `yolo11s-detect.pt` (SPHAR human detection)

**GPU**: NVIDIA GeForce GTX 1660 SUPER (6GB)
