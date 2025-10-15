# Human Detection Pipeline for SPHAR Dataset

## Tổng quan

Pipeline này giải quyết vấn đề phân loại frame có người/không có người trong dataset SPHAR thay vì phân loại theo hành vi (normal/abnormal). Pipeline bao gồm 3 bước chính:

1. **Tạo dataset**: Trích xuất frame từ video và phân loại có người/không có người
2. **Training**: Huấn luyện mô hình YOLO để detect human
3. **Classification**: Sử dụng mô hình đã train để phân loại frame mới

## Cài đặt Requirements

```bash
pip install opencv-python ultralytics pyyaml tqdm numpy
```

## Cấu trúc Files

```
scripts/
├── create_human_detection_dataset.py    # Tạo dataset từ videos
├── train_human_detection.py             # Training YOLO model
├── classify_frames_with_human.py        # Phân loại frame với model đã train
├── human_detection_pipeline.py          # Pipeline hoàn chỉnh
└── README_human_detection.md            # File hướng dẫn này
```

## Cách sử dụng

### Phương pháp 1: Chạy pipeline hoàn chỉnh (Khuyến nghị)

```bash
# Chạy toàn bộ pipeline từ đầu đến cuối
python human_detection_pipeline.py --videos-dir "D:\SPHAR-Dataset\videos"

# Với tham số tùy chỉnh
python human_detection_pipeline.py \
    --videos-dir "D:\SPHAR-Dataset\videos" \
    --frame-interval 15 \
    --epochs 150 \
    --batch-size 32
```

### Phương pháp 2: Chạy từng bước riêng biệt

#### Bước 1: Tạo dataset

```bash
python create_human_detection_dataset.py \
    --source "D:\SPHAR-Dataset\videos" \
    --output "D:\SPHAR-Dataset\train\human_detection_dataset" \
    --frame-interval 30
```

**Tham số:**
- `--source`: Thư mục chứa videos gốc
- `--output`: Thư mục output cho dataset
- `--frame-interval`: Trích xuất 1 frame mỗi N frames (default: 30)
- `--train-ratio`: Tỷ lệ dữ liệu train (default: 0.7)
- `--val-ratio`: Tỷ lệ dữ liệu validation (default: 0.15)
- `--test-ratio`: Tỷ lệ dữ liệu test (default: 0.15)

#### Bước 2: Training model

```bash
python train_human_detection.py \
    --dataset "D:\SPHAR-Dataset\train\human_detection_dataset" \
    --model yolo11s.pt \
    --epochs 100 \
    --batch 16
```

**Tham số:**
- `--dataset`: Đường dẫn đến dataset đã tạo
- `--model`: Mô hình YOLO base (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt)
- `--epochs`: Số epoch training (default: 100)
- `--imgsz`: Kích thước ảnh (default: 640)
- `--batch`: Batch size (default: 16)

#### Bước 3: Phân loại frame mới

```bash
# Phân loại từ videos
python classify_frames_with_human.py \
    --model "path/to/trained/model.pt" \
    --input "D:\SPHAR-Dataset\videos" \
    --output "D:\classified_frames" \
    --mode videos

# Phân loại từ thư mục ảnh
python classify_frames_with_human.py \
    --model "path/to/trained/model.pt" \
    --input "D:\images_folder" \
    --output "D:\classified_frames" \
    --mode images
```

**Tham số:**
- `--model`: Đường dẫn đến model đã train
- `--input`: Thư mục input (videos hoặc images)
- `--output`: Thư mục output
- `--mode`: Chế độ xử lý (videos/images)
- `--confidence`: Ngưỡng confidence (default: 0.5)
- `--frame-interval`: Interval trích xuất frame từ video (default: 30)

## Output Structure

### Dataset sau khi tạo:
```
human_detection_dataset/
├── images/
│   ├── train/          # Ảnh training
│   ├── val/            # Ảnh validation  
│   └── test/           # Ảnh test
├── labels/
│   ├── train/          # Labels YOLO format
│   ├── val/            # Labels validation
│   └── test/           # Labels test
├── annotations/        # Metadata và annotations
├── dataset.yaml        # Config file cho YOLO
└── README.md
```

### Kết quả training:
```
training_results/
├── human_detection_training/
│   ├── weights/
│   │   ├── best.pt     # Model tốt nhất
│   │   └── last.pt     # Model cuối cùng
│   └── results.png     # Biểu đồ training
├── best_human_detection_model.pt  # Model copy
└── training_summary.json          # Tóm tắt kết quả
```

### Kết quả classification:
```
classified_frames/
├── frames/
│   ├── with_human/     # Frame có người
│   └── without_human/  # Frame không có người
├── annotations/        # Metadata
└── classification_report.txt
```

## Tham số quan trọng

### Frame Interval
- **15-20**: Chất lượng cao, nhiều frame, training lâu
- **30**: Cân bằng (khuyến nghị)
- **60+**: Nhanh nhưng có thể mất thông tin

### Model Size
- **yolo11n.pt**: Nhanh, nhẹ, độ chính xác thấp hơn
- **yolo11s.pt**: Cân bằng (khuyến nghị)
- **yolo11m.pt**: Chậm hơn, độ chính xác cao hơn
- **yolo11l.pt**: Rất chậm, độ chính xác cao nhất

### Batch Size
- **8-16**: Cho GPU 4-8GB
- **32-64**: Cho GPU 16GB+
- **4-8**: Cho CPU hoặc GPU yếu

## Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory**
   - Giảm batch size: `--batch 8`
   - Giảm image size: `--imgsz 416`

2. **Không tìm thấy video files**
   - Kiểm tra đường dẫn videos directory
   - Đảm bảo có file .mp4, .avi, .mov trong thư mục

3. **Model không load được**
   - Kiểm tra đường dẫn model file
   - Đảm bảo đã cài đặt ultralytics

4. **Dataset trống**
   - Kiểm tra videos có đúng format không
   - Tăng frame-interval nếu video quá ngắn

### Performance Tips:

1. **Tăng tốc training:**
   ```bash
   --batch 32 --imgsz 416 --epochs 50
   ```

2. **Chất lượng cao:**
   ```bash
   --model yolo11m.pt --epochs 200 --imgsz 640
   ```

3. **Test nhanh:**
   ```bash
   --frame-interval 60 --epochs 10 --batch 8
   ```

## Ví dụ hoàn chỉnh

```bash
# 1. Tạo dataset với frame interval 20
python create_human_detection_dataset.py \
    --source "D:\SPHAR-Dataset\videos" \
    --output "D:\SPHAR-Dataset\train\human_detection_dataset" \
    --frame-interval 20

# 2. Training với YOLOv11s, 150 epochs
python train_human_detection.py \
    --dataset "D:\SPHAR-Dataset\train\human_detection_dataset" \
    --model yolo11s.pt \
    --epochs 150 \
    --batch 24

# 3. Phân loại frame mới
python classify_frames_with_human.py \
    --model "D:\SPHAR-Dataset\train\human_detection_dataset\training_results\best_human_detection_model.pt" \
    --input "D:\new_videos" \
    --output "D:\classified_results" \
    --confidence 0.6
```

## Kết quả mong đợi

Sau khi hoàn thành pipeline, bạn sẽ có:

1. **Dataset chuẩn YOLO format** cho human detection
2. **Model đã train** có thể detect human với độ chính xác cao
3. **Frame đã phân loại** theo có người/không có người
4. **Reports và statistics** chi tiết về quá trình

Model tốt thường đạt:
- **mAP50**: > 0.8
- **Precision**: > 0.85  
- **Recall**: > 0.80

## Liên hệ

Nếu gặp vấn đề, hãy kiểm tra:
1. Log files trong thư mục output
2. Requirements đã cài đủ chưa
3. Đường dẫn files có đúng không
4. GPU memory có đủ không
