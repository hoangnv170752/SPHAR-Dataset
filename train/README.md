# YOLO v11 Human Detection Fine-tuning

This directory contains scripts for fine-tuning YOLO v11 for human detection using the abnormal detection dataset.

## Files

- `yolo_human_detection_trainer.py` - Main training class with full pipeline
- `train_human_detection.py` - Simple script to start training with default parameters
- `README.md` - This documentation file

## Quick Start

### 1. Install Dependencies

Make sure you have the required packages installed:

```bash
pip install -r ../requirements_yolo.txt
```

### 2. Run Training

#### Option A: Simple Training (Recommended for beginners)
```bash
python train_human_detection.py
```

#### Option B: Custom Training with Parameters
```bash
python yolo_human_detection_trainer.py --dataset "D:\abnormal_detection_dataset" --model "D:\SPHAR-Dataset\models\yolo11s.pt" --output "D:\SPHAR-Dataset\train\human_detection_results" --epochs 50 --batch-size 8 --frames-per-video 3
```

## Parameters

- `--dataset`: Path to the abnormal detection dataset (default: `D:\abnormal_detection_dataset`)
- `--model`: Path to pre-trained YOLO model (default: `D:\SPHAR-Dataset\models\yolo11s.pt`)
- `--output`: Output directory for results (default: `D:\SPHAR-Dataset\train\human_detection_results`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 16)
- `--frames-per-video`: Number of frames to extract per video (default: 5)

## What the Script Does

1. **Dataset Preparation**: Converts video dataset to YOLO format by extracting frames
2. **Label Creation**: Creates bounding box labels for human detection (simplified approach)
3. **YOLO Configuration**: Creates dataset.yaml file for YOLO training
4. **Model Training**: Fine-tunes YOLO v11 on the extracted frames
5. **Evaluation**: Validates the trained model on test data
6. **Results Saving**: Saves training metrics, model weights, and configuration

## Output Structure

After training, you'll find:

```
human_detection_results/
├── datasets/
│   └── human_detection_yolo/
│       ├── images/
│       ├── labels/
│       └── dataset.yaml
├── runs/
│   └── human_detection/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       ├── results.png
│       └── confusion_matrix.png
├── models/
└── training_info.json
```

## Important Notes

### Current Limitations

1. **Label Quality**: The script creates simplified bounding boxes assuming humans are present in the center of each frame. For production use, you should:
   - Use a pre-trained human detection model to generate better labels
   - Manually annotate a subset of frames
   - Use active learning techniques

2. **Frame Extraction**: Currently extracts frames uniformly from videos. You might want to:
   - Use motion detection to select more informative frames
   - Extract frames at key moments (scene changes, etc.)

### Recommendations for Better Results

1. **Improve Labels**: 
   ```python
   # Use a pre-trained model like YOLO or MediaPipe to detect humans
   # and create more accurate bounding boxes
   ```

2. **Data Augmentation**: The script includes various augmentation parameters that you can tune

3. **Hyperparameter Tuning**: Adjust learning rates, batch sizes, and other parameters based on your hardware

4. **Transfer Learning**: Start with a model pre-trained on COCO dataset (which includes person class)

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support recommended
- **RAM**: At least 8GB system RAM
- **VRAM**: At least 4GB GPU memory for batch_size=8
- **Storage**: Several GB for extracted frames and model checkpoints

## Monitoring Training

During training, you can monitor:

- Loss curves in the results plots
- Validation metrics
- TensorBoard logs (if enabled)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch_size or image size
2. **Slow Training**: Reduce frames_per_video or use GPU
3. **Poor Results**: Improve label quality or increase dataset size

### Performance Tips

- Use SSD storage for faster data loading
- Increase `workers` parameter if you have multiple CPU cores
- Use mixed precision training for faster training on modern GPUs

## Next Steps

After training, you can:

1. **Test the Model**: Use the trained model for inference on new videos
2. **Deploy**: Integrate the model into your application
3. **Improve**: Collect more data and retrain with better labels

## Citation

If you use this training pipeline, please cite the original YOLO and dataset papers.
