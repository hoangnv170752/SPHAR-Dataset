"""
Simple script to start YOLO v11 human detection training
"""

from yolo_human_detection_trainer import YOLOHumanDetectionTrainer

def main():
    # Configuration
    dataset_path = r'D:\abnormal_detection_dataset'
    model_path = r'D:\SPHAR-Dataset\models\yolo11s.pt'
    output_dir = r'D:\SPHAR-Dataset\train\human_detection_results'
    
    # Training parameters
    epochs = 50  # Reduced for initial testing
    batch_size = 8  # Reduced for systems with limited GPU memory
    frames_per_video = 3  # Extract 3 frames per video
    
    print("Starting YOLO v11 Human Detection Training")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Frames per video: {frames_per_video}")
    
    # Create trainer
    trainer = YOLOHumanDetectionTrainer(
        dataset_path=dataset_path,
        model_path=model_path,
        output_dir=output_dir
    )
    
    # Run training
    try:
        results, eval_results = trainer.run_full_pipeline(
            epochs=epochs,
            batch_size=batch_size,
            frames_per_video=frames_per_video
        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
