#!/usr/bin/env python3
"""
Create Real Paper Images from Dataset
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random
from ultralytics import YOLO

class RealPaperImageGenerator:
    """Generate real images from dataset for paper"""
    
    def __init__(self, output_dir="paper_real_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load YOLO from models folder
        yolo_paths = [
            Path(r'D:\SPHAR-Dataset\models\yolo11n.pt'),
            Path(r'D:\SPHAR-Dataset\models\yolov8n.pt'),
            Path('yolo11n.pt')
        ]
        
        self.yolo_model = None
        for yolo_path in yolo_paths:
            try:
                if yolo_path.exists():
                    self.yolo_model = YOLO(str(yolo_path))
                    print(f"✅ YOLO loaded: {yolo_path.name}")
                    break
            except:
                continue
        
        if self.yolo_model is None:
            print("⚠️ No YOLO model found, using manual detection")
        
        # Professional colors
        self.colors = {
            'person_box': '#FF4444',      # Red for person
            'action_box': '#00CC88',      # Green for action
            'text': '#2C3E50',            # Dark text
            'alert': '#E74C3C',           # Red alert
            'normal': '#27AE60',          # Green normal
            'background': '#F8F9FA'       # Light background
        }
    
    def detect_person_in_frame(self, frame):
        """Detect person in frame using YOLO with lower threshold"""
        if self.yolo_model is None:
            return []
        
        # Run YOLO with lower confidence threshold
        results = self.yolo_model(frame, conf=0.1, verbose=False)  # Lower threshold
        persons = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if detection is a person (class 0) with lower threshold
                    if int(box.cls) == 0 and float(box.conf) > 0.1:  # Much lower threshold
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf)
                        persons.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence
                        })
                        print(f"   🔍 Person detected: conf={confidence:.3f}, bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
        
        return persons
    
    def create_annotated_frame(self, frame, persons, has_person=True):
        """Create professionally annotated frame based on actual YOLO detection"""
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display frame
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Use actual detection results
        actual_has_person = len(persons) > 0
        
        if actual_has_person:
            # Draw person detection boxes
            for person in persons:
                x1, y1, x2, y2 = person['bbox']
                conf = person['confidence']
                
                # Person bounding box
                person_rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                              linewidth=3, edgecolor=self.colors['person_box'],
                                              facecolor='none', linestyle='-')
                ax.add_patch(person_rect)
                
                # Person label
                ax.text(x1, y1-10, f'Person: {conf:.2f}', 
                       fontsize=11, fontweight='bold', color='white',
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor=self.colors['person_box'], alpha=0.8))
        
        # System status based on actual detection
        if actual_has_person:
            status_text = 'ALERT: Person Detected'
            status_color = self.colors['alert']
            bg_color = '#FADBD8'
        else:
            status_text = 'MONITORING: No Person Detected'
            status_color = self.colors['normal']
            bg_color = '#D5F4E6'
        
        ax.text(20, 40, status_text, 
               fontsize=14, fontweight='bold', color=status_color,
               bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color, alpha=0.9))
        
        # Camera info
        ax.text(frame.shape[1]-200, 40, 'Camera: Indoor_01', 
               fontsize=10, color=self.colors['text'],
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        ax.text(frame.shape[1]-200, 65, 'FPS: 30', 
               fontsize=10, color=self.colors['text'],
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Title based on actual detection
        if actual_has_person:
            title = 'Human Action Recognition System - Person Detected'
        else:
            title = 'Human Action Recognition System - No Person Detected'
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color=self.colors['text'])
        
        # Remove axes for clean look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        return fig
    
    def get_sample_frames(self):
        """Get real indoor sample frames and detect persons"""
        
        # Look for indoor images first
        img_dirs = [
            Path(r'D:\SPHAR-Dataset\action_recognition_images\train\fall'),
            Path(r'D:\SPHAR-Dataset\action_recognition_images\train\normal'), 
            Path(r'D:\SPHAR-Dataset\action_recognition_images\val\fall'),
            Path(r'D:\SPHAR-Dataset\action_recognition_images\val\normal'),
            Path(r'D:\SPHAR-Dataset\train\human_detection_results\datasets\images'),
            Path(r'D:\SPHAR-Dataset\videos\falling'),
            Path(r'D:\SPHAR-Dataset\videos\normal')
        ]
        
        person_frame = None
        no_person_frame = None
        
        # Find images with and without persons
        for img_dir in img_dirs:
            if img_dir.exists():
                img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
                if img_files:
                    print(f"📸 Found {len(img_files)} images in {img_dir}")
                    
                    # Prioritize indoor scenes - look for specific patterns
                    indoor_keywords = ['indoor', 'room', 'office', 'home', 'kitchen', 'living']
                    
                    # First try to find indoor images
                    indoor_files = []
                    for img_file in img_files:
                        if any(keyword in str(img_file).lower() for keyword in indoor_keywords):
                            indoor_files.append(img_file)
                    
                    # If no indoor files found, use all files
                    if not indoor_files:
                        indoor_files = img_files
                    
                    # Randomly select 2 different images each time
                    if len(indoor_files) >= 2:
                        selected_files = random.sample(indoor_files, 2)
                        
                        # First image
                        frame1 = cv2.imread(str(selected_files[0]))
                        if frame1 is not None:
                            frame1 = cv2.resize(frame1, (640, 480))
                            persons1 = self.detect_person_in_frame(frame1)
                            action1 = "FALL" if 'fall' in str(img_dir).lower() else "NORMAL"
                            
                            if person_frame is None:
                                person_frame = (frame1, action1)
                                print(f"✅ Image 1: {selected_files[0].name} ({len(persons1)} persons) - {action1}")
                        
                        # Second image  
                        frame2 = cv2.imread(str(selected_files[1]))
                        if frame2 is not None:
                            frame2 = cv2.resize(frame2, (640, 480))
                            persons2 = self.detect_person_in_frame(frame2)
                            action2 = "FALL" if 'fall' in str(img_dir).lower() else "NORMAL"
                            
                            if no_person_frame is None:
                                no_person_frame = (frame2, action2)
                                print(f"✅ Image 2: {selected_files[1].name} ({len(persons2)} persons) - {action2}")
                        
                        # If we got both images, break
                        if person_frame and no_person_frame:
                            break
                    
                    if person_frame and no_person_frame:
                        break
        
        # If still no suitable frames found, try video frames with randomization
        if person_frame is None or no_person_frame is None:
            video_dirs = [Path(r'D:\SPHAR-Dataset\videos')]
            
            all_videos = []
            for video_dir in video_dirs:
                if video_dir.exists():
                    for action_dir in video_dir.iterdir():
                        if action_dir.is_dir():
                            videos = list(action_dir.glob('*.mp4'))
                            for video in videos:
                                all_videos.append((video, action_dir.name.upper()))
            
            if all_videos:
                print(f"📹 Found {len(all_videos)} videos, selecting random frames...")
                
                # Shuffle and select random videos each time
                random.shuffle(all_videos)
                
                # Try to find frames with different detection results
                attempts = 0
                for video_path, action in all_videos:
                    if attempts >= 50:  # Limit attempts
                        break
                        
                    cap = cv2.VideoCapture(str(video_path))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if total_frames > 50:
                        # Get random frame numbers each time
                        random_frames = random.sample(range(20, total_frames-20), min(10, total_frames-40))
                        
                        for frame_num in random_frames:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                            ret, frame = cap.read()
                            if not ret:
                                continue
                            
                            frame = cv2.resize(frame, (640, 480))
                            persons = self.detect_person_in_frame(frame)
                            
                            # Prioritize finding one with person and one without
                            if len(persons) > 0 and person_frame is None:
                                person_frame = (frame, action)
                                print(f"✅ Person frame: {video_path.name} frame {frame_num} ({len(persons)} persons)")
                            elif len(persons) == 0 and no_person_frame is None:
                                no_person_frame = (frame, "EMPTY")
                                print(f"✅ Empty frame: {video_path.name} frame {frame_num}")
                            
                            attempts += 1
                            if person_frame and no_person_frame:
                                break
                        
                        if person_frame and no_person_frame:
                            break
                    
                    cap.release()
        
        # Last resort: create synthetic frames
        if person_frame is None:
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
            cv2.rectangle(frame, (250, 150), (350, 350), (80, 80, 80), -1)
            cv2.circle(frame, (300, 120), 30, (80, 80, 80), -1)
            person_frame = (frame, "SYNTHETIC")
            print("⚠️ Created synthetic person frame")
        
        if no_person_frame is None:
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 220
            cv2.rectangle(frame, (50, 400), (590, 470), (180, 180, 180), -1)
            no_person_frame = (frame, "EMPTY")
            print("⚠️ Created synthetic empty frame")
        
        return person_frame, no_person_frame
    
    def generate_paper_images(self):
        """Generate real paper images"""
        
        print("="*60)
        print("📸 GENERATING REAL PAPER IMAGES")
        print("="*60)
        
        # Get sample frames
        person_frame_data, no_person_frame_data = self.get_sample_frames()
        
        if person_frame_data is None:
            print("❌ Could not get sample frames")
            return
        
        person_frame, person_action = person_frame_data
        no_person_frame, _ = no_person_frame_data
        
        # Detect persons in frames
        print("🔍 Detecting persons in frames...")
        persons_in_frame1 = self.detect_person_in_frame(person_frame)
        persons_in_frame2 = self.detect_person_in_frame(no_person_frame)
        
        print(f"   Frame 1: {len(persons_in_frame1)} persons detected")
        print(f"   Frame 2: {len(persons_in_frame2)} persons detected")
        
        # Create annotated images based on actual detection
        print("🎨 Creating annotated images...")
        
        # Image 1: First frame (whatever detection result)
        fig1 = self.create_annotated_frame(person_frame, persons_in_frame1)
        
        # Save image 1
        if len(persons_in_frame1) > 0:
            output_path1 = self.output_dir / 'real_scene_with_person.png'
        else:
            output_path1 = self.output_dir / 'real_scene_without_person_1.png'
        
        fig1.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig1)
        print(f"✅ Saved: {output_path1}")
        
        # Image 2: Second frame (whatever detection result)
        fig2 = self.create_annotated_frame(no_person_frame, persons_in_frame2)
        
        # Save image 2
        if len(persons_in_frame2) > 0:
            output_path2 = self.output_dir / 'real_scene_with_person_2.png'
        else:
            output_path2 = self.output_dir / 'real_scene_without_person.png'
            
        fig2.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f"✅ Saved: {output_path2}")
        
        print("\n✅ Images generated successfully!")
        print(f"📁 Location: {self.output_dir}")
        print("📸 Files:")
        print("   - real_scene_with_person.png")
        print("   - real_scene_without_person.png")
        
        # Show using matplotlib instead of cv2
        print("\n📊 Displaying images...")
        
        # Load and display images
        import matplotlib.image as mpimg
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Image 1
        img1 = mpimg.imread(str(output_path1))
        ax1.imshow(img1)
        ax1.set_title('Scene with Person Detection', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Image 2  
        img2 = mpimg.imread(str(output_path2))
        ax2.imshow(img2)
        ax2.set_title('Scene without Person Detection', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("🎯 Ready for paper!")

def main():
    """Main function"""
    
    # Create output directory
    output_dir = Path(r'D:\SPHAR-Dataset\paper_real_images')
    
    # Generate real images
    generator = RealPaperImageGenerator(output_dir)
    generator.generate_paper_images()
    
    print("\n🎯 Ready for paper!")
    print("📊 High resolution (300 DPI)")
    print("📄 Professional annotations")
    print("🖼️ Real dataset frames")

if __name__ == "__main__":
    main()
