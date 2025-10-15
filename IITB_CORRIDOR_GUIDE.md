# 📹 IITB-Corridor Video Testing Guide

## 📋 Overview

IITB-Corridor dataset contains **74 surveillance videos** (IDs: 000209 to 000358).

Location: `D:\SPHAR-Dataset\videos\IITB-Corridor\`

Each video is in format: `000XXX/000XXX.avi`

## 🚀 Quick Start

### 1️⃣ List All Available Videos
```bash
cd D:\SPHAR-Dataset\scripts
python test_iitb_corridor.py --list
```

### 2️⃣ Test Specific Video
```bash
# Method 1: Using helper script
python test_iitb_corridor.py 000209

# Method 2: Using main script with ID
python test_yolo_deepsort.py --source 000209

# Method 3: Using main script with prefix
python test_yolo_deepsort.py --source iitb:000220
```

### 3️⃣ Test with Output
```bash
python test_iitb_corridor.py 000209 --output tracked_corridor.mp4
```

### 4️⃣ Test Random Sample
```bash
# Test 3 random videos
python test_iitb_corridor.py --random 3
```

## 📊 Available Commands

### Using `test_iitb_corridor.py`

```bash
# List all videos
python test_iitb_corridor.py --list

# Test specific video
python test_iitb_corridor.py 000209

# Save output
python test_iitb_corridor.py 000220 --output tracked.mp4

# Higher confidence
python test_iitb_corridor.py 000250 --conf 0.4

# No display (faster)
python test_iitb_corridor.py 000209 --no-display

# Test 5 random videos
python test_iitb_corridor.py --random 5
```

### Using `test_yolo_deepsort.py`

```bash
# Short ID format (6 digits)
python test_yolo_deepsort.py --source 000209

# With prefix
python test_yolo_deepsort.py --source iitb:000220

# Full path (still works)
python test_yolo_deepsort.py --source "D:\SPHAR-Dataset\videos\IITB-Corridor\000209\000209.avi"
```

### Using `test_sample_videos.py`

```bash
# Preset IITB videos
python test_sample_videos.py iitb_corridor_1  # 000209
python test_sample_videos.py iitb_corridor_2  # 000220
python test_sample_videos.py iitb_corridor_3  # 000250
```

## 📹 Video Format

| Property | Value |
|----------|-------|
| **Format** | AVI |
| **Type** | Surveillance footage |
| **Content** | Corridor monitoring |
| **Count** | 74 videos |
| **IDs** | 000209 - 000358 |

## 🎯 Recommended Settings

### For Surveillance Videos

```bash
# High confidence (reduce false positives)
python test_yolo_deepsort.py --source 000209 --conf 0.35

# Longer tracking persistence
python test_yolo_deepsort.py --source 000209 --max-age 50

# Best quality
python test_yolo_deepsort.py \
    --source 000209 \
    --conf 0.35 \
    --max-age 50 \
    --output tracked_corridor.mp4
```

## 💡 Tips

1. **Surveillance videos** benefit from higher confidence (0.35-0.4)
2. **Longer max-age** (40-60) for better tracking through occlusions
3. **No-display mode** for batch processing
4. **Random testing** good for quick quality checks

## 📈 Expected Results

Surveillance videos typically show:
- ✅ Multiple people walking in corridor
- ✅ Occlusions (people overlapping)
- ✅ Long-duration tracks
- ⚠️ Lower quality than action datasets
- ⚠️ More challenging lighting

## 🔧 Examples

### Example 1: Quick Test
```bash
python test_iitb_corridor.py 000209
```

### Example 2: Batch Processing
```bash
# Test first 10 videos without display
for i in {209..218}; do
    python test_yolo_deepsort.py --source 0002$i --no-display
done
```

### Example 3: Compare Settings
```bash
# Standard conf
python test_yolo_deepsort.py --source 000220 --conf 0.25 --output tracked_025.mp4

# High conf
python test_yolo_deepsort.py --source 000220 --conf 0.4 --output tracked_040.mp4
```

## 📊 Video List

**Available Videos:**
```
000209, 000210, 000211, 000212, 000213, 000214, 000215, 000216, 000217, 000218
000219, 000220, 000221, 000222, 000223, 000224, 000225, 000226, 000227, 000228
000229, 000230, 000231, 000232, 000233, 000234, 000235, 000236, 000237, 000238
000239, 000240, 000241, 000242, 000243, 000244, 000245, 000246, 000247, 000248
000249, 000250, 000251, 000252, 000253, 000254, 000255, 000256, 000257, 000258
000259, 000260, 000261, 000262, 000263, 000264, 000265, 000266, 000267, 000268
000269, 000270, 000271, 000272, 000273, 000274, 000275, 000276, 000277, 000278
000279, 000280, 000281, 000282
```

(74 videos total)

## 🎨 Tracking Features

All IITB-Corridor videos will show:
- ✅ Person ID tracking
- ✅ Confidence scores (> 0.5 filter applied)
- ✅ Trajectory lines
- ✅ Real-time FPS
- ✅ "Confidence Filter: > 0.5" indicator
- ✅ DeepSORT status

## 🆘 Troubleshooting

### Video not found?
```bash
# List available videos
python test_iitb_corridor.py --list

# Check if ID is correct (6 digits: 000209 not 209)
```

### No detections?
```bash
# Lower confidence
python test_yolo_deepsort.py --source 000209 --conf 0.2

# Note: Confidence > 0.5 filter still applies!
# Edit test_yolo_deepsort.py line ~116 to change
```

### Too slow?
```bash
# Disable display
python test_yolo_deepsort.py --source 000209 --no-display
```

## 📚 Related Files

- `test_yolo_deepsort.py` - Main tracking script
- `test_iitb_corridor.py` - Helper script for IITB videos
- `test_sample_videos.py` - Includes 3 IITB presets
- `CONFIDENCE_FILTER.md` - Info about > 0.5 filter

---

**Dataset**: IITB-Corridor Surveillance Dataset  
**Videos**: 74 corridor surveillance recordings  
**Format**: AVI files  
**Use Case**: Multi-person tracking in surveillance scenarios
