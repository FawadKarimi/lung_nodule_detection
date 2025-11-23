# 🚀 Quick Start Guide - University Lab Setup

## Step-by-Step Instructions

### 1. Upload Project to Lab PC
- Upload the entire `lung_nodule_detection` folder to the lab PC
- Keep the folder structure intact

### 2. Download LIDC-IDRI Dataset
1. Go to: https://www.cancerimagingarchive.net/collection/lidc-idri/
2. Download the dataset (requires registration)
3. Extract to: `./data/LIDC-IDRI/`

**Expected structure:**
```
lung_nodule_detection/
├── main.py
├── data/
│   └── LIDC-IDRI/
│       ├── LIDC-IDRI-0001/
│       ├── LIDC-IDRI-0002/
│       └── ...
```

### 3. Install Dependencies
```bash
# Navigate to project folder
cd lung_nodule_detection

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 4. Verify Setup (Optional)
```bash
python scripts/check_setup.py
```

This will check:
- ✅ Python version
- ✅ All packages installed
- ✅ CUDA available
- ✅ Project structure
- ✅ LIDC data found

### 5. Run Training
```bash
# Automatic mode - handles everything
python main.py --data_root ./data
```

**What happens automatically:**
1. 🔍 Finds LIDC-IDRI dataset
2. 🔄 Converts DICOM to NIfTI
3. 🎨 Preprocesses all scans (normalization, resampling, segmentation)
4. 📝 Parses XML annotations
5. ✂️ Splits dataset (70/15/15)
6. 🚂 Starts training

**Time estimates:**
- Preprocessing: 4-8 hours (one-time)
- Annotation parsing: 10-30 minutes (one-time)
- Training: Depends on epochs (60 epochs ≈ 1-2 days on H100)

### 6. Monitor Training
- Progress shown in terminal
- Checkpoints saved to `./checkpoints/`
- Best model: `checkpoints/best_model.pth`

### 7. Resume Training (if interrupted)
```bash
python main.py --data_root ./data --resume checkpoints/best_model.pth
```

## Common Commands

```bash
# Full pipeline (first time)
python main.py --data_root ./data

# Skip preprocessing (already done)
python main.py --data_root ./data --skip_preprocessing

# Skip annotations (already created)
python main.py --data_root ./data --skip_annotations

# Use specific GPU
python main.py --data_root ./data --gpu_id 0

# Custom config
python main.py --data_root ./data --config my_config.yaml
```

## Troubleshooting

### "Could not find LIDC-IDRI dataset"
- Check data is in `./data/LIDC-IDRI/`
- Or specify exact path: `--data_root /full/path/to/LIDC-IDRI`

### "CUDA out of memory"
- Reduce batch size in `config/config.yaml`:
  ```yaml
  data:
    batch_size: 8  # Reduce from 16
  ```

### "Module not found"
- Install dependencies: `pip install -r requirements.txt`

### Preprocessing too slow?
- It's normal! Full dataset takes 4-8 hours
- You can test with subset first (modify `scripts/preprocess_lidc.py`)

## Output Files

After running, you'll have:
- `processed_data/` - Preprocessed CT scans (.npy files)
- `annotations/` - JSON annotation files
- `checkpoints/` - Model checkpoints
- `logs/` - Training logs (if TensorBoard enabled)

## Next Steps

After training completes:
1. Evaluate: `python scripts/evaluate.py --checkpoint checkpoints/best_model.pth`
2. Inference: `python scripts/inference.py --checkpoint checkpoints/best_model.pth`

## Need Help?

Check `README.md` for detailed documentation.

