# Quick Start Guide - Week 2

This is a condensed guide to get you started immediately.

## 🎯 Your Current Phase: Week 2 (Nov 12-19)

**Goal**: Setup & Initial Model Training (No Hardware Needed Yet)

## ✅ What's Already Done

All the code infrastructure is ready! You have:

- ✅ Project structure created
- ✅ Training scripts prepared
- ✅ Evaluation tools ready
- ✅ Benchmarking scripts ready
- ✅ Documentation complete

## 📝 Your Next 3 Steps

### Step 1: Setup Environment (15 minutes)

```bash
# Navigate to project
cd ai-hardware-project-proposal-visionmasters

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Dataset (30 minutes)

**Option A - Kaggle API (Recommended)**:
```bash
# Setup Kaggle (one-time)
pip install kaggle
# Get API key from: https://www.kaggle.com/account
# Download kaggle.json and place in ~/.kaggle/

# Download dataset
cd data/fer2013
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip
cd ../..

# Verify
python src/model/prepare_data.py
```

**Option B - Manual**: Download from https://www.kaggle.com/datasets/msambare/fer2013

### Step 3: Train Model (2-3 hours with GPU)

```bash
# Start training
python src/model/train_baseline.py

# This will:
# - Train MobileNetV2 model
# - Use data augmentation
# - Save best model to models/baseline_fp32_best.h5
# - Generate training curves in results/
```

## 📊 Check Results

After training completes:

```bash
# Evaluate accuracy
python src/model/evaluate.py

# Quantize for Edge TPU
python src/model/quantize_model.py

# Benchmark performance
python benchmarks/benchmark_model.py
```

## 🎮 Prepare Emotes (Optional)

```bash
# Copy emotes from reference repo
bash scripts/prepare_emotes.sh

# Or manually organize emotes in data/emotes/
```

## 📈 Expected Results

After completing these steps, you should have:

| Metric | Target | Status |
|--------|--------|--------|
| Test Accuracy | >85% | Check results/ |
| Model Size (FP32) | ~14MB | Check models/ |
| Model Size (INT8) | ~3.5MB | After quantization |
| Training Time | 2-3 hrs | With GPU |

## 🎤 Prepare for Midterm (Week 4)

While models train, work on:

1. **Slides**: Problem, approach, architecture, initial results
2. **Diagrams**: System architecture, data flow
3. **Results**: Training curves, confusion matrix, accuracy metrics

## 🐛 Common Issues

**Out of Memory?**
- Reduce batch_size in `train_baseline.py` (line 29): `'batch_size': 16`

**No GPU?**
- Training works on CPU (just slower, 8-12 hours)

**Dataset not found?**
- Check structure: `data/fer2013/train/angry/`, etc.

## 📚 Full Documentation

- **Complete Guide**: `GETTING_STARTED.md`
- **Dataset Info**: `data/README.md`
- **Model Training**: `src/model/README.md`
- **Hardware Setup** (later): `src/hardware/README.md`

## 💡 Tips

1. **Start training ASAP** - it takes several hours
2. **Monitor training** - check TensorBoard logs
3. **Document results** - save all metrics for your presentation
4. **Team collaboration** - split tasks (Allen: hardware research, Marvin: docs, Sami: model training)

## ⏭️ After Week 2

Once you have a trained model:

1. **Week 3**: Quantization, benchmarking, midterm prep
2. **Week 4**: Midterm presentation
3. **Week 5+**: Hardware integration (when Pi + Coral arrive)

---

**Need Help?** Check `GETTING_STARTED.md` for detailed troubleshooting.

**Questions?** Review the documentation or ask your team/professor.

🚀 **You're all set to start! Good luck!**

