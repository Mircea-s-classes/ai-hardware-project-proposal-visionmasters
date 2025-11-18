# Getting Started Guide

Welcome to the Real-Time Facial Expression Recognition project! This guide will help you get started with the project without needing the physical hardware.

## 📋 Prerequisites

- **Python 3.9+**
- **Git**
- **Kaggle Account** (for dataset download)
- **8GB RAM minimum** (16GB recommended for training)
- **GPU optional** (significantly speeds up training)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd ai-hardware-project-proposal-visionmasters
```

### 2. Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download FER2013 Dataset

#### Option A: Using Kaggle API (Recommended)

```bash
# Install Kaggle
pip install kaggle

# Setup Kaggle credentials
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Download kaggle.json
# 4. Move to ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<username>\.kaggle\kaggle.json (Windows)
# 5. Set permissions (Linux/Mac only)
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
cd data/fer2013
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip
rm fer2013.zip
cd ../..
```

#### Option B: Manual Download

1. Visit https://www.kaggle.com/datasets/msambare/fer2013
2. Click "Download"
3. Extract to `data/fer2013/`

### 4. Verify Dataset

```bash
python src/model/prepare_data.py
```

This will:
- ✅ Verify dataset structure
- 📊 Show dataset statistics
- 🎨 Create sample visualizations

Expected output:
```
FER2013 Dataset Preparation
============================================================
📊 Analyzing FER2013 Dataset...
============================================================

Emotion      Train      Test     Total
------------------------------------------------------------
Angry        3,995      958      4,953
...
============================================================
✅ Dataset preparation complete!
```

### 5. Train Baseline Model

```bash
python src/model/train_baseline.py
```

This will:
- Build MobileNetV2-based model
- Train for up to 50 epochs (with early stopping)
- Save best model to `models/baseline_fp32_best.h5`
- Generate training curves

**Expected time**: 2-3 hours with GPU, 8-12 hours with CPU

### 6. Evaluate Model

```bash
python src/model/evaluate.py --model models/baseline_fp32_best.h5
```

This will:
- Test model on test set
- Generate confusion matrix
- Create per-class accuracy charts
- Show classification report

**Target accuracy**: >85%

### 7. Quantize Model

```bash
python src/model/quantize_model.py
```

This will:
- Convert FP32 model to INT8
- Compare model sizes and accuracy
- Save quantized model to `models/model_int8.tflite`

**Expected**: ~4x size reduction, <5% accuracy drop

### 8. Benchmark Performance

```bash
python benchmarks/benchmark_model.py
```

This will:
- Measure inference latency for all models
- Test face detection speed
- Generate comparison charts

## 📊 Current Workflow

```
┌─────────────────────────────────────────────────────────┐
│                    PHASE 1: Model Development           │
│                 (No Hardware Required)                  │
└─────────────────────────────────────────────────────────┘
           ↓
    prepare_data.py
           ↓
    train_baseline.py  →  models/baseline_fp32_best.h5
           ↓
    evaluate.py  →  Accuracy: 85%+
           ↓
    quantize_model.py  →  models/model_int8.tflite
           ↓
    benchmark_model.py  →  Latency analysis

┌─────────────────────────────────────────────────────────┐
│              PHASE 2: Hardware Integration              │
│         (Requires Raspberry Pi + Coral)                 │
└─────────────────────────────────────────────────────────┘
           ↓
    Edge TPU Compiler  →  model_int8_edgetpu.tflite
           ↓
    Deploy to Raspberry Pi
           ↓
    inference_demo.py  →  Real-time demo
           ↓
    Performance testing & optimization
```

## 📁 Project Structure

```
.
├── src/
│   ├── model/
│   │   ├── prepare_data.py      # Dataset verification
│   │   ├── train_baseline.py    # Model training
│   │   ├── evaluate.py          # Model evaluation
│   │   └── quantize_model.py    # Model quantization
│   ├── hardware/
│   │   └── inference_demo.py    # Real-time demo (Pi + Coral)
│   └── utils/
│       └── face_detection.py    # Face detection utilities
├── benchmarks/
│   └── benchmark_model.py       # Performance benchmarking
├── data/
│   ├── fer2013/                 # Dataset (download separately)
│   └── emotes/                  # Clash Royale emotes
├── models/                      # Trained models (generated)
├── results/                     # Evaluation results (generated)
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```

## 🎯 Current Objectives (Week 2-3)

Based on your timeline, you're in **Week 2** (Nov 12-19). Here's what you should focus on:

### Week 2: Hardware Setup & Initial Model (Current)

- [x] ✅ Project structure setup
- [ ] 📥 Download FER2013 dataset
- [ ] 🏋️ Train baseline FP32 model
- [ ] 📊 Evaluate model accuracy
- [ ] 🎮 Prepare emote assets

**Deliverable**: Basic FP32 model with >85% accuracy

### Week 3: Model Optimization & TPU Preparation

- [ ] 🔢 Quantize model to INT8
- [ ] 📏 Benchmark inference latency
- [ ] 📦 Prepare for Edge TPU compilation
- [ ] 📑 Prepare midterm presentation slides

**Deliverable**: Quantized INT8 model, initial performance metrics

### Week 4: Midterm Presentation

- [ ] 🎤 Present setup, model, and early results
- [ ] 📊 Show training curves and accuracy metrics
- [ ] 🚧 Discuss any challenges faced

## 🎮 Preparing Clash Royale Emotes

You can copy emotes from the reference repository:

```bash
# Copy emote images
mkdir -p data/emotes/images
cp clash-royale-emote-detector/images/* data/emotes/images/

# Copy emote sounds
mkdir -p data/emotes/sounds
cp clash-royale-emote-detector/sounds/* data/emotes/sounds/

# Rename to match your emotion labels
cd data/emotes/images
mv laughing.png happy.png
mv crying.png sad.png
# ... (add more mappings as needed)
```

Or find Clash Royale emote packs online.

## 🧪 Testing Without Hardware

You can test most components without the Raspberry Pi:

### Test Face Detection

```bash
python src/utils/face_detection.py
```

This will open your webcam and show face detection in real-time.

### Test Model on Webcam (CPU/GPU)

Create a simple test script to use your trained model with webcam on your development machine (without Edge TPU).

## 📝 Documentation Tasks

While training models, you can work on:

1. **Midterm Presentation Slides**
   - Problem statement
   - Approach and methodology
   - Model architecture
   - Initial results

2. **Update README**
   - Add results as you get them
   - Update performance table

3. **Create System Diagram**
   - Show data flow
   - Component interactions

## 🐛 Troubleshooting

### Out of Memory During Training

```bash
# Reduce batch size in train_baseline.py
# Change from 32 to 16 or 8
batch_size = 16
```

### GPU Not Detected

```bash
# Check TensorFlow GPU support
python -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"

# If not available, training will use CPU (slower but works)
```

### Dataset Not Found

Make sure you've extracted the dataset to the correct location:
```
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    └── (same structure)
```

## 📚 Useful Resources

### Documentation
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [Coral Edge TPU](https://coral.ai/docs/)

### Tutorials
- [Transfer Learning with TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Post-Training Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [Face Detection with MediaPipe](https://google.github.io/mediapipe/solutions/face_detection.html)

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**: Most scripts provide detailed error messages
2. **Review documentation**: Check relevant README files
3. **Search the error**: Google the error message
4. **Ask team members**: Collaborate with your team
5. **Office hours**: Ask your professor or TA

## ✅ Next Steps

After completing the baseline model:

1. ✅ Review results and ensure >85% accuracy
2. ✅ Document findings for midterm presentation
3. ✅ Begin quantization experiments
4. ✅ Prepare slides and demo for midterm
5. ⏳ Wait for hardware to arrive for integration phase

## 🎯 Success Criteria

By the end of Week 2-3, you should have:

- ✅ FP32 model trained with >85% accuracy
- ✅ Confusion matrix and evaluation metrics
- ✅ INT8 quantized model with <5% accuracy drop
- ✅ Benchmark results showing inference latency
- ✅ Midterm presentation ready
- ✅ Clear plan for hardware integration

---

**Good luck with your project! 🚀**

For questions or issues, check the documentation or reach out to your team.

