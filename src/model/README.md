# Model Training and Optimization

This directory contains scripts for training, evaluating, and optimizing the facial expression recognition model.

## Workflow

```
1. prepare_data.py      → Verify and analyze dataset
2. train_baseline.py    → Train FP32 baseline model
3. evaluate.py          → Evaluate model performance
4. quantize_model.py    → Convert to INT8 for Edge TPU
5. convert_to_edgetpu.py → Compile for Edge TPU (requires Edge TPU Compiler)
```

## Scripts

### 1. Data Preparation

**File**: `prepare_data.py`

Verifies dataset and creates visualizations.

```bash
python src/model/prepare_data.py
```

**Outputs**:
- Dataset statistics
- Sample images from each class
- Class distribution charts

### 2. Baseline Model Training

**File**: `train_baseline.py`

Trains a MobileNetV2-based model on FER2013 dataset.

```bash
python src/model/train_baseline.py
```

**Features**:
- Transfer learning from ImageNet
- Data augmentation
- Early stopping and learning rate scheduling
- TensorBoard logging
- Saves best model based on validation accuracy

**Outputs**:
- `models/baseline_fp32_best.h5` - Best model (highest val accuracy)
- `models/baseline_fp32_final.h5` - Final model after all epochs
- `results/charts/training_history.png` - Training curves
- `results/training_results.txt` - Summary

**Configuration**:
```python
img_size = (224, 224)
batch_size = 32
epochs = 50
learning_rate = 0.001
```

**Expected Results**:
- Training Time: ~2-3 hours (on GPU)
- Target Accuracy: >85% on test set
- Model Size: ~10-15 MB

### 3. Model Evaluation

**File**: `evaluate.py`

Comprehensive evaluation with metrics and visualizations.

```bash
python src/model/evaluate.py --model models/baseline_fp32_best.h5
```

**Outputs**:
- Confusion matrix
- Per-class accuracy chart
- Classification report (precision, recall, F1)
- Error analysis

**Options**:
```bash
--model        Path to model file
--data-dir     Path to dataset directory
--img-size     Input image size (default: 224)
--batch-size   Batch size (default: 32)
```

### 4. Model Quantization

**File**: `quantize_model.py`

Converts FP32 model to INT8 quantized TFLite model.

```bash
python src/model/quantize_model.py
```

**Features**:
- Post-training quantization
- Uses representative dataset for calibration
- Compares FP32 vs INT8 accuracy
- Checks if accuracy drop < 5%

**Outputs**:
- `models/model_fp32.tflite` - FP32 TFLite model
- `models/model_int8.tflite` - INT8 quantized model
- `results/quantization_results.txt` - Comparison

**Expected Results**:
- Size Reduction: ~4x (from ~14MB to ~3.5MB)
- Accuracy Drop: <5%
- Speed Improvement: 2-3x on Edge TPU

**Options**:
```bash
--model              Path to input Keras model
--data-dir           Path to dataset
--num-calib-samples  Number of calibration samples (default: 100)
--num-eval-samples   Number of evaluation samples (default: 500)
```

### 5. Edge TPU Compilation

**File**: `convert_to_edgetpu.py`

Compiles INT8 TFLite model for Edge TPU.

```bash
python src/model/convert_to_edgetpu.py
```

**Requirements**:
- Edge TPU Compiler (installed on Raspberry Pi or Linux machine)
- INT8 quantized TFLite model

**Install Edge TPU Compiler**:
```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler
```

**Compile Model**:
```bash
edgetpu_compiler models/model_int8.tflite
```

**Output**:
- `models/model_int8_edgetpu.tflite` - Edge TPU optimized model
- Compilation log showing which ops are mapped to TPU

## Model Architecture

```
Input (224x224x3)
     ↓
MobileNetV2 Base (ImageNet pretrained)
     ↓
GlobalAveragePooling
     ↓
Dropout (0.5)
     ↓
Dense (256, ReLU, L2 regularization)
     ↓
Dropout (0.3)
     ↓
Dense (7, Softmax)
     ↓
Output (7 classes)
```

**Why MobileNetV2?**
- Designed for mobile/edge devices
- Efficient inverted residual blocks
- Small size (~14MB for FP32)
- Fast inference (~10-20ms on Edge TPU)
- Good accuracy vs speed tradeoff

## Training Tips

### Improving Accuracy

1. **More Training Data**:
   - Use data augmentation (already implemented)
   - Consider additional datasets (FER+, RAF-DB)

2. **Class Imbalance**:
   - Use class weights in training
   - Oversample minority classes
   - Focus on per-class metrics

3. **Fine-tuning**:
   - Unfreeze more layers of MobileNetV2
   - Train for more epochs with lower learning rate

4. **Quantization-Aware Training**:
   - If INT8 accuracy drop >5%, use QAT
   - Simulates quantization during training

### Reducing Latency

1. **Smaller Input Size**:
   - Try 160x160 or 192x192 (faster but may reduce accuracy)

2. **Model Pruning**:
   - Remove less important weights
   - Can reduce size by 30-50%

3. **Knowledge Distillation**:
   - Train smaller model to mimic larger one

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Test Accuracy | >85% | TBD |
| Model Size (INT8) | <5MB | ~3.5MB |
| Inference Latency | <20ms | TBD |
| FPS (with detection) | >30 | TBD |

## Next Steps

After training and quantization:

1. **Benchmark Performance**:
```bash
python benchmarks/benchmark_model.py
```

2. **Deploy to Raspberry Pi**:
   - Copy models to Pi
   - Install PyCoral and TFLite Runtime
   - Run inference demo

3. **Optimize Pipeline**:
   - Profile bottlenecks
   - Optimize preprocessing
   - Parallelize face detection and inference

## References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/)
- [Post-Training Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
