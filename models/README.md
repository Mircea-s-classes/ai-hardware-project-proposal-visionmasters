# Models Directory

This directory contains trained models at various stages of optimization.

## Model Files (Generated during training)

### Training Outputs

- `baseline_fp32_best.h5` - Best FP32 model (highest validation accuracy)
- `baseline_fp32_final.h5` - Final FP32 model after all training epochs

### TFLite Models

- `model_fp32.tflite` - FP32 TensorFlow Lite model (baseline)
- `model_int8.tflite` - INT8 quantized TFLite model (for Edge TPU)
- `model_int8_edgetpu.tflite` - Edge TPU compiled model (final deployment)

## Model Sizes

| Model | Size | Accuracy | Latency (Edge TPU) |
|-------|------|----------|-------------------|
| FP32 (.h5) | ~14 MB | Baseline | N/A (not TPU compatible) |
| FP32 TFLite | ~14 MB | Baseline | ~40-60ms |
| INT8 TFLite | ~3.5 MB | -3% to -5% | ~15-25ms |
| INT8 EdgeTPU | ~3.5 MB | -3% to -5% | ~5-15ms |

## Note

⚠️ **Model files are not tracked in Git** due to their large size.

To get the models:
1. Train them yourself: `python src/model/train_baseline.py`
2. Or download from team's shared storage (if available)

## Model Format Details

### .h5 (Keras/HDF5)
- Native Keras format
- Full precision (FP32)
- Best for training and fine-tuning
- Not suitable for Edge TPU

### .tflite (TensorFlow Lite)
- Optimized for mobile/edge deployment
- Supports quantization (FP32, FP16, INT8)
- Smaller size, faster inference
- Can run on CPU, GPU, or TPU

### _edgetpu.tflite (Edge TPU)
- Compiled specifically for Google Coral Edge TPU
- INT8 quantized
- Hardware-accelerated operations
- Requires Coral USB Accelerator or Dev Board

## Compilation Pipeline

```
Training (GPU/CPU)
      ↓
baseline_fp32_best.h5 (Keras model, ~14MB)
      ↓
[Conversion]
      ↓
model_fp32.tflite (TFLite, ~14MB)
      ↓
[Quantization]
      ↓
model_int8.tflite (Quantized TFLite, ~3.5MB)
      ↓
[Edge TPU Compiler]
      ↓
model_int8_edgetpu.tflite (Edge TPU optimized, ~3.5MB)
```

## Usage

### Keras Model
```python
from tensorflow import keras
model = keras.models.load_model('models/baseline_fp32_best.h5')
predictions = model.predict(image)
```

### TFLite Model (CPU)
```python
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='models/model_int8.tflite')
interpreter.allocate_tensors()
# ... run inference
```

### Edge TPU Model (Coral)
```python
from pycoral.utils import edgetpu
interpreter = edgetpu.make_interpreter('models/model_int8_edgetpu.tflite')
interpreter.allocate_tensors()
# ... run inference
```

## Generating Models

Follow these steps in order:

1. **Train baseline model**:
   ```bash
   python src/model/train_baseline.py
   ```
   Generates: `baseline_fp32_best.h5`

2. **Quantize for Edge TPU**:
   ```bash
   python src/model/quantize_model.py
   ```
   Generates: `model_fp32.tflite`, `model_int8.tflite`

3. **Compile for Edge TPU** (on Raspberry Pi or Linux):
   ```bash
   edgetpu_compiler model_int8.tflite
   ```
   Generates: `model_int8_edgetpu.tflite`

## Model Versioning

When experimenting, consider naming models with versions:
- `baseline_v1_fp32.h5`
- `mobilenet_v2_int8.tflite`
- `fer2013_final_edgetpu.tflite`

This helps track different experiments and configurations.

