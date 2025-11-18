# Results Directory

This directory contains evaluation results, benchmarks, and performance metrics.

## Generated Files

### Training Results

- `training_results.txt` - Summary of training metrics
- `charts/training_history.png` - Training and validation curves

### Evaluation Results

- `classification_report.txt` - Precision, recall, F1-score per class
- `error_analysis.txt` - Analysis of misclassifications
- `charts/confusion_matrix.png` - Confusion matrix visualization
- `charts/per_class_accuracy.png` - Per-emotion accuracy chart

### Quantization Results

- `quantization_results.txt` - FP32 vs INT8 comparison
- Model size reduction metrics
- Accuracy drop analysis

### Benchmark Results

- `benchmark_results.csv` - Latency measurements
- `charts/latency_comparison.png` - Latency across models
- `charts/latency_distribution.png` - Latency distributions

### Dataset Analysis

- `dataset_samples.png` - Sample images from each emotion class
- `charts/class_distribution.png` - Class imbalance visualization

## Note

⚠️ **Most result files are not tracked in Git** as they can be regenerated.

To generate results:
1. Train model: `python src/model/train_baseline.py`
2. Evaluate: `python src/model/evaluate.py`
3. Quantize: `python src/model/quantize_model.py`
4. Benchmark: `python benchmarks/benchmark_model.py`

## Directory Structure

```
results/
├── charts/                    # Visualizations
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   ├── class_distribution.png
│   ├── latency_comparison.png
│   └── latency_distribution.png
├── logs/                      # TensorBoard logs
│   └── [timestamp]/
├── training_results.txt
├── classification_report.txt
├── error_analysis.txt
├── quantization_results.txt
└── benchmark_results.csv
```

## Expected Results

### Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Accuracy** | >85% | On FER2013 test set |
| **Accuracy Drop (INT8)** | <5% | After quantization |
| **Model Size (INT8)** | <5MB | ~3.5MB typical |
| **Inference Latency (TPU)** | <20ms | Model inference only |
| **Total Latency** | <50ms | Detection + inference + post-processing |
| **FPS** | >30 | Complete pipeline |
| **Power** | <5W | Pi + Coral + Camera |

### Typical Results

**Training** (after ~30-40 epochs):
- Training accuracy: 85-90%
- Validation accuracy: 80-85%
- Test accuracy: 82-87%

**Per-Class Performance**:
- Happy: 90-95% (most samples, easiest to detect)
- Sad: 80-85%
- Angry: 75-85%
- Surprise: 80-90%
- Neutral: 75-85%
- Fear: 70-80% (harder to distinguish)
- Disgust: 60-75% (fewest samples, hardest to detect)

**Quantization**:
- Size reduction: ~4x (14MB → 3.5MB)
- Accuracy drop: 2-4%
- Speed improvement: 2-3x on Edge TPU

**Latency Breakdown** (typical):
- Face detection: 10-20ms
- Preprocessing: 2-5ms
- Model inference (Edge TPU): 5-15ms
- Post-processing: 1-2ms
- **Total**: 18-42ms (24-60 FPS possible)

## Using Results for Presentation

### For Midterm Presentation

Include:
1. Training curves showing convergence
2. Confusion matrix showing per-class performance
3. Class distribution (show dataset imbalance)
4. Initial accuracy metrics

### For Final Presentation

Include:
1. All of the above
2. Quantization comparison (size, accuracy, speed)
3. Latency benchmarks (breakdown by component)
4. FPS achieved on Raspberry Pi
5. Power consumption measurements
6. Real-world demo video

## Visualization Tips

All generated charts are publication-ready:
- High resolution (150 DPI)
- Clear labels and titles
- Color-coded for easy interpretation
- Suitable for presentations and reports

## Troubleshooting

**No charts generated?**
- Ensure matplotlib is installed: `pip install matplotlib seaborn`
- Check that scripts completed successfully

**TensorBoard logs empty?**
- Training may have stopped early
- Check for errors in training logs

**Poor results?**
- Check dataset is downloaded correctly
- Verify training completed (50 epochs or early stopping)
- Check GPU was used (training much slower on CPU)
- Try longer training or adjust hyperparameters

