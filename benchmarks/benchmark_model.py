"""
Model Benchmarking Script
Measures inference latency, throughput, and resource usage
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.face_detection import FaceDetector, FacePreprocessor


def benchmark_keras_model(model_path, img_size=(224, 224), num_iterations=100):
    """Benchmark Keras model inference time"""
    print(f"\n⏱️  Benchmarking Keras model...")
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Create dummy input
    dummy_input = np.random.rand(1, *img_size, 3).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=0)
    
    # Benchmark
    latencies = []
    for i in range(num_iterations):
        start = time.perf_counter()
        _ = model.predict(dummy_input, verbose=0)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        'model_type': 'Keras (FP32)',
        'mean_latency': np.mean(latencies),
        'std_latency': np.std(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
        'median_latency': np.median(latencies),
        'fps': 1000.0 / np.mean(latencies),
        'latencies': latencies
    }


def benchmark_tflite_model(model_path, img_size=(224, 224), num_iterations=100):
    """Benchmark TFLite model inference time"""
    print(f"\n⏱️  Benchmarking TFLite model...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create dummy input
    if input_details[0]['dtype'] == np.uint8:
        dummy_input = np.random.randint(0, 256, (1, *img_size, 3), dtype=np.uint8)
    else:
        dummy_input = np.random.rand(1, *img_size, 3).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    # Benchmark
    latencies = []
    for i in range(num_iterations):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    model_type = 'TFLite (INT8)' if input_details[0]['dtype'] == np.uint8 else 'TFLite (FP32)'
    
    return {
        'model_type': model_type,
        'mean_latency': np.mean(latencies),
        'std_latency': np.std(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
        'median_latency': np.median(latencies),
        'fps': 1000.0 / np.mean(latencies),
        'latencies': latencies
    }


def benchmark_face_detection(num_iterations=100):
    """Benchmark face detection latency"""
    print(f"\n⏱️  Benchmarking face detection...")
    
    detector = FaceDetector()
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        _ = detector.detect_faces(dummy_frame)
    
    # Benchmark
    latencies = []
    for i in range(num_iterations):
        start = time.perf_counter()
        _ = detector.detect_faces(dummy_frame)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    detector.close()
    
    return {
        'component': 'Face Detection',
        'mean_latency': np.mean(latencies),
        'std_latency': np.std(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
        'median_latency': np.median(latencies),
        'fps': 1000.0 / np.mean(latencies),
        'latencies': latencies
    }


def benchmark_preprocessing(img_size=(224, 224), num_iterations=100):
    """Benchmark image preprocessing latency"""
    print(f"\n⏱️  Benchmarking preprocessing...")
    
    preprocessor = FacePreprocessor(target_size=img_size)
    
    # Create dummy face image
    dummy_face = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    
    # Benchmark
    latencies = []
    for i in range(num_iterations):
        start = time.perf_counter()
        _ = preprocessor.preprocess(dummy_face)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        'component': 'Preprocessing',
        'mean_latency': np.mean(latencies),
        'std_latency': np.std(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
        'median_latency': np.median(latencies),
        'fps': 1000.0 / np.mean(latencies),
        'latencies': latencies
    }


def plot_latency_comparison(results, output_path):
    """Plot latency comparison across different components"""
    print("\n📊 Creating latency comparison chart...")
    
    # Prepare data
    labels = [r['model_type'] if 'model_type' in r else r['component'] for r in results]
    means = [r['mean_latency'] for r in results]
    stds = [r['std_latency'] for r in results]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(labels, means, yerr=stds, capsize=10, 
                  color='#3498db', edgecolor='black', alpha=0.7)
    
    # Color bars based on latency
    for bar, mean in zip(bars, means):
        if mean < 20:
            bar.set_color('#27ae60')  # Green (target met)
        elif mean < 50:
            bar.set_color('#f39c12')  # Orange (acceptable)
        else:
            bar.set_color('#e74c3c')  # Red (needs optimization)
    
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
               f'{mean:.2f}ms\n({1000/mean:.1f} FPS)',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add target line
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Target (<20ms)')
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Latency comparison chart saved to: {output_path}")
    plt.close()


def plot_latency_distribution(results, output_path):
    """Plot latency distribution for each component"""
    print("\n📊 Creating latency distribution plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(results[:4]):  # Plot first 4 results
        if idx >= len(axes):
            break
            
        label = result.get('model_type') or result.get('component')
        latencies = result['latencies']
        
        axes[idx].hist(latencies, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        axes[idx].axvline(result['mean_latency'], color='red', linestyle='--', 
                         linewidth=2, label=f"Mean: {result['mean_latency']:.2f}ms")
        axes[idx].axvline(result['median_latency'], color='green', linestyle='--',
                         linewidth=2, label=f"Median: {result['median_latency']:.2f}ms")
        
        axes[idx].set_xlabel('Latency (ms)', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    # Remove unused subplots
    for idx in range(len(results), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Latency distribution plot saved to: {output_path}")
    plt.close()


def save_benchmark_results(results, output_path):
    """Save benchmark results to CSV"""
    print("\n💾 Saving benchmark results...")
    
    # Convert to DataFrame
    data = []
    for result in results:
        label = result.get('model_type') or result.get('component')
        data.append({
            'Component': label,
            'Mean Latency (ms)': f"{result['mean_latency']:.4f}",
            'Std Latency (ms)': f"{result['std_latency']:.4f}",
            'Min Latency (ms)': f"{result['min_latency']:.4f}",
            'Max Latency (ms)': f"{result['max_latency']:.4f}",
            'Median Latency (ms)': f"{result['median_latency']:.4f}",
            'FPS': f"{result['fps']:.2f}"
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Results saved to: {output_path}")
    
    # Print table
    print("\n" + "="*80)
    print("Benchmark Results:")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


def main():
    """Main benchmarking function"""
    parser = argparse.ArgumentParser(description='Benchmark model performance')
    parser.add_argument('--keras-model', type=str, default='models/baseline_fp32_best.h5',
                       help='Path to Keras model')
    parser.add_argument('--fp32-tflite', type=str, default='models/model_fp32.tflite',
                       help='Path to FP32 TFLite model')
    parser.add_argument('--int8-tflite', type=str, default='models/model_int8.tflite',
                       help='Path to INT8 TFLite model')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Model Performance Benchmarking")
    print("="*80)
    
    results = []
    
    # Benchmark face detection
    face_result = benchmark_face_detection(args.iterations)
    results.append(face_result)
    
    # Benchmark preprocessing
    preprocess_result = benchmark_preprocessing(num_iterations=args.iterations)
    results.append(preprocess_result)
    
    # Benchmark Keras model
    keras_path = project_root / args.keras_model
    if keras_path.exists():
        keras_result = benchmark_keras_model(keras_path, num_iterations=args.iterations)
        results.append(keras_result)
    else:
        print(f"⚠️  Keras model not found: {keras_path}")
    
    # Benchmark FP32 TFLite
    fp32_path = project_root / args.fp32_tflite
    if fp32_path.exists():
        fp32_result = benchmark_tflite_model(fp32_path, num_iterations=args.iterations)
        results.append(fp32_result)
    else:
        print(f"⚠️  FP32 TFLite model not found: {fp32_path}")
    
    # Benchmark INT8 TFLite
    int8_path = project_root / args.int8_tflite
    if int8_path.exists():
        int8_result = benchmark_tflite_model(int8_path, num_iterations=args.iterations)
        results.append(int8_result)
    else:
        print(f"⚠️  INT8 TFLite model not found: {int8_path}")
    
    # Save results
    results_dir = project_root / 'results'
    charts_dir = results_dir / 'charts'
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    save_benchmark_results(results, results_dir / 'benchmark_results.csv')
    plot_latency_comparison(results, charts_dir / 'latency_comparison.png')
    plot_latency_distribution(results, charts_dir / 'latency_distribution.png')
    
    print("\n" + "="*80)
    print("✅ Benchmarking complete!")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print(f"Charts saved to: {charts_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

