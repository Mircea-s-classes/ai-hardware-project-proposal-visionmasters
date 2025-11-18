"""
Model Quantization Script
Converts FP32 model to INT8 quantized TFLite model for Edge TPU deployment
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_model(model_path):
    """Load trained Keras model"""
    print(f"\n📦 Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    return model


def create_representative_dataset(data_dir, num_samples=100, img_size=(224, 224)):
    """Create representative dataset for quantization calibration"""
    print(f"\n📊 Creating representative dataset ({num_samples} samples)...")
    
    datagen = ImageDataGenerator(rescale=1./255)
    
    generator = datagen.flow_from_directory(
        data_dir / 'train',
        target_size=img_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=True,
        color_mode='rgb'
    )
    
    def representative_dataset_gen():
        for i in range(num_samples):
            image, _ = next(generator)
            yield [image.astype(np.float32)]
    
    print("✅ Representative dataset created")
    return representative_dataset_gen


def convert_to_tflite(model, output_path, quantize=False, representative_dataset=None):
    """Convert Keras model to TFLite format"""
    print(f"\n🔄 Converting to TFLite (quantized={quantize})...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # INT8 quantization for Edge TPU
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if representative_dataset:
            converter.representative_dataset = representative_dataset
            # Full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            print("   Using full INT8 quantization with representative dataset")
        else:
            print("   Using dynamic range quantization")
    else:
        print("   No quantization (FP32)")
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ TFLite model saved to: {output_path}")
    print(f"   Size: {size_mb:.2f} MB")
    
    return tflite_model


def evaluate_tflite_model(tflite_path, test_data_dir, img_size=(224, 224), num_samples=500):
    """Evaluate TFLite model accuracy"""
    print(f"\n🧪 Evaluating TFLite model (first {num_samples} samples)...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input dtype: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output dtype: {output_details[0]['dtype']}")
    
    # Prepare test data
    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory(
        test_data_dir / 'test',
        target_size=img_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )
    
    # Evaluate
    correct = 0
    total = 0
    
    for i in range(min(num_samples, test_generator.samples)):
        image, label = next(test_generator)
        
        # Preprocess for INT8 input if needed
        if input_details[0]['dtype'] == np.uint8:
            input_data = (image * 255).astype(np.uint8)
        else:
            input_data = image.astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output if needed
        if output_details[0]['dtype'] == np.uint8:
            scale, zero_point = output_details[0]['quantization']
            output_data = scale * (output_data.astype(np.float32) - zero_point)
        
        # Check prediction
        pred_class = np.argmax(output_data)
        true_class = np.argmax(label)
        
        if pred_class == true_class:
            correct += 1
        total += 1
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{num_samples} samples...")
    
    accuracy = correct / total
    print(f"\n✅ TFLite Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Correct: {correct}/{total}")
    
    return accuracy


def compare_models(fp32_path, int8_path):
    """Compare FP32 and INT8 model sizes"""
    print("\n📊 Model Comparison:")
    print("="*60)
    
    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    int8_size = int8_path.stat().st_size / (1024 * 1024)
    compression_ratio = fp32_size / int8_size
    
    print(f"FP32 Model: {fp32_size:.2f} MB")
    print(f"INT8 Model: {int8_size:.2f} MB")
    print(f"Compression Ratio: {compression_ratio:.2f}x")
    print(f"Size Reduction: {(1 - int8_size/fp32_size)*100:.1f}%")
    print("="*60)


def main():
    """Main quantization function"""
    parser = argparse.ArgumentParser(description='Quantize model for Edge TPU deployment')
    parser.add_argument('--model', type=str, default='models/baseline_fp32_best.h5',
                       help='Path to input Keras model')
    parser.add_argument('--data-dir', type=str, default='data/fer2013',
                       help='Path to dataset directory')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num-calib-samples', type=int, default=100,
                       help='Number of samples for calibration')
    parser.add_argument('--num-eval-samples', type=int, default=500,
                       help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Model Quantization for Edge TPU")
    print("="*60)
    
    # Setup paths
    model_path = project_root / args.model
    data_dir = project_root / args.data_dir
    models_dir = project_root / 'models'
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    
    # Create representative dataset
    representative_dataset = create_representative_dataset(
        data_dir,
        num_samples=args.num_calib_samples,
        img_size=(args.img_size, args.img_size)
    )
    
    # Convert to FP32 TFLite (baseline)
    fp32_path = models_dir / 'model_fp32.tflite'
    convert_to_tflite(model, fp32_path, quantize=False)
    
    # Convert to INT8 TFLite (quantized)
    int8_path = models_dir / 'model_int8.tflite'
    convert_to_tflite(model, int8_path, quantize=True, 
                     representative_dataset=representative_dataset)
    
    # Compare model sizes
    compare_models(fp32_path, int8_path)
    
    # Evaluate both models
    print("\n" + "="*60)
    print("Evaluating Models:")
    print("="*60)
    
    fp32_acc = evaluate_tflite_model(fp32_path, data_dir, 
                                     img_size=(args.img_size, args.img_size),
                                     num_samples=args.num_eval_samples)
    
    int8_acc = evaluate_tflite_model(int8_path, data_dir,
                                     img_size=(args.img_size, args.img_size),
                                     num_samples=args.num_eval_samples)
    
    # Calculate accuracy drop
    acc_drop = (fp32_acc - int8_acc) * 100
    
    print("\n" + "="*60)
    print("Quantization Results:")
    print("="*60)
    print(f"FP32 Accuracy: {fp32_acc:.4f} ({fp32_acc*100:.2f}%)")
    print(f"INT8 Accuracy: {int8_acc:.4f} ({int8_acc*100:.2f}%)")
    print(f"Accuracy Drop: {acc_drop:.2f}%")
    
    if acc_drop < 5.0:
        print("✅ Accuracy drop is acceptable (<5%)")
    else:
        print("⚠️  Warning: Accuracy drop exceeds 5% threshold")
        print("   Consider quantization-aware training")
    
    print("="*60)
    
    # Save results
    results_file = project_root / 'results' / 'quantization_results.txt'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("Model Quantization Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"FP32 Model: {fp32_path}\n")
        f.write(f"INT8 Model: {int8_path}\n\n")
        f.write(f"Model Sizes:\n")
        f.write(f"  FP32: {fp32_path.stat().st_size / (1024*1024):.2f} MB\n")
        f.write(f"  INT8: {int8_path.stat().st_size / (1024*1024):.2f} MB\n")
        f.write(f"  Compression: {fp32_path.stat().st_size / int8_path.stat().st_size:.2f}x\n\n")
        f.write(f"Accuracy:\n")
        f.write(f"  FP32: {fp32_acc:.4f} ({fp32_acc*100:.2f}%)\n")
        f.write(f"  INT8: {int8_acc:.4f} ({int8_acc*100:.2f}%)\n")
        f.write(f"  Drop: {acc_drop:.2f}%\n")
    
    print(f"\n✅ Results saved to: {results_file}")
    
    print("\n" + "="*60)
    print("✅ Quantization complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Compile for Edge TPU: edgetpu_compiler model_int8.tflite")
    print("2. This step requires Edge TPU Compiler (will do on Raspberry Pi)")
    print("3. Deploy to Raspberry Pi + Coral USB Accelerator")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

