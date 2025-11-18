"""
Model Evaluation Script
Comprehensive evaluation of trained models with confusion matrix and metrics
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def load_model(model_path):
    """Load trained model"""
    print(f"\n📦 Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully")
    
    return model


def prepare_test_data(data_dir, img_size=(224, 224), batch_size=32):
    """Prepare test data generator"""
    print("\n📊 Preparing test data...")
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        data_dir / 'test',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )
    
    print(f"✅ Test samples: {test_generator.samples}")
    print(f"✅ Classes: {list(test_generator.class_indices.keys())}")
    
    return test_generator


def evaluate_model(model, test_generator):
    """Evaluate model and get predictions"""
    print("\n🧪 Evaluating model...")
    
    # Get predictions
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("="*60)
    
    return y_true, y_pred, predictions


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Create and save confusion matrix"""
    print("\n📊 Creating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Confusion matrix (counts)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Confusion matrix (percentages)
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                ax=ax2, cbar_kws={'label': 'Percentage (%)'})
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to: {output_path}")
    plt.close()
    
    return cm


def plot_per_class_accuracy(y_true, y_pred, output_path):
    """Plot per-class accuracy"""
    print("\n📈 Creating per-class accuracy chart...")
    
    # Calculate per-class accuracy
    accuracies = []
    for i in range(len(EMOTION_LABELS)):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum()
            accuracies.append(acc * 100)
        else:
            accuracies.append(0)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(EMOTION_LABELS, accuracies, color='#3498db', edgecolor='black')
    
    # Color bars based on accuracy
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        if acc >= 85:
            bar.set_color('#27ae60')  # Green
        elif acc >= 70:
            bar.set_color('#f39c12')  # Orange
        else:
            bar.set_color('#e74c3c')  # Red
    
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add target line
    ax.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Target (85%)')
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Per-class accuracy chart saved to: {output_path}")
    plt.close()


def generate_classification_report(y_true, y_pred, output_path):
    """Generate and save classification report"""
    print("\n📋 Generating classification report...")
    
    report = classification_report(y_true, y_pred, 
                                   target_names=EMOTION_LABELS,
                                   digits=4)
    
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(report)
    print("="*60)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"✅ Classification report saved to: {output_path}")


def analyze_errors(y_true, y_pred, predictions, test_generator, output_path, top_n=10):
    """Analyze worst predictions"""
    print(f"\n🔍 Analyzing top {top_n} worst predictions...")
    
    # Calculate confidence for each prediction
    confidences = np.max(predictions, axis=1)
    correct = y_true == y_pred
    
    # Find incorrect predictions with high confidence (false positives)
    incorrect_mask = ~correct
    incorrect_indices = np.where(incorrect_mask)[0]
    incorrect_confidences = confidences[incorrect_mask]
    
    # Sort by confidence (descending)
    sorted_indices = incorrect_indices[np.argsort(-incorrect_confidences)]
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write("Error Analysis - Top Confident Misclassifications\n")
        f.write("="*80 + "\n\n")
        
        for idx in sorted_indices[:top_n]:
            true_label = EMOTION_LABELS[y_true[idx]]
            pred_label = EMOTION_LABELS[y_pred[idx]]
            confidence = confidences[idx] * 100
            
            f.write(f"Sample #{idx}:\n")
            f.write(f"  True Label: {true_label}\n")
            f.write(f"  Predicted: {pred_label} (confidence: {confidence:.2f}%)\n")
            f.write(f"  All probabilities: {predictions[idx]}\n")
            f.write("-" * 80 + "\n")
        
        # Summary statistics
        f.write("\n" + "="*80 + "\n")
        f.write("Error Summary:\n")
        f.write("="*80 + "\n")
        f.write(f"Total samples: {len(y_true)}\n")
        f.write(f"Correct predictions: {correct.sum()} ({correct.sum()/len(y_true)*100:.2f}%)\n")
        f.write(f"Incorrect predictions: {(~correct).sum()} ({(~correct).sum()/len(y_true)*100:.2f}%)\n")
        f.write(f"Average confidence (correct): {confidences[correct].mean()*100:.2f}%\n")
        f.write(f"Average confidence (incorrect): {confidences[~correct].mean()*100:.2f}%\n")
    
    print(f"✅ Error analysis saved to: {output_path}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate facial expression recognition model')
    parser.add_argument('--model', type=str, default='models/baseline_fp32_best.h5',
                       help='Path to model file')
    parser.add_argument('--data-dir', type=str, default='data/fer2013',
                       help='Path to dataset directory')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Setup paths
    model_path = project_root / args.model
    data_dir = project_root / args.data_dir
    results_dir = project_root / 'results'
    charts_dir = results_dir / 'charts'
    
    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    
    # Prepare test data
    test_generator = prepare_test_data(data_dir, 
                                       img_size=(args.img_size, args.img_size),
                                       batch_size=args.batch_size)
    
    # Evaluate model
    y_true, y_pred, predictions = evaluate_model(model, test_generator)
    
    # Generate visualizations and reports
    plot_confusion_matrix(y_true, y_pred, 
                         charts_dir / 'confusion_matrix.png')
    
    plot_per_class_accuracy(y_true, y_pred,
                           charts_dir / 'per_class_accuracy.png')
    
    generate_classification_report(y_true, y_pred,
                                  results_dir / 'classification_report.txt')
    
    analyze_errors(y_true, y_pred, predictions, test_generator,
                  results_dir / 'error_analysis.txt')
    
    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")
    print(f"Charts saved to: {charts_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

