"""
Baseline MobileNetV2 Model Training Script
Trains a facial expression recognition model on FER2013 dataset
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configuration
CONFIG = {
    'data_dir': project_root / 'data' / 'fer2013',
    'model_dir': project_root / 'models',
    'results_dir': project_root / 'results',
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'num_classes': 7,
    'seed': 42
}

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def setup_gpu():
    """Configure GPU settings"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ {len(gpus)} GPU(s) available and configured")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
    else:
        print("ℹ️  No GPU found, using CPU")


def create_data_generators():
    """Create data generators for training and validation"""
    print("\n📊 Setting up data generators...")
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2  # Use 20% of training data for validation
    )
    
    # Test data (no augmentation, only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        CONFIG['data_dir'] / 'train',
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=CONFIG['seed'],
        color_mode='rgb'
    )
    
    val_generator = train_datagen.flow_from_directory(
        CONFIG['data_dir'] / 'train',
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=CONFIG['seed'],
        color_mode='rgb'
    )
    
    test_generator = test_datagen.flow_from_directory(
        CONFIG['data_dir'] / 'test',
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )
    
    print(f"✅ Train samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Test samples: {test_generator.samples}")
    print(f"✅ Class indices: {train_generator.class_indices}")
    
    return train_generator, val_generator, test_generator


def build_model():
    """Build MobileNetV2-based model for facial expression recognition"""
    print("\n🏗️  Building MobileNetV2 model...")
    
    # Load pre-trained MobileNetV2 (without top layer)
    base_model = MobileNetV2(
        input_shape=(*CONFIG['img_size'], 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build complete model
    inputs = keras.Input(shape=(*CONFIG['img_size'], 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(CONFIG['num_classes'], activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    print(f"✅ Model built successfully")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    return model


def create_callbacks():
    """Create training callbacks"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        ModelCheckpoint(
            filepath=CONFIG['model_dir'] / f'baseline_fp32_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=CONFIG['results_dir'] / 'logs' / timestamp,
            histogram_freq=1
        )
    ]
    
    return callbacks


def plot_training_history(history, output_path):
    """Plot and save training history"""
    print("\n📈 Creating training history plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training history saved to: {output_path}")
    plt.close()


def evaluate_model(model, test_generator):
    """Evaluate model on test set"""
    print("\n🧪 Evaluating model on test set...")
    
    results = model.evaluate(test_generator, verbose=1)
    
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
    if len(results) > 2:
        print(f"Test Top-2 Accuracy: {results[2]:.4f} ({results[2]*100:.2f}%)")
    print("="*60)
    
    return results


def fine_tune_model(model, train_generator, val_generator, initial_epochs=20):
    """Fine-tune the model by unfreezing some layers"""
    print("\n🔧 Fine-tuning model (unfreezing base layers)...")
    
    # Unfreeze base model
    base_model = model.layers[1]
    base_model.trainable = True
    
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    # Continue training
    fine_tune_epochs = 20
    total_epochs = initial_epochs + fine_tune_epochs
    
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=total_epochs,
        initial_epoch=initial_epochs,
        callbacks=create_callbacks(),
        verbose=1
    )
    
    return history_fine


def main():
    """Main training function"""
    print("\n" + "="*60)
    print("Baseline MobileNetV2 Training")
    print("="*60)
    
    # Create directories
    CONFIG['model_dir'].mkdir(parents=True, exist_ok=True)
    CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
    (CONFIG['results_dir'] / 'charts').mkdir(parents=True, exist_ok=True)
    
    # Setup
    setup_gpu()
    tf.random.set_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Check if dataset exists
    if not (CONFIG['data_dir'] / 'train').exists():
        print(f"\n❌ Dataset not found at {CONFIG['data_dir']}")
        print("Please run: python src/model/prepare_data.py")
        sys.exit(1)
    
    # Prepare data
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Build model
    model = build_model()
    model.summary()
    
    # Train model
    print("\n🚀 Starting training...")
    print(f"   Epochs: {CONFIG['epochs']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Learning rate: {CONFIG['learning_rate']}")
    print("="*60 + "\n")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['epochs'],
        callbacks=create_callbacks(),
        verbose=1
    )
    
    # Save final model
    final_model_path = CONFIG['model_dir'] / 'baseline_fp32_final.h5'
    model.save(final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")
    
    # Plot training history
    plot_training_history(
        history,
        CONFIG['results_dir'] / 'charts' / 'training_history.png'
    )
    
    # Evaluate on test set
    test_results = evaluate_model(model, test_gen)
    
    # Save results
    results_file = CONFIG['results_dir'] / 'training_results.txt'
    with open(results_file, 'w') as f:
        f.write("Baseline MobileNetV2 Training Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Image size: {CONFIG['img_size']}\n")
        f.write(f"  Batch size: {CONFIG['batch_size']}\n")
        f.write(f"  Epochs: {CONFIG['epochs']}\n")
        f.write(f"  Learning rate: {CONFIG['learning_rate']}\n\n")
        f.write(f"Test Results:\n")
        f.write(f"  Test Loss: {test_results[0]:.4f}\n")
        f.write(f"  Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)\n")
        if len(test_results) > 2:
            f.write(f"  Test Top-2 Accuracy: {test_results[2]:.4f} ({test_results[2]*100:.2f}%)\n")
    
    print(f"\n✅ Results saved to: {results_file}")
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review results in results/")
    print("2. Run evaluation: python src/model/evaluate.py")
    print("3. Quantize model: python src/model/quantize_model.py")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

