"""
FER2013 Dataset Preparation Script
Downloads and prepares the FER2013 dataset for training
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Emotion labels
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}


def check_dataset_exists(data_dir):
    """Check if FER2013 dataset exists"""
    data_path = Path(data_dir)
    
    # Check for common FER2013 directory structures
    possible_paths = [
        data_path / 'train',
        data_path / 'test',
        data_path / 'fer2013.csv',
    ]
    
    exists = any(p.exists() for p in possible_paths)
    
    if not exists:
        print("\n❌ FER2013 dataset not found!")
        print(f"Expected location: {data_path}")
        print("\nPlease download the dataset manually:")
        print("1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
        print("2. Download the dataset")
        print("3. Extract to: data/fer2013/")
        print("\nOr use Kaggle API:")
        print("   kaggle datasets download -d msambare/fer2013")
        print(f"   unzip fer2013.zip -d {data_path}")
        return False
    
    return True


def analyze_dataset(data_dir):
    """Analyze dataset statistics"""
    data_path = Path(data_dir)
    
    print("\n📊 Analyzing FER2013 Dataset...")
    print("=" * 60)
    
    # Count images per emotion and split
    stats = {
        'train': {label: 0 for label in EMOTION_LABELS.values()},
        'test': {label: 0 for label in EMOTION_LABELS.values()}
    }
    
    for split in ['train', 'test']:
        split_path = data_path / split
        if not split_path.exists():
            print(f"⚠️  Warning: {split} directory not found")
            continue
            
        for emotion_idx, emotion_name in EMOTION_LABELS.items():
            emotion_path = split_path / emotion_name.lower()
            if emotion_path.exists():
                count = len(list(emotion_path.glob('*.jpg'))) + len(list(emotion_path.glob('*.png')))
                stats[split][emotion_name] = count
    
    # Print statistics
    print(f"\n{'Emotion':<12} {'Train':>10} {'Test':>10} {'Total':>10}")
    print("-" * 60)
    
    total_train = 0
    total_test = 0
    
    for emotion in EMOTION_LABELS.values():
        train_count = stats['train'][emotion]
        test_count = stats['test'][emotion]
        total = train_count + test_count
        total_train += train_count
        total_test += test_count
        print(f"{emotion:<12} {train_count:>10,} {test_count:>10,} {total:>10,}")
    
    print("-" * 60)
    print(f"{'TOTAL':<12} {total_train:>10,} {total_test:>10,} {total_train + total_test:>10,}")
    print("=" * 60)
    
    return stats


def visualize_samples(data_dir, output_path='results/dataset_samples.png'):
    """Visualize sample images from each emotion class"""
    data_path = Path(data_dir) / 'train'
    
    if not data_path.exists():
        print("⚠️  Cannot create visualization: train directory not found")
        return
    
    print("\n🎨 Creating sample visualization...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (emotion_idx, emotion_name) in enumerate(EMOTION_LABELS.items()):
        emotion_path = data_path / emotion_name.lower()
        
        if emotion_path.exists():
            # Get first image
            image_files = list(emotion_path.glob('*.jpg')) + list(emotion_path.glob('*.png'))
            if image_files:
                img = Image.open(image_files[0])
                axes[idx].imshow(img, cmap='gray')
                axes[idx].set_title(emotion_name, fontsize=14, fontweight='bold')
                axes[idx].axis('off')
    
    # Remove extra subplot
    if len(EMOTION_LABELS) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    # Save visualization
    output_file = Path(project_root) / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Sample visualization saved to: {output_file}")
    plt.close()


def create_distribution_chart(stats, output_path='results/charts/class_distribution.png'):
    """Create bar chart showing class distribution"""
    print("\n📈 Creating class distribution chart...")
    
    emotions = list(EMOTION_LABELS.values())
    train_counts = [stats['train'][e] for e in emotions]
    test_counts = [stats['test'][e] for e in emotions]
    
    x = np.arange(len(emotions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_counts, width, label='Train', color='#3498db')
    bars2 = ax.bar(x + width/2, test_counts, width, label='Test', color='#e74c3c')
    
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('FER2013 Dataset - Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save chart
    output_file = Path(project_root) / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Distribution chart saved to: {output_file}")
    plt.close()


def main():
    """Main function"""
    data_dir = project_root / 'data' / 'fer2013'
    
    print("\n" + "="*60)
    print("FER2013 Dataset Preparation")
    print("="*60)
    
    # Check if dataset exists
    if not check_dataset_exists(data_dir):
        sys.exit(1)
    
    # Analyze dataset
    stats = analyze_dataset(data_dir)
    
    # Create visualizations
    visualize_samples(data_dir)
    create_distribution_chart(stats)
    
    print("\n" + "="*60)
    print("✅ Dataset preparation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated visualizations in results/")
    print("2. Run training: python src/model/train_baseline.py")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

