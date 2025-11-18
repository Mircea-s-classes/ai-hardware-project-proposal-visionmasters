# Dataset Information

## FER2013 Dataset

The **FER2013** (Facial Expression Recognition 2013) dataset is used for training and evaluating the emotion recognition model.

### Dataset Details

- **Classes**: 7 emotions
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral

- **Images**: ~35,000 images
- **Image Size**: 48x48 pixels (grayscale in original, but we use augmented color versions)
- **Splits**: Train and Test

### Download Instructions

#### Option 1: Kaggle API (Recommended)

1. **Install Kaggle API**:
```bash
pip install kaggle
```

2. **Setup Kaggle Credentials**:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Download `kaggle.json`
   - Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download Dataset**:
```bash
cd data/fer2013
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip
rm fer2013.zip
```

#### Option 2: Manual Download

1. Visit: https://www.kaggle.com/datasets/msambare/fer2013
2. Click "Download" button
3. Extract the zip file to `data/fer2013/`
4. Ensure the structure is:
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
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

### Verify Dataset

After downloading, run:
```bash
python src/model/prepare_data.py
```

This will:
- Verify dataset structure
- Display dataset statistics
- Create sample visualizations
- Generate class distribution charts

## Clash Royale Emotes

Emote images and sounds should be placed in `data/emotes/`:

```
data/emotes/
├── images/
│   ├── angry.png
│   ├── disgust.png
│   ├── fear.png
│   ├── happy.png
│   ├── sad.png
│   ├── surprise.png
│   └── neutral.png
└── sounds/
    ├── angry.mp3
    ├── disgust.mp3
    ├── fear.mp3
    ├── happy.mp3
    ├── sad.mp3
    ├── surprise.mp3
    └── neutral.mp3
```

You can:
1. Use emotes from the reference `clash-royale-emote-detector/` directory
2. Find Clash Royale emote assets online
3. Create custom emotes

### Emotion to Emote Mapping

| Emotion | Suggested Emote |
|---------|----------------|
| Happy | 😂 Laughing King / Crying Laughing |
| Sad | 😢 Crying Face |
| Angry | 😠 Angry Face / Red Face |
| Surprise | 😲 Shocked / Wow Face |
| Fear | 😱 Screaming Face |
| Disgust | 🤢 Sick Face / Vomit |
| Neutral | 👍 Thumbs Up / Good Game |

## Dataset Statistics

After running `prepare_data.py`, you'll see statistics like:

```
Emotion      Train      Test     Total
------------------------------------------------
Angry        3,995      958      4,953
Disgust      436        111      547
Fear         4,097      1,024    5,121
Happy        7,215      1,774    8,989
Sad          4,830      1,247    6,077
Surprise     3,171      831      4,002
Neutral      4,965      1,233    6,198
------------------------------------------------
TOTAL        28,709     7,178    35,887
```

Note: The dataset is **imbalanced** - Happy and Sad emotions have more samples than Disgust. This should be considered during training.

## License

FER2013 dataset is publicly available for research and educational purposes.
