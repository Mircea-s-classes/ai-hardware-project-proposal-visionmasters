#!/bin/bash
# Script to prepare Clash Royale emotes from reference repository

echo "=============================================="
echo "Preparing Clash Royale Emotes"
echo "=============================================="

# Create emotes directories
mkdir -p data/emotes/images
mkdir -p data/emotes/sounds

# Check if reference repo exists
if [ ! -d "clash-royale-emote-detector" ]; then
    echo "❌ Reference repository not found: clash-royale-emote-detector/"
    echo "Please ensure the reference repo is in the project root."
    exit 1
fi

echo ""
echo "📁 Copying emote images..."

# Copy and rename images to match emotion labels
if [ -f "clash-royale-emote-detector/images/laughing.png" ]; then
    cp clash-royale-emote-detector/images/laughing.png data/emotes/images/happy.png
    echo "✅ Copied: happy.png"
fi

if [ -f "clash-royale-emote-detector/images/crying.png" ]; then
    cp clash-royale-emote-detector/images/crying.png data/emotes/images/sad.png
    echo "✅ Copied: sad.png"
fi

if [ -f "clash-royale-emote-detector/images/taunting.png" ]; then
    cp clash-royale-emote-detector/images/taunting.png data/emotes/images/angry.png
    echo "✅ Copied: angry.png"
fi

if [ -f "clash-royale-emote-detector/images/yawning.png" ]; then
    cp clash-royale-emote-detector/images/yawning.png data/emotes/images/neutral.png
    echo "✅ Copied: neutral.png"
fi

echo ""
echo "🔊 Copying emote sounds..."

# Copy and rename sounds
if [ -f "clash-royale-emote-detector/sounds/laughing.mp3" ]; then
    cp clash-royale-emote-detector/sounds/laughing.mp3 data/emotes/sounds/happy.mp3
    echo "✅ Copied: happy.mp3"
fi

if [ -f "clash-royale-emote-detector/sounds/crying.mp3" ]; then
    cp clash-royale-emote-detector/sounds/crying.mp3 data/emotes/sounds/sad.mp3
    echo "✅ Copied: sad.mp3"
fi

if [ -f "clash-royale-emote-detector/sounds/taunting.mp3" ]; then
    cp clash-royale-emote-detector/sounds/taunting.mp3 data/emotes/sounds/angry.mp3
    echo "✅ Copied: angry.mp3"
fi

if [ -f "clash-royale-emote-detector/sounds/yawning.mp3" ]; then
    cp clash-royale-emote-detector/sounds/yawning.mp3 data/emotes/sounds/neutral.mp3
    echo "✅ Copied: neutral.mp3"
fi

echo ""
echo "⚠️  Note: You still need emotes for:"
echo "   - disgust.png / disgust.mp3"
echo "   - fear.png / fear.mp3"
echo "   - surprise.png / surprise.mp3"
echo ""
echo "You can:"
echo "  1. Find these emotes online"
echo "  2. Create placeholder images"
echo "  3. Reuse existing emotes for these emotions"
echo ""
echo "=============================================="
echo "✅ Emote preparation complete!"
echo "=============================================="
echo ""
echo "Emotes location: data/emotes/"
echo ""

