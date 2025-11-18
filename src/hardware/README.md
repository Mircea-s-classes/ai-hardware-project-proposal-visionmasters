# Hardware Integration Guide

This directory contains scripts for deploying the model on Raspberry Pi + Google Coral USB Accelerator.

## Hardware Setup

### Required Hardware

- **Raspberry Pi 4** (4GB RAM)
- **Google Coral USB Accelerator**
- **USB Webcam** (720p, 30fps recommended)
- **MicroSD Card** (64GB, Class 10)
- **Power Supply** (5V, 3A recommended for Pi 4)
- **HDMI Monitor** and cable
- **USB Keyboard** and Mouse (for initial setup)

### Raspberry Pi Setup

#### 1. Install Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Flash Raspberry Pi OS (64-bit) to microSD card
3. Boot Raspberry Pi and complete initial setup

#### 2. System Update

```bash
sudo apt update
sudo apt upgrade -y
```

#### 3. Install Dependencies

```bash
# Python and pip
sudo apt install python3-pip python3-venv -y

# OpenCV dependencies
sudo apt install libopencv-dev python3-opencv -y

# Other dependencies
sudo apt install libatlas-base-dev libhdf5-dev libjpeg-dev libpng-dev -y
```

#### 4. Setup Virtual Environment

```bash
cd /home/pi/
git clone <your-repo-url> emotion-recognition
cd emotion-recognition

python3 -m venv venv
source venv/bin/activate

# Install basic requirements
pip install numpy opencv-python matplotlib pillow
```

### Coral Edge TPU Setup

#### 1. Add Coral Repository

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt update
```

#### 2. Install Edge TPU Runtime

**For maximum performance:**
```bash
sudo apt install libedgetpu1-std
```

**For reduced power consumption:**
```bash
sudo apt install libedgetpu1-max
```

Note: The "max" version may cause the USB Accelerator to run hot. Start with "std" for stability.

#### 3. Install PyCoral

```bash
pip install pycoral
```

#### 4. Install TFLite Runtime

```bash
pip install tflite-runtime
```

#### 5. Test Coral USB Accelerator

```bash
# Plug in Coral USB Accelerator
lsusb | grep "Google"
```

You should see output like:
```
Bus 001 Device 002: ID 1a6e:089a Global Unichip Corp.
```

#### 6. Run Test Inference

```bash
# Download test model
wget https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite

# Run test
python3 -c "from pycoral.utils import edgetpu; print('Edge TPU available:', edgetpu.list_edge_tpus())"
```

If successful, you should see the Edge TPU device listed.

### Camera Setup

#### 1. Test Camera

```bash
# List video devices
ls -l /dev/video*

# Test with OpenCV
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera working:', cap.isOpened()); cap.release()"
```

#### 2. Adjust Camera Settings (Optional)

```bash
# Install v4l-utils
sudo apt install v4l-utils

# View camera capabilities
v4l2-ctl --list-devices
v4l2-ctl --list-formats-ext

# Set camera resolution and FPS
v4l2-ctl --set-fmt-video=width=640,height=480,pixelformat=MJPG
```

## Model Deployment

### 1. Transfer Models to Raspberry Pi

From your development machine:

```bash
# Using SCP
scp models/model_int8.tflite pi@<raspberry-pi-ip>:/home/pi/emotion-recognition/models/
scp models/model_int8_edgetpu.tflite pi@<raspberry-pi-ip>:/home/pi/emotion-recognition/models/
```

Or use a USB drive or GitHub.

### 2. Compile Model for Edge TPU (if not done)

On Raspberry Pi:

```bash
# Install Edge TPU Compiler
sudo apt install edgetpu-compiler

# Compile model
cd /home/pi/emotion-recognition/models
edgetpu_compiler model_int8.tflite
```

This will generate `model_int8_edgetpu.tflite`.

### 3. Transfer Project Files

Copy or clone your entire project to the Pi:

```bash
cd /home/pi/
git clone <your-repo-url> emotion-recognition
cd emotion-recognition
```

### 4. Install Project Dependencies

```bash
source venv/bin/activate
pip install -r requirements_pi.txt
```

Create `requirements_pi.txt`:
```
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=9.0.0
mediapipe>=0.10.0
pycoral
tflite-runtime
```

## Running the Demo

### Basic Demo

```bash
source venv/bin/activate
python src/hardware/inference_demo.py
```

### With Options

```bash
# Specify model
python src/hardware/inference_demo.py --model models/model_int8_edgetpu.tflite

# Disable Edge TPU (CPU only)
python src/hardware/inference_demo.py --no-edgetpu

# Show FPS
python src/hardware/inference_demo.py --display-fps

# Use different camera
python src/hardware/inference_demo.py --camera 1
```

### Autostart on Boot (Optional)

Create a systemd service:

```bash
sudo nano /etc/systemd/system/emotion-recognition.service
```

Add:
```ini
[Unit]
Description=Emotion Recognition Demo
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/emotion-recognition
Environment="DISPLAY=:0"
ExecStart=/home/pi/emotion-recognition/venv/bin/python src/hardware/inference_demo.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable emotion-recognition
sudo systemctl start emotion-recognition
```

## Performance Optimization

### 1. CPU Governor

Set CPU to performance mode:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 2. Disable Desktop Environment (Optional)

For headless operation with lower overhead:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

To re-enable desktop:
```bash
sudo systemctl set-default graphical.target
sudo reboot
```

### 3. Increase GPU Memory

Edit `/boot/config.txt`:
```bash
sudo nano /boot/config.txt
```

Add or modify:
```
gpu_mem=128
```

Reboot:
```bash
sudo reboot
```

### 4. Overclock (Use with Caution)

Edit `/boot/config.txt`:
```
over_voltage=2
arm_freq=1750
```

Note: This may void warranty and requires good cooling.

## Power Measurement

### Using USB Power Meter

1. Connect USB power meter between power supply and Pi
2. Run demo application
3. Record idle and active power consumption

### Software Monitoring

```bash
# Install vcgencmd
vcgencmd measure_temp    # CPU temperature
vcgencmd measure_volts   # Voltage
```

## Troubleshooting

### Coral USB Accelerator Not Detected

1. **Check USB connection**: Use USB 3.0 port if available
2. **Update runtime**: 
   ```bash
   sudo apt update
   sudo apt install --only-upgrade libedgetpu1-std
   ```
3. **Check permissions**: 
   ```bash
   sudo usermod -aG plugdev $USER
   ```
4. **Reboot**: `sudo reboot`

### Camera Not Working

1. **Check connection**: `ls -l /dev/video*`
2. **Test with raspistill** (for Pi Camera):
   ```bash
   raspistill -o test.jpg
   ```
3. **Enable camera interface**: `sudo raspi-config` → Interface Options → Camera

### Out of Memory

1. **Increase swap**:
   ```bash
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile
   # Set CONF_SWAPSIZE=2048
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

2. **Close other applications**
3. **Use lighter desktop environment** or run headless

### Slow Performance

1. **Verify Edge TPU is being used**: Check logs for "Using Edge TPU: True"
2. **Check CPU temperature**: `vcgencmd measure_temp`
3. **Add cooling**: Heatsinks or fan
4. **Optimize preprocessing**: Reduce image size or face detection frequency

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| FPS | >30 | With face detection + inference |
| Inference Latency | <20ms | Edge TPU only |
| Total Latency | <50ms | Face detection + preprocessing + inference |
| Power | <5W | Entire system (Pi + Coral + Camera) |
| Temperature | <70°C | With passive cooling |

## Next Steps

After successful deployment:

1. **Benchmark Performance**: Run `benchmarks/benchmark_raspberry_pi.py`
2. **Optimize Pipeline**: Profile and optimize bottlenecks
3. **Add Features**: Emotion history, statistics, UI improvements
4. **Create Demo Video**: Record for final presentation

## References

- [Coral USB Accelerator Guide](https://coral.ai/docs/accelerator/get-started/)
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [PyCoral API](https://coral.ai/docs/reference/py/)
- [TFLite Runtime](https://www.tensorflow.org/lite/guide/python)
