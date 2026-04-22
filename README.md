# Advanced Drowsiness Detection System (IoT)

A real-time, hardware-integrated Drowsiness Detection System built for the Raspberry Pi 4. 

This project goes beyond standard Eye Aspect Ratio (EAR) checks by utilizing **Google MediaPipe Face Mesh (468 landmarks)** for highly robust facial tracking, and introduces a novel **Blink Quality Degradation Index** to detect drowsiness *3-5 seconds before* the eyes fully close.

It features a live, lightweight **Flask Web Stream** for headless remote monitoring and direct **GPIO integration** to sound a physical hardware buzzer when the driver falls asleep.

## Key Features

- **Robust Tracking**: Uses MediaPipe instead of jittery Haar Cascades for pinpoint eye tracking.
- **Early Warning Algorithm**: Monitors blink speed degradation to catch drowsiness early.
- **IoT Web Streaming**: Runs headless on the Pi but broadcasts a live 30FPS video feed with UI overlays to any web browser on your local network.
- **Hardware Alerts**: Triggers a physical buzzer via GPIO when critical drowsiness is detected.
- **Auto-Calibration**: Automatically learns the driver's baseline blink speed during the first 5 blinks.

## 🚀 Quick Start (Raspberry Pi Setup)

The system is optimized to run on a headless Raspberry Pi using a Python 3.11 Miniforge environment.

### 1. Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y python3-opencv libgl1-mesa-glx
```

### 2. Setup Python Environment
*(We recommend Miniforge to easily get Python 3.11 on ARM64)*
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/bin/activate

conda create -n drowsiness python=3.11 -y
conda activate drowsiness
```

### 3. Install Python Libraries
```bash
pip install "numpy<2"
pip install mediapipe flask
pip install opencv-python-headless==4.9.0.80
pip install rpi-lgpio  # Modern replacement for RPi.GPIO
```

### 4. Run the Engine
Run the detector without trying to open a desktop window (`--no-display`) and spin up the web server (`--web-stream`).
```bash
python drowsiness_detector.py --no-display --web-stream
```

### 5. View the Live Stream
Open a web browser on your laptop/phone connected to the same WiFi and navigate to:
**`http://<YOUR_PI_IP_ADDRESS>:5000`**

---

## ⚡ Hardware Wiring (Buzzer)

The system is configured to trigger **BCM Pin 17** when a critical drowsiness event occurs.

**Basic "Lite" Wiring (Direct to GPIO):**
1. Connect the Positive (+) leg of the Active Buzzer directly to **Physical Pin 11 (GPIO 17)**.
2. Connect the Negative (-) leg of the Buzzer directly to **Physical Pin 6 (Ground)**.

*Note: For long-term deployment, it is recommended to use an NPN Transistor (e.g., 2N2222) and a 330Ω resistor to drive the buzzer from the 5V power rail (Pin 2) to protect the Pi's GPIO pins from drawing too much current.*

## ⚙️ Configuration

You can tune the system parameters in `drowsiness_detector_config.json`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ear_threshold` | 0.22 | The EAR value below which the eye is considered closed. |
| `min_blink_speed` | 3 | Minimum frames a blink must last to be considered valid (filters noise). |
| `sustained_closure_frames` | 15 | Number of frames eyes must be closed to trigger an instant critical alert. |
| `gpio_buzzer` | 17 | The BCM pin number for the hardware buzzer. |

## 📁 Repository Structure

```text
├── core_engine.py                  # Handles EAR math, blink degradation logic, and GPIO
├── drowsiness_detector.py          # Main MediaPipe pipeline, UI renderer, and Flask server
├── drowsiness_detector_config.json # Tunable parameters
├── test_functions.py               # Unit tests
└── requirements.txt                # Dependency list
```
