# University of Virginia

## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  
**Proposal Deadline:** November 5, 2025 — 11:59 PM  
**Submission:** Upload to Canvas (PDF) and to GitHub (`/docs` folder)

---

# AI Hardware Project Proposal Template

## 1. Project Title

**Real-Time Facial Expression Recognition on Edge AI Hardware with Interactive Emote Display**

**Team Name:** VisionMasters

**Team Members:**

- [Allen Chen] - [wmm7wr@virginia.edu]
- [Marvin Rivera] - [tkk9wg@virginia.edu]
- [Sami Kang] - [ajp3cx@virginia.edu]

## 2. Platform Selection

Select one platform category and justify your choice.

**Undergraduates:** Edge-AI, TinyML, or Neuromorphic platforms  
**Graduates:** open-source AI accelerators (Ztachip, VTA, Gemmini, VeriGOOD-ML, NVDLA) or any of the above

**Selected Platform:** Edge-AI Platform

**Specific Hardware:** Raspberry Pi 4 + Google Coral USB Accelerator (Edge TPU)

**Reasoning:**  
The Google Coral Edge TPU is specially designed for running AI tasks directly on devices with very low power usage (around 2 watts). This makes it a great fit for our facial expression recognition project because:

1. **Fast Response Time:** Its design allows it to quickly process video data in real-time, aiming for over 30 frames per second, so we can see expressions instantly.

2. **Low Power Use:** Since it's energy-efficient, it's perfect for devices that run on batteries or need to stay on all the time without draining power. The TPU uses hardware optimized for fast and efficient AI calculations.

3. **Real-World Edge Use:** It reflects the real limits of small, connected devices — they often have limited power, can't always connect to the internet, and need to process data instantly.

4. **Design and Programming Integration:** To work best with this hardware, models need to be simplified and fine-tuned specifically for the TPU, which highlights important aspects of building dedicated AI hardware.

Overall, this platform helps us understand and tackle key challenges in designing AI systems: how to keep models accurate while making them smaller and faster, how to get the best performance from special hardware, and how to deliver quick results without using much power.

## 3. Problem Definition

Real-time facial expression recognition means detecting and understanding people's emotions quickly enough to keep up with what they’re doing, usually requiring processing over 30 video frames each second with less than 50 milliseconds of delay. Doing this on small, battery-powered devices is challenging. Usually, there are two main approaches:

- Using powerful, energy-hungry graphics cards, which aren’t practical for small devices.
- Using simpler, less accurate models on regular computer chips, which improve efficiency but reduce accuracy.

**The AI Hardware Challenge:**

How can we develop systems that recognize facial expressions accurately (over 85% correct) at real-time speeds (more than 30 frames per second) while using very little power (less than 5 watts)?

**Why This Matters for AI Hardware:**

This project focuses on the key challenges in designing AI hardware:

1. **Efficiency:** We’re using special hardware called Edge TPU that works with lower-precision calculations (INT8). To keep accuracy high, we need to carefully optimize our models so they use less computation—about a quarter of the normal amount—without losing quality.

2. **Speed:** Processing video in real time means each frame must be handled in less than 33 milliseconds. We need to make sure all parts of the process—preparing data, running the AI model, and interpreting results—are optimized, considering limits like memory bandwidth and processing power.

3. **Scalability:** The solution should run on small devices with limited resources, like 4GB of RAM and ARM-based CPUs, proving that advanced AI can work outside of big servers.

4. **Hardware-Software Co-Design:** Achieving this requires designing the AI model (like MobileNet), choosing the right quantization methods (INT8), and matching these with hardware capabilities (TPU vs. CPU) in a way that all parts work together smoothly.

By solving this, we show how specially designed AI chips allow complex applications—like facial expression recognition—to run on tiny, portable devices, something that’s nearly impossible with general-purpose processors alone.

**Real-World Use Case:**

Our demo shows how facial expressions can be linked to game emojis in Clash Royale. This highlights possible uses such as:

- Gaming accessories that react to your emotions
- Devices to help people communicate without words
- Monitoring mental health by tracking emotions
- Smart home systems that can respond to how you’re feeling

## 4. Technical Objectives

1. **Real-Time Performance**: Ensure the system can process at least 30 frames per second from capturing the camera image to showing the emotion, with each frame being analyzed in less than 20 milliseconds.

2. **Power Efficiency**: Keep the total power used by the system (including the Raspberry Pi, Coral device, and camera) below 5 watts, showing that it can run efficiently on low power.

3. **Accuracy Preservation**: Keep at least 85% correct predictions on the FER2013 test set, even after simplifying the model to run faster, with less than 5% drop in accuracy compared to the original full-precision version.

4. **Quantization Optimization**: Use techniques to simplify the model after training that help it run better on the Edge TPU without losing much accuracy. This includes testing both post-training quantization and methods that prepare the model for quantization during training.

5. **System Optimization**: Analyze the entire setup to find and fix any slow parts, making the system run smoothly. This involves deciding which tasks run on the main processor (CPU) versus the specialized hardware (TPU), as well as optimizing memory usage and overall system performance.

## 5. Methodology

### **Hardware Setup**

**Devices Used:**

- Raspberry Pi 4 with 4GB of memory
- Google Coral USB Accelerator (special chip to speed up AI tasks)
- USB webcam that broadcasts at 720p quality and 30 frames per second
- A USB power meter to monitor how much power the system uses
- HDMI monitor to display real-time emotion icons

### **Software Used**

- **Operating System:** Raspberry Pi OS (64-bit version)
- **AI Software:** TensorFlow Lite with special support for the Coral device
- **Programming Language:** Python 3.9 or newer
- **Key Libraries:** OpenCV (for camera and image processing), NumPy, Pillow
- **Face Detection:** MediaPipe Face Detection (optimized for mobile devices)
- **Version Control:** Git and GitHub for managing code

### **How the Model Is Built and Made Efficient**

**Starting Point:**

- Uses MobileNetV2 (a lightweight neural network suitable for small devices)
- Trained to recognize 7 facial expressions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Initially trained on a large dataset called ImageNet
- Then fine-tuned with around 35,000 images from FER2013 dataset

**Making the Model Smaller and Faster:**

1. Train the full-precision (32-bit) model and see how accurate it is
2. Convert the model to a smaller, integer format (8-bit) to run faster on devices
3. If accuracy drops more than 5%, retrain the model with special techniques to keep accuracy high
4. Prepare the model specifically for the Coral Edge TPU chip, ensuring all parts run on the hardware without fallback to the main CPU

**Preparing the Input Data:**

- Use images of size 224x224 pixels in color
- Run face detection, crop the face, resize, and normalize the image on the CPU before sending it for prediction
- Run the emotion recognition on the TPU
- After prediction, process the results on the CPU to get the final emotion label and confidence level

### **Optimizing Performance**

1. Use profiling tools to find parts of the process that slow things down
2. Make sure the most demanding AI tasks are done on the TPU
3. Reduce unnecessary data movement between the CPU and TPU
4. Overlap camera capture and inference processes to save time

### **Measuring Performance**

- **Latency:** How long each step takes (detect face, preprocess, run prediction, interpret results)
- **Frames per second:** How many frames (images) are processed each second
- **Power Usage:** Measure power while the system is idle and when actively predicting, with a USB power meter
- **Accuracy:** How often the model correctly identifies emotions on test images
- **Resource Use:** Monitor CPU and memory usage, TPU activity, and temperature to ensure stability

### **Demo Application**

A live system that:

- Shows camera feed on the screen
- Draws boxes around faces
- Displays the predicted emotion and confidence
- Shows a matching game emote (like in Clash Royale)
- Shows an FPS counter for how fast it's working
- Displays system info like latency and power use

**Emotion-to-Emote Examples:**

- Happy → Laughing king icon
- Sad → Crying face
- Angry → Angry face icon
- Surprise → Shocked face
- Fear → Screaming face
- Disgust → Sick face
- Neutral → Thumbs-up

### **Testing and Validation**

1. **Functionality Checks:** Make sure it predicts correctly with different photos
2. **Performance Tests:** Record video clips to see how fast and smoothly it runs
3. **Accuracy Checks:** Test on standard datasets to see how well it recognizes emotions
4. **Long Runs:** Use the system for over an hour to check if it stays cool and performs reliably

## 6. Expected Deliverables

1. **Working Demo System**:

   - A live facial expression recognition system running in real-time on a Raspberry Pi with Coral hardware
   - An interactive display that shows your emotions as they are detected
   - A recorded video demonstrating how the system works
   - The ability to show a live demo during the final presentation

2. **GitHub Repository** (organized, well-documented, open to the public):

   - /src # Source code files
   - /models # Trained models (standard and smaller, optimized versions)
   - /data # Sample test images
   - /benchmarks # Scripts to measure system performance
   - /docs # Documentation and system diagrams
   - /results # Performance data (CSV files), charts
   - README.md # Instructions for setting up the system
   - requirements.txt

3. **Trained Models**:

   - The basic model with full precision (FP32)
   - A smaller, optimized version of the model (INT8 quantized TFLite, compatible with Edge TPU)
   - A report comparing how well each model performs

4. **Benchmark Results**:

   - Breakdown of system response times
   - Frame rates (how many images processed per second)
   - Power usage information
   - Comparison of accuracy between the full precision and optimized models
   - Charts and graphs showing performance data

5. **Technical Documentation**:

   - Diagram showing how the system is put together
   - Instructions for setting up the hardware
   - Explanation of how the model was trained and converted
   - Details about the optimizations made to improve performance
   - Analysis of where the system spends most of its time and potential bottlenecks

6. **Presentations**:

   - **Midterm**: Explanation of the problem, approach, initial results, and early performance tests
   - **Final**: Live demo of the complete system, detailed performance analysis, optimization learnings, and lessons learned

7. **Final Report**:
   - Introduction and reasons for doing the project
   - Background info on Edge AI and Edge TPU hardware
   - How the system was designed and built
   - How the models were optimized and converted
   - Results from testing and how the system performs
   - Combining hardware and software considerations
   - Challenges faced and how they were solved
   - Conclusions and ideas for future improvements
   - References to related work

## 7. Team Responsibilities

List each member’s main role.

| Name            | Role       | Responsibilities            |
| --------------- | ---------- | --------------------------- |
| [Marvin Rivera] | Team Lead  | Coordination, documentation |
| [Allen Chen]    | Hardware   | Setup, integration          |
| [Sami Kang]     | Software   | Model training, inference   |
| [ALL]           | Evaluation | Testing, benchmarking       |

## 8. Timeline and Milestones

| Week | Date   | Key Task                                     | What to Deliver                                                                                                                          |
| ---- | ------ | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Nov 5  | **Project Idea Proposal**                    | Submit a PDF and upload files to GitHub in the `/docs` folder                                                                            |
| 2    | Nov 12 | Setting up Hardware & Creating Initial Model | Get Raspberry Pi and Coral device working, train a basic facial recognition model using standard precision (FP32) on the FER2013 dataset |
| 3    | Nov 19 | Improving the Model & Using TPU Hardware     | Convert the model to run on Edge TPU with INT8 precision, get initial testing working, compare accuracy                                  |
| 4    | Nov 19 | **Midterm Presentation**                     | Prepare slides showing setup, model results, early testing, and any problems faced                                                       |
| 5    | Dec 3  | Adding App Features & Making It Faster       | Create a working demo that shows camera capturing, making predictions, and displaying results, start optimizing performance              |
| 6    | Dec 10 | Fine-Tuning Performance & Testing            | Measure how well everything works, identify and fix slow parts, run full performance tests                                               |
| 7    | Dec 17 | **Final Presentation & Submission**          | Submit the final report, give your presentation, and upload all project files with clear instructions to GitHub                          |

## 9. Resources Required

### **Hardware (Requesting from Course):**

- Raspberry Pi 4 (4GB RAM) — 1 unit
- Google Coral USB Accelerator — 1 unit
- MicroSD card (64GB) — 1 unit

### **Hardware:**

- USB webcam (720p) — ~$25 (team can purchase)
- USB power meter — ~$15 (for power measurements)
- HDMI monitor (using personal monitor)
- Micro-HDMI to HDMI cable (if needed)
- Webcam, monitor, and HDMI cable might not be needed, could use individual laptop components

### **Software & Cloud:**

- TensorFlow / TensorFlow Lite
- Google Colab
- Edge TPU Compiler
- MediaPipe
- Git/GitHub

### **Datasets:**

- FER2013
- Clash Royale emote images

### **Compute:**

- Personal laptops for development
- Google Colab free tier for model training

## 10. References

Include relevant papers, repositories, and documentation.

- https://www.electromaker.io/blog/article/coral-usb-accelerator-add-fast-edge-ai-to-any-host
- https://de.mathworks.com/company/technical-articles/what-is-int8-quantization-and-why-is-it-popular-for-deep-neural-networks.html
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11620061/
