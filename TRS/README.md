# 🚗 Traffic Sign Recognition System (GTSRB-Based)

An end-to-end deep learning pipeline for **real-time German traffic sign recognition**, built using **TensorFlow/Keras** and **YOLOv8**, with a modern **Streamlit web app** interface.  
This project is inspired by the [GTSRB (German Traffic Sign Recognition Benchmark)](https://benchmark.ini.rub.de/) challenge and focuses on **accurately classifying road signs** from real-world images or camera input.

---

## 🔥 Features

- ✅ Trained on the official **GTSRB dataset** (43 classes of German signs)
- 📸 Supports **live camera** or **uploaded image** inputs
- 🧠 Uses a **CNN classifier** trained from scratch with 96%+ accuracy
- 🎯 Integrated with **YOLOv8** to first detect road signs in noisy images before classifying them
- 🕹️ **Streamlit web app** for easy testing and user interaction
- 🚙 Planned deployment on an **Autonomous Rover** for real-world traffic sign response

---

## 🧪 Live Demo

🌐 **Try it here:**  
[🔗 Open the Web App](https://gtrsbchallenge-v1pred.streamlit.app/)  
> Upload or take a photo of a German traffic sign and get real-time predictions!

---

## 🚀 Project Structure

```bash
.
├── streamlit_TRSapp.py        # Frontend for camera & image input
├── final_model.h5             # Trained CNN classifier
├── best.pt                    # YOLOv8 trained detector for road signs
├── preprocess_image.py        # Handles resizing & preprocessing
├── predict_image.py           # Maps prediction to sign class & warning
└── README.md                  # This file
