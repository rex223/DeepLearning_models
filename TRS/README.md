# 🚗 Traffic Sign Recognition System (GTSRB-Based)

An end-to-end deep learning pipeline for **real-time German traffic sign recognition**, built using **TensorFlow/Keras** and **opencv-python**, with a modern **Streamlit web app** interface.  
This project is inspired by the [GTSRB (German Traffic Sign Recognition Benchmark)](https://benchmark.ini.rub.de/) challenge and focuses on **accurately classifying road signs** from real-world images.

---

## 🔥 Features

- ✅ Trained on the official **GTSRB dataset** (43 classes of German signs)
- 🧠 Uses a **CNN classifier** trained from scratch with 95%+ accuracy
- 🕹️ **Streamlit web app** for easy testing and user interaction

---

## 🧪 Live Demo

🌐 **Try it here:**  
[🔗 Open the Web App](https://gtrsbchallenge-v1pred.streamlit.app/)  
> Upload a photo of a German traffic sign and get real-time predictions!

[🔗 PPT + DEMO link](https://www.canva.com/design/DAGmrtOFxFk/wDazzaR3fPo37WVM9QX2Wg/edit?utm_content=DAGmrtOFxFk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)  

---

## 🚀 Project Structure

```bash
.
├── streamlit_TRSapp.py        # Frontend for image input
├── final_model.h5             # Trained CNN classifier
├── preprocess_image.py        # Handles resizing & preprocessing
├── predict_image.py           # Maps prediction to sign class & warning
└── README.md                  # This file
