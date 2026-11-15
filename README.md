# 🚀 Applied AI & Deep Learning Engineering Portfolio

[![Author](https://img.shields.io/badge/Author-Bui_Van_Thang-0A66C2?logo=github&logoColor=white)](https://github.com/vanthangqtbv-cpu)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flutter](https://img.shields.io/badge/Flutter-Edge_AI_%26_Mobile-02569B?logo=flutter&logoColor=white)](https://flutter.dev/)

Welcome to my **Applied AI & Deep Learning Engineering Portfolio**. This repository showcases production-oriented Machine Learning, Computer Vision, Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), and Multi-Agent Automation projects with a strong focus on **real-world business impact, data integrity, and end-to-end deployment**.

---

## 🗂️ Featured Projects & Modules

| # | Project / Module | Domain | Key Tech Stack | Highlight Metric / Outcome | Status |
| :-: | :--- | :--- | :--- | :--- | :-: |
| **01** | [**Hotel Booking Cancellation Prediction**](./01-hotel-booking-cancellation-ml/) | Tabular ML & Deep Neural Networks (MLP) | `Python`, `Pandas`, `Scikit-Learn`, `TensorFlow/Keras` | **`86.25%` Accuracy (`0.9333` ROC-AUC)** across `119,210` bookings; full data-leakage & anomaly audit | ✅ Completed |
| **02** | [**CNN Handwritten Digit Recognition & Error Analysis**](./02-cnn-mnist-digit-recognition/) | Computer Vision (CNN) | `TensorFlow/Keras`, `Conv2D`, `MaxPooling2D`, `Seaborn` | **`99.19%` Test Accuracy** (`9,919/10,000` unseen samples); idempotent pipeline & error diagnostics | ✅ Completed |
| **03** | [**Modern CNN: Mini-ResNet with Skip Connections (`CIFAR-10`)**](./03-modern-cnn-resnet-cifar10/) | Deep CV Architectures | `Keras Functional API`, `Add (Skip Connection)`, `BatchNormalization`, `GlobalAveragePooling2D` | **`45,450` Lightweight Params** (`90%` reduction vs. `Flatten`); residual identity & $1\times 1$ projection blocks on RGB images | ✅ Completed |
| **04** | [**Transfer Learning & Data Augmentation (`MobileNetV2` on `CIFAR-10`)**](./04-transfer-learning-mobilenetv2-cifar10/) | Transfer Learning & Edge CV | `MobileNetV2 (ImageNet)`, `Data Augmentation`, `Resizing`, `Feature Extraction` | **`76.50%` Test Accuracy (`+31.10%` vs. Scratch CNN)** training only **`12,810` params (`50 KB`)** in 22s on CPU | ✅ Completed |
| **05** | **Real-Time Object Detection (`YOLO` / Mobile Inspection)** | Computer Vision & Edge AI | `YOLOv8`, `OpenCV`, `TFLite` | Real-time pest & object detection for mobile field operations | 🔄 In Progress |
| **06** | **Transformer Fine-Tuning & Enterprise RAG Chatbot** | NLP & GenAI / LLMs | `HuggingFace`, `LangChain`, `Vector DB`, `Gemini / OpenAI` | Grounded Q&A over enterprise PDFs with semantic retrieval & guardrails | 📅 Upcoming |
| **07** | **Autonomous Multi-Agent B2B Workflow Engine** | Agentic AI | `Multi-Agent`, `MCP`, `FastAPI`, `Playwright` | Automated B2B lead intelligence, document extraction & quotation generation | 📅 Upcoming |

---

## 🔬 Visual Highlights

### 1. Transfer Learning & Data Augmentation with `MobileNetV2` ([`04-transfer-learning-mobilenetv2-cifar10`](./04-transfer-learning-mobilenetv2-cifar10/))
* Leveraged an ImageNet pre-trained **`MobileNetV2`** backbone (`trainable=False`) with in-graph **Data Augmentation** (`RandomFlip`, `RandomRotation`, `RandomZoom`) and spatial upsampling (`32x32` $\rightarrow$ `96x96`), boosting `CIFAR-10` test accuracy from **`45.40%` (Scratch Mini-ResNet) to `76.50%` (+31.10%)** while training only **`12,810` parameters (`50.04 KB`)**.

| Transfer Learning Curves (6 Epochs on CPU) | Unseen Test Predictions (`9/10` Correct) |
| :---: | :---: |
| ![MobileNetV2 Learning Curves](./04-transfer-learning-mobilenetv2-cifar10/assets/mobilenetv2_learning_curves.png) | ![MobileNetV2 Predictions](./04-transfer-learning-mobilenetv2-cifar10/assets/mobilenetv2_test_predictions.png) |

---

### 2. Modern CNN: Mini-ResNet with Residual Skip Connections ([`03-modern-cnn-resnet-cifar10`](./03-modern-cnn-resnet-cifar10/))
* Designed a custom **Mini-ResNet** using the Keras Functional API (`Conv2D` + `BatchNormalization` + `Add()` Identity & $1\times 1$ Projection Skip Connections + `GlobalAveragePooling2D`), reducing parameter count to just **`45,450` (`177.54 KB`)** on 3-channel RGB `CIFAR-10` images.

| Mini-ResNet Learning Curves (`EarlyStopping`) | Unseen RGB Test Predictions (`2x5` Grid) |
| :---: | :---: |
| ![ResNet Learning Curves](./03-modern-cnn-resnet-cifar10/assets/resnet_learning_curves.png) | ![ResNet Predictions](./03-modern-cnn-resnet-cifar10/assets/resnet_test_predictions.png) |

---

### 3. Computer Vision: CNN Feature Extraction & Error Diagnostics ([`02-cnn-mnist-digit-recognition`](./02-cnn-mnist-digit-recognition/))
* Built a 421K-parameter Convolutional Neural Network (`Conv2D` $\rightarrow$ `MaxPooling2D` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Softmax`) achieving **99.19% test accuracy** in 10 epochs.

| Training & Validation Curves | Confusion Matrix (`10,000` Unseen Test Images) |
| :---: | :---: |
| ![Training Curves](./02-cnn-mnist-digit-recognition/assets/02_training_loss_accuracy.png) | ![Confusion Matrix](./02-cnn-mnist-digit-recognition/assets/05_confusion_matrix.png) |

---

### 4. Tabular ML vs. Deep Learning Benchmark ([`01-hotel-booking-cancellation-ml`](./01-hotel-booking-cancellation-ml/))
* Benchmarked **Logistic Regression**, **Random Forest**, and **12 Multi-Layer Perceptron (MLP)** architectures across `119,210` cleaned hotel reservations after eliminating target leakage (`reservation_status`) and domain anomalies (`total_nights == 0`, room type mismatches).

| MLP Optimizer & Learning Rate Search | Final Benchmark (LogReg vs. Random Forest vs. Best MLP) |
| :---: | :---: |
| ![MLP Tuning](./01-hotel-booking-cancellation-ml/assets/chart_cell_23_07.png) | ![Final Benchmark](./01-hotel-booking-cancellation-ml/assets/chart_cell_24_09.png) |

---

## 🛠️ Core Engineering Principles
1. **Idempotent & Reproducible Pipelines:** Every preprocessing step includes explicit shape and range assertions so cells and scripts can be re-executed safely without state corruption.
2. **Data-First Auditing:** Uncovering domain traps, target leakage, and edge-case anomalies before training complex architectures.
3. **Business-Driven Metrics:** Optimizing for precision, recall, F1-score, ROC-AUC, and inference latency rather than vanity training accuracy.

---

## 📫 Contact & Collaboration
* **Author:** Bùi Văn Thắng (`vanthangqtbv-cpu`)
* **Email:** [vanthangqtbv@gmail.com](mailto:vanthangqtbv@gmail.com)
* **Focus Areas:** Applied AI Engineering, Computer Vision, Enterprise RAG & Multi-Agent Automation, Cross-Platform Mobile (Flutter)
