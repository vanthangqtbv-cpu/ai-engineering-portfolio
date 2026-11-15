# 🔄 Transfer Learning & Data Augmentation with MobileNetV2 on CIFAR-10

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Architecture](https://img.shields.io/badge/Backbone-MobileNetV2_(ImageNet)-0A66C2)](./transfer_learning_mobilenetv2_cifar10.ipynb)
[![Test Accuracy](https://img.shields.io/badge/Test_Accuracy-76.50%25_(%2B31.10%25_vs_Scratch)-2ea44f)](./transfer_learning_mobilenetv2_cifar10.ipynb)
[![Trainable Params](https://img.shields.io/badge/Trainable_Params-12%2C810_(50.04_KB)-purple)](./transfer_learning_mobilenetv2_cifar10.ipynb)

## 📌 Executive Summary
Training deep Convolutional Neural Networks from scratch on small datasets often leads to severe overfitting or slow convergence. In [**Project 03 (`Mini-ResNet`)**](../03-modern-cnn-resnet-cifar10/), training a residual network from scratch on a subset of `CIFAR-10` achieved **`45.40%` test accuracy** while requiring `45,130` trainable parameters.

In this project, we apply **Transfer Learning** using a pre-trained **`MobileNetV2`** backbone (trained on **1.4 million ImageNet images**) combined with an in-graph **Data Augmentation** pipeline (`RandomFlip`, `RandomRotation`, `RandomZoom`). By freezing the `2.25M`-parameter convolutional base and training only a lightweight **`12,810`-parameter classification head** for 6 epochs on just `4,000` training images, we boost unseen test accuracy from **`45.40%` to `76.50%` (+31.10% absolute improvement)** in ~22 seconds on CPU.

---

## 🏗️ End-to-End Functional API Architecture

To prevent data-leakage and double-normalization bugs, both spatial upsampling (`32x32` $\rightarrow$ `96x96`) and `MobileNetV2` pixel scaling (`[0, 255]` $\rightarrow$ `[-1, 1]`) are encapsulated directly inside the Keras Functional API graph:

```text
Input Layer: (32, 32, 3) Raw RGB [0..255]
   │
   ▼
Data Augmentation (Active only during training):
   ├── RandomFlip("horizontal")
   ├── RandomRotation(0.1)
   └── RandomZoom(0.1)
   │
   ▼
Spatial Upsampling: Resizing(96, 96)
   │
   ▼
Backbone Normalization: mobilenet_v2.preprocess_input (scales [0..255] -> [-1..1])
   │
   ▼
Frozen Pre-trained Backbone: MobileNetV2 (include_top=False, weights="imagenet", trainable=False)
   │  Output Feature Maps: (None, 3, 3, 1280) — 2,257,984 Non-Trainable Params
   ▼
Custom Classification Head (12,810 Trainable Params):
   ├── GlobalAveragePooling2D() -> (None, 1280)
   ├── Dropout(0.3)
   └── Dense(10, activation="softmax")
```

### Parameter Breakdown
| Component | Parameters | Memory Footprint | Status |
| :--- | :---: | :---: | :---: |
| **`MobileNetV2` Backbone (`ImageNet`)** | `2,257,984` | `8.61 MB` | ❄️ **Frozen (`trainable=False`)** |
| **Custom Head (`GAP` + `Dropout(0.3)` + `Dense(10)`)** | **`12,810`** | **`50.04 KB`** | 🔥 **Trainable** |
| **Total Model** | `2,270,794` | `8.66 MB` | ✅ Ready for Edge / TFLite Export |

---

## 📊 Visual Results & Benchmark Comparison

### 1. Real-Time Data Augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`)
Generating synthetic spatial variations on-the-fly forces the classification head to learn invariant object representations rather than memorizing pixel coordinates:

![Data Augmentation Preview](./assets/data_augmentation_preview.png)

### 2. Learning Curves (6 Epochs on CPU)
Because `MobileNetV2` already extracts rich hierarchical visual features (edges, textures, object parts), validation accuracy reaches **`69.00%` in Epoch 1** and climbs smoothly to **`76.25%` by Epoch 6** without overfitting (`Dropout(0.3)` + `Data Augmentation` keep training loss higher than validation loss):

![MobileNetV2 Learning Curves](./assets/mobilenetv2_learning_curves.png)

### 3. Unseen Test Set Predictions (`9/10` Correct on Sample Strip)
Evaluating on `1,000` unseen test images achieves **`76.50%` Test Accuracy**, correctly classifying **`9 out of 10`** sample test images (`Cat`, `Ship`, `Airplane`, `Frog`, `Automobile`):

![MobileNetV2 Test Predictions](./assets/mobilenetv2_test_predictions.png)

### 4. Head-to-Head Benchmark: Scratch `Mini-ResNet` vs. `MobileNetV2` Transfer Learning

| Metric | Project 03: Scratch `Mini-ResNet` | Project 04: `MobileNetV2` Transfer Learning | Engineering Impact |
| :--- | :---: | :---: | :--- |
| **Training Dataset Size** | `8,000` RGB images | **`4,000` RGB images** | **50% less training data required** |
| **Trainable Parameters** | `45,130` (`176.29 KB`) | **`12,810` (`50.04 KB`)** | **71.6% fewer trainable weights** |
| **Training Time (CPU)** | ~36s (8 epochs) | **~22s (6 epochs, ~3.5s/epoch)** | **1.6x faster convergence** |
| **Unseen Test Accuracy** | `45.40%` | **`76.50%`** | **🚀 `+31.10%` absolute accuracy jump** |
| **Sample Strip Accuracy** | `4 / 10` (`40%`) | **`9 / 10` (`90%`)** | Robust feature generalization |

---

## 📂 Repository Structure
```text
04-transfer-learning-mobilenetv2-cifar10/
├── transfer_learning_mobilenetv2_cifar10.ipynb  # Full executed Jupyter Notebook
├── README.md                                    # Engineering documentation & benchmark analysis
└── assets/
    ├── data_augmentation_preview.png            # 1 original vs. 5 augmented training samples
    ├── mobilenetv2_learning_curves.png          # 6-epoch Accuracy & Loss curves
    └── mobilenetv2_test_predictions.png         # 10 unseen test predictions (9/10 correct)
```
