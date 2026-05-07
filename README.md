# 🩺 Medical Image Segmentation with UNet Model

This project features segmenting medical images of cardiac disease using UNet model built from scratch based on the official paper.

You can also deploy this model using streamlit and convert your saved model to onnx file using corresponding files. Demo can viewed using the following link: [Medical_Image_Segmentation](https://akhra92-medical-image-segmentation.streamlit.app/)

---

## 📊 Project Overview

- **Task**: Medical Image Semantic Segmentation
- **Dataset**: Medical Images of Cardiac Disease
- **Model**: UNet built from scratch, or a pretrained SegFormer (selectable via `MODEL_TYPE` in `config.py`)
- **Framework**: PyTorch
- **Evaluation Metrics**: mean intersection over union, pixel accuracy, and generated masks

---

## 🏗️ UNet Model

The UNet model used is a Pytorch implementation of the original UNet based on the official paper.

---

## 📈 Training & Validation Curves

Here is the **Pixel Accuracy** curve over epochs:

### Pixel Accuracy
![PA](plots/pa_curve.png)

---

## 🧮 Mean Intersection Over Union (mIoU)

The **mIoU** plot below illustrates the model's performance in the given dataset:

![mIoU](plots/iou_curve.png)

---

## 🧠 Mask Visualization

Generated masks of the model:

![Masks](inference_results/inference_visualization.png)

---


## 🚀 How to Run

1. Clone the repository:

   ```
   git clone https://github.com/akhra92/Cardiac-Semantic-Segmentation.git
   cd Cardiac-Semantic-Segmentation
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Train and test the model:

   ```
   python main.py
   ```

4. Deploy locally using streamlit:

   ```
   streamlit run demo.py
   ```

5. Convert to onnx file:

   ```
   python convert_onnx.py
   ```



