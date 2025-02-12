# 🏥 Skin Cancer Classification using Supervised & Unsupervised Learning  

## 📌 Overview  
This project focuses on classifying skin cancer images as **benign** or **malignant** using various **feature extraction methods** and **machine learning techniques**. The classification is performed using **supervised learning (Random Forest)** and **unsupervised learning (K-Means Clustering)** to analyze model performance across different feature sets.  

## 🎯 Objectives  
- Classify skin cancer images into **benign** or **malignant** categories.  
- Compare different **feature extraction techniques** for skin cancer detection.  
- Evaluate **supervised vs. unsupervised** learning approaches.  

## 📊 Dataset  
- **Source**: Skin cancer image dataset with predefined train/test splits.  
- **Train/Test Ratios**: 50:50, 30:70, 70:30.  

## 🔬 Feature Extraction Methods  
The following **texture, color, and spatial features** are used for classification:  
✅ **Haralick** – Texture patterns from gray-level co-occurrence matrix  
✅ **LBP (Local Binary Patterns)** – Texture descriptor capturing micro-patterns  
✅ **Gabor Filters** – Multi-scale frequency and orientation analysis  
✅ **Wavelet Features** – Multi-resolution analysis of image structure  
✅ **Tamura Features** – Human perception-based texture descriptors  
✅ **Markovian Features** – Probabilistic texture modeling  
✅ **Color Features** – Statistical analysis of color channels  
✅ **Fractal Dimension** – Complexity measure of texture patterns  
✅ **Run-Length Features** – Gray-level run-length matrix-based texture description  
✅ **WLD (Weber Local Descriptor)** – Local contrast and orientation analysis  

## 🏗️ Methodology  
1️⃣ **Image Preprocessing** – Standardizing input images for consistency.  
2️⃣ **Feature Extraction** – Generating meaningful feature representations.  
3️⃣ **Classification Approaches**:  
   - **Supervised Learning**: Random Forest Classifier  
   - **Unsupervised Learning**: K-Means Clustering  
4️⃣ **Performance Evaluation** – Using accuracy, precision, recall, F1-score, and confusion matrices.  

## 📏 Evaluation Metrics  
📌 **Accuracy** – Overall correctness of the model.  
📌 **Precision** – Correctly identified malignant cases / total identified malignant cases.  
📌 **Recall** – Correctly identified malignant cases / total actual malignant cases.  
📌 **F1-Score** – Harmonic mean of precision and recall.  
📌 **Confusion Matrix** – Visual representation of classification performance.  
