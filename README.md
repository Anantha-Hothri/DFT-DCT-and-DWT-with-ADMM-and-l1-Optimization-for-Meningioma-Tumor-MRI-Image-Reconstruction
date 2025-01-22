# DFT-DCT-and-DWT-with-ADMM-and-l1-Optimization-for-Meningioma-Tumor-MRI-Image-Reconstruction
In the context of modern medical imaging in MRI, it is becoming increasingly challenging to minimize the amount of time taken for scanning and data burden that is expected to be processed while still ensuring that high quality images are obtained. This study sought to compare the performances of different transforms in compressed sensing including DFT, DCT, and DWT based on conjugate ADMM and l1-optimization for reconstruction of meningioma tumor MRI images. The objective of this initiative is to entail the determination of the optimal transform and optimization methodologies whose outcomes will provide the best image reconstruction quality with the lowest computational complexities. Through the use of compressed sensing, the number of measurements used was much lower than normally expected, lowering the amount of data used. It was found  DWT may be preferred for non-time-sensitive applications that demand the highest quality images, DCT's balanced attributes render it generally the best choice for the broadest range of medical imaging applications, expected to lead to a significant enhancement in the MRI diagnosis, patient comfort and the overall healthcare delivery system costs. The report outlines the method utilized for this study and the results observed using these techniques in real-world medical environments.

## **Table of Contents**

1. [About the Project](#about-the-project)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Publication Details](#publication-details)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

---

### **About the Project**

This project explores the application of compressed sensing techniques combined with transformation methods (DFT, DCT, DWT) and optimization algorithms (ADMM, \( l_1 \)-Optimization) for reconstructing high-quality meningioma tumor MRI images. It aims to reduce scanning times and computational complexity while enhancing the accuracy of medical diagnostics, addressing challenges in resource-constrained healthcare settings.

---

### **Features**

- Comparison of DFT, DCT, and DWT for image reconstruction.
- Implementation of compressed sensing using ADMM and \( l_1 \)-Optimization.
- Evaluation based on Peak Signal-to-Noise Ratio (PSNR) and computational time.
- Optimization of image reconstruction for high diagnostic quality with minimal data.

---

### **Technologies Used**

- Programming Languages: Python
- Libraries: NumPy, Matplotlib, Scikit-learn
- Frameworks: Compressed Sensing Algorithms (custom implementation)
- Tools: MRI Image Processing and Analysis

---

### **Installation**

Follow the steps below to set up the project:

```bash
# Clone the repository
git clone https://github.com/username/repository-name.git

# Navigate to the project directory
cd repository-name

# Install required libraries
pip install -r requirements.txt
 bash```

---

### **Usage**

1. **Dataset**: Load the brain tumor dataset from Kaggle, containing MRI images of meningiomas.
   - The dataset includes diverse MRI scans essential for automatic detection and classification of meningioma tumors.
   - Use the Kaggle API or download directly from their website.
2. **Preprocessing**: Standardize image dimensions and normalize pixel values.
   - Resize all images to a uniform size (e.g., 128x128 pixels).
   - Normalize pixel intensity values to enhance algorithmic performance.
3. **Transformations**:
   - Apply **Discrete Fourier Transform (DFT)** for frequency domain analysis.
   - Use **Discrete Cosine Transform (DCT)** for energy compaction and contrast enhancement.
   - Apply **Discrete Wavelet Transform (DWT)** for multi-resolution analysis and edge detection.
4. **Optimization**:
   - **ADMM Optimization**: Decomposes complex optimization problems into smaller, manageable subproblems.
   - **\( l_1 \)-Optimization**: Promotes sparsity in solutions and reconstructs images from minimal samples.
5. **Evaluation**:
   - Assess the reconstructed image's quality using **Peak Signal-to-Noise Ratio (PSNR)**.
   - Measure the computational efficiency by recording the time required for reconstruction.
   - Visualize original and reconstructed images using Matplotlib.

---

### **Publication Details**

- **Paper Title**: Comparative Analysis of DFT, DCT, and DWT Transformations with ADMM and \( l_1 \)-Optimization for Compressed Sensing in Meningioma Tumor MRI Image Reconstruction.
- **Authors**: Advik Narendran, Anantha Hothri, Hemanth Saga, Sarada Jayan.
- **Conference**: 2024 First International Conference on Innovations in Communications, Electrical and Computer Engineering (ICICEC)
- **Publication Date**: 30 December 2024
- **DOI**: 10.1109/ICICEC62498.2024.10808572.
- **Citation**: A. Narendran, A. Hothri, H. Saga and S. Jayan, "Comparative Analysis of DFT, DCT, and DWT Transformations with ADMM and l1-Optimization for Compressed Sensing in Meningioma Tumor MRI Image Reconstruction," 2024 First International Conference on Innovations in Communications, Electrical and Computer Engineering (ICICEC), Davangere, India, 2024, pp. 1-7, doi: 10.1109/ICICEC62498.2024.10808572.

---

### **License**

This project is licensed under the MIT License.

---

### **Contact**

- **Name**: Anantha Hothri Inuguri
- **Email**: [bl.en.u4aie22003@bl.students.amrita.edu](mailto:bl.en.u4aie22003@bl.students.amrita.edu)
- **LinkedIn**: [Anantha's LinkedIn](https://www.linkedin.com/in/anantha-hothri)

---
