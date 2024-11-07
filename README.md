# Cervical-Cancer-Cell-Detection
## Project Overview
This project explores the automated detection of cervical cancer cells in Pap smear images using both traditional computer vision techniques and modern deep learning architectures. Using the **SIPaKMeD** dataset, the study investigates the effects of preprocessing strategies—specifically **Otsu’s thresholding** and **watershed segmentation**—on classification accuracy. Feature extraction is performed using five traditional methods, and the resulting features are classified using various machine learning models. Additionally, deep learning architectures are applied directly to the original images to provide a comparative performance analysis.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Feature Extraction Techniques](#feature-extraction-techniques)
- [Classification Models](#classification-models)
- [Deep Learning Architectures](#deep-learning-architectures)
- [Running the Project on Google Colab](#running-the-project-on-google-colab)

## Dataset
The dataset used in this project is **SIPaKMeD**, which is available on Kaggle: [Cervical Cancer Largest Dataset - SIPaKMeD](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed).

The dataset contains five distinct categories of cervical cell images:
1. **Dyskeratotic Cells**
2. **Koilocytotic Cells**
3. **Metaplastic Cells**
4. **Parabasal Cells**
5. **Superficial-Intermediate Cells**

Each category has images of Pap smear cells to train and validate the detection model. These images are organized into folders by category and are preprocessed with **Otsu’s thresholding** and **watershed segmentation** to enhance detection accuracy.

## Technologies Used
- **Python**: Core programming language.
- **OpenCV**: Image processing and feature extraction.
- **scikit-image**: Advanced image processing techniques.
- **scikit-learn**: Machine learning model training and evaluation.
- **Deep Learning Frameworks**: Implemented using **TensorFlow** or **PyTorch** for deep learning architectures.
- **Google Colab**: Cloud-based environment for running Python code and deep learning models.

## Feature Extraction Techniques
1. **HOG (Histogram of Oriented Gradients)**: Captures gradient and edge information.
2. **SIFT (Scale-Invariant Feature Transform)**: Detects and describes distinctive keypoints.
3. **ORB (Oriented FAST and Rotated BRIEF)**: A fast, rotation-invariant alternative to SIFT.
4. **LBP (Local Binary Pattern)**: Analyzes texture patterns around each pixel.
5. **Gabor Filters**: Highlights specific frequencies and orientations, useful for texture analysis.

Each technique is applied to segmented images after preprocessing, and the resulting features are used for training the classification models.

## Classification Models
The following machine learning classifiers are applied to classify the extracted features:
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

## Deep Learning Architectures
In addition to traditional computer vision techniques, three deep learning models are applied directly to the original images to explore classification performance further:
- **AlexNet**
- **VGG19**
- **ResNet**

The deep learning models are evaluated on various performance metrics, including accuracy and confusion matrices, to highlight strengths and potential improvements over traditional feature-based methods.
## Running the Project on Google Colab
To run this project on Google Colab:
1. **Upload the `cancer_detection.ipynb` notebook** to Google Colab.
2. **Set up the dataset**:
   - Upload the dataset to your Google Drive.
   - Mount Google Drive in the Colab notebook:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Update the dataset path in the notebook to match the location in your Google Drive.
3. **Run each cell**:
   - The notebook is organized with sections for image preprocessing, feature extraction, machine learning, and deep learning models.
   - Follow the notebook instructions to apply feature extraction techniques and train both ML and DL models.
4. **View Results**:
   - The notebook includes visualizations of feature extractions (first five images for each technique).
   - It prints the classification accuracy for each model, along with confusion matrices for deep learning models.


