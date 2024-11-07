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

To run this project on Google Colab, follow these steps:

1. **Open the `cancer_detection.ipynb` Notebook**:
   - Upload the `cancer_detection.ipynb` notebook to Google Colab or open it directly if it's hosted on GitHub.

2. **Set Up the Dataset**:
   - Download the dataset from Kaggle if it’s not already in Google Drive.
   - Upload the dataset to Google Drive or use Kaggle’s API to download it directly in Colab. To use Kaggle’s API:
     - First, upload your Kaggle API token (`kaggle.json`) to Colab.
     - Run the following code to install and authenticate Kaggle:
       ```python
       from google.colab import files
       files.upload()  # Upload kaggle.json

       ! mkdir -p ~/.kaggle
       ! cp kaggle.json ~/.kaggle/
       ! chmod 600 ~/.kaggle/kaggle.json
       ```
     - Download the dataset with:
       ```python
       ! kaggle datasets download -d prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed
       ```
   - Alternatively, **mount Google Drive** and access the dataset stored there:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

3. **Install Project Dependencies**:
   - Run the following code in Colab to install all dependencies listed in `requirements.txt`:
     ```python
     !pip install -r requirements.txt
     ```
   - This will install packages for image processing, machine learning, deep learning, and other utilities required for the notebook.

4. **Run Each Cell in the Notebook**:
   - The notebook is organized with sections for image preprocessing, feature extraction, machine learning, and deep learning models.
   - Follow the instructions within each cell to apply feature extraction techniques and train both ML and DL models.

5. **View Results**:
   - The notebook includes visualizations of feature extraction techniques (showing the first five images for each).
   - It will also display classification accuracy and confusion matrices for each model, allowing for a detailed performance comparison.

By following these steps, you’ll be able to set up and run the entire project on Google Colab with minimal hassle.


