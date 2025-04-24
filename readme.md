Plant Disease Classifier

A deep learning-based project to classify plant leaf diseases using Convolutional Neural Networks (CNNs) and deploy predictions through an interactive Streamlit web application.

Project Overview
The Plant Disease Classifier leverages a Convolutional Neural Network (CNN) to identify plant leaf diseases from high-resolution images. Trained on the PlantVillage Dataset, the model classifies 38 disease categories across multiple crops (e.g., Apple, Grape, Tomato). The trained model is integrated into a user-friendly Streamlit web app, allowing users to upload leaf images and receive real-time disease predictions. This tool aims to support farmers and researchers in early disease detection, promoting sustainable agriculture.

Features

Accurate Disease Classification: CNN model with ~94% training accuracy and ~91% validation accuracy.
Real-Time Predictions: Streamlit app processes images and delivers results in under 2 seconds.
User-Friendly Interface: Intuitive UI with image upload and clear prediction output.
Scalable Design: Modular code structure for easy extension (e.g., adding new crops or models).


Dataset

Source: PlantVillage Dataset (https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data)
Details: ~54,000 high-resolution RGB images of healthy and diseased plant leaves, covering 38 classes across crops like Apple, Grape, and Tomato.
Organization: Images are stored in class-specific subdirectories for easy integration with TensorFlow’s ImageDataGenerator.


Directory Structure
plant-disease-classifier/
├── notebooks/
│   └── project.ipynb               # Model training and evaluation code
├── app/
│   └── app.py                     # Streamlit web app code
├── data/
│   └── class_indices.json         # Class index to disease name mapping
├── logo.png                       # Web app logo
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation


Prerequisites

Python: Version 3.8 or higher
Hardware: GPU recommended for faster model training
Dependencies: Listed in requirements.txt


Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/plant-disease-classifier.git
cd plant-disease-classifier


Create a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download the Dataset:

Visit the PlantVillage Dataset.
Download and extract the dataset to a local directory (e.g., data/plantvillage/).
Update the dataset path in notebooks/project.ipynb if necessary.


Train the Model (optional):

Open notebooks/project.ipynb in Jupyter Notebook.
Run the cells to preprocess data, train the CNN, and save the model to model/plant_disease_prediction_model.h5.


Run the Streamlit App:
streamlit run app/app.py


Open the provided URL (e.g., http://localhost:8501) in a web browser.




Usage

Launch the Web App:

Run streamlit run app/app.py.
The app opens in your default browser.


Upload an Image:

Click the image uploader and select a leaf image (JPEG/PNG).
Ensure the image clearly shows the leaf for accurate predictions.


View Results:

The app displays the predicted disease name and confidence score.
Example output: “Tomato - Early Blight (92% confidence)”.




Technologies Used

TensorFlow/Keras: For building and training the CNN model.
Streamlit: For creating the interactive web app.
Pillow: For image processing.
NumPy: For numerical computations.
Jupyter Notebook: For model development and experimentation.


Results

Model Performance:
Training Accuracy: ~94.8%
Validation Accuracy: ~91.5%
Loss: ~0.15 (training), ~0.22 (validation)


App Performance: Processes images in ~1.5 seconds on standard hardware.
Use Case: Enables farmers and researchers to diagnose plant diseases quickly, reducing crop losses.


Future Enhancements

Implement transfer learning with pre-trained models (e.g., ResNet50, EfficientNet) for higher accuracy.
Expand the dataset to include additional crops and diseases.
Develop a mobile app version for field use.
Add multi-language support for global accessibility.


Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m 'Add feature').
Push to the branch (git push origin feature-name).
Open a pull request.


License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

PlantVillage: For providing the dataset.
TensorFlow & Streamlit Communities: For robust tools and documentation.


