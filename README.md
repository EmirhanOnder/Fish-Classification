
# Fish Species Classification Project

This project aims to classify various species of fish using image data. The dataset contains multiple classes of fish images, and the goal is to train an Artificial Neural Network (ANN) model to correctly classify each image into its corresponding species. The project utilizes Python libraries such as TensorFlow, Keras, NumPy, and Pandas to preprocess the data, build the model, and evaluate its performance.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Conclusion](#conclusion)

## Overview

This project builds and trains a neural network to classify images of different fish species into 9 classes. The project involves data loading, preprocessing, model building, training, and evaluation.

## Dataset

The dataset used in this project is hosted on Kaggle and contains images of different fish species organized into folders by class. Each image is in PNG format and is pre-labeled by its corresponding folder name.

## Requirements

To run this project, you need the following libraries:

- Python 
- TensorFlow 
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- PIL (Python Imaging Library)

You can install the necessary packages using:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow
```

## Project Structure

- **Dataset**: Contains fish images organized in folders by species.
- **Scripts**: Python scripts for data preprocessing, model training, and evaluation.
- **Plots**: Visualizations of model training and data characteristics.

## Data Preprocessing

1. **Loading Image Data**: 
   - The project loads the dataset using the file paths and extracts class labels based on folder names. Images are filtered to exclude folders containing "GT" in their names.

   ```python
   root_dir = Path('/kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset')
   filepaths = [str(path) for path in root_dir.glob("**/*.png") if "GT" not in str(path)]
   labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
   df = pd.DataFrame({'Filepath': filepaths, 'Class': labels})
   ```

2. **Visualizing Sample Images**:
   - A function is defined to plot sample images from each class to get a visual understanding of the dataset.

   ```python
   def plot_sample_images(df, num_samples=5):
       # Code to plot images from each class
   ```

3. **Analyzing Image Resolutions**:
   - Image resolutions are analyzed to check for consistency and visualize color histograms.

   ```python
   df['resolution'] = df['Filepath'].apply(lambda x: Image.open(x).size)
   ```

4. **Resizing and Normalizing Images**:
   - Images are resized to a standard size and normalized to improve model training efficiency.

   ```python
   img_size = (256, 256)
   def load_and_preprocess_image(filepath):
       # Code to load and preprocess images
   ```

5. **Splitting the Dataset**:
   - The dataset is split into training, validation, and test sets using stratified sampling.

   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
   ```

6. **Encoding and One-Hot Encoding Labels**:
   - Labels are encoded and transformed into a one-hot format.

   ```python
   le = LabelEncoder()
   y_train_cat = to_categorical(le.fit_transform(y_train), num_classes=9)
   ```

## Model Architecture

The neural network model is built using the Keras `Sequential` API. The model consists of multiple dense layers, batch normalization, dropout, and L2 regularization to improve generalization and prevent overfitting.

```python
model = Sequential()
model.add(Input(shape=(256 * 256 * 3,)))
# Code for adding layers
```

### Key Layers

- **Dense Layers with ReLU Activation**: For learning complex patterns.
- **Batch Normalization**: To stabilize and speed up training.
- **Dropout**: To prevent overfitting.
- **L2 Regularization**: To penalize large weights.

## Training the Model

The model is trained using the Adam optimizer and categorical cross-entropy loss function. Early stopping is used to halt training if validation accuracy does not improve.

```python
history = model.fit(
    X_train, y_train_cat,
    epochs=20,
    validation_data=(X_val, y_val_cat),
    callbacks=[early_stopping]
)
```

## Evaluation

The trained model is evaluated on the test dataset to determine its generalization performance.

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
```

## Visualization

1. **Plotting Loss and Accuracy**:
   - Loss and accuracy are plotted to visualize the training and validation performance over epochs.

   ```python
   plt.figure(figsize=(12, 5))
   # Code to plot loss and accuracy graphs
   ```

2. **Confusion Matrix and Classification Report**:
   - A confusion matrix and classification report are generated to evaluate model performance on each class.

## Conclusion

The project successfully builds and trains an artificial neural network to classify fish species based on images. The model achieves a certain level of accuracy on the test dataset and demonstrates the importance of data preprocessing, regularization, and hyperparameter tuning in improving model performance.

### Future Work

- Explore more complex neural network architectures such as Convolutional Neural Networks (CNNs) for image classification tasks.
- Experiment with different data augmentation techniques to improve model generalization.


Kaggle Link for Notebook:
https://www.kaggle.com/code/emirhanonder/akbank-project
