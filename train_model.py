# train_model.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

print("TensorFlow Version:", tf.__version__)

def load_and_preprocess_data():
    """Loads and prepares the Sign Language MNIST dataset."""
    print("Loading data...")
    train_df = pd.read_csv('sign_mnist_train.csv')
    test_df = pd.read_csv('sign_mnist_test.csv')

    # Separate labels and features
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    X_train = train_df.drop('label', axis=1).values
    X_test = test_df.drop('label', axis=1).values

    # Reshape the data to be 28x28x1 images
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)


    # Normalize the pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # One-hot encode the labels
    num_classes = 25  # There are 24 classes, but labels go up to 24
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    print("Data preprocessed successfully.")
    return X_train, y_train, X_test, y_test, num_classes

def build_cnn_model(num_classes):
    """Builds, compiles, and returns the CNN model."""
    print("Building model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Model built and compiled.")
    model.summary()
    return model

def main():
    """Main function to run the training pipeline."""
    X_train, y_train, X_test, y_test, num_classes = load_and_preprocess_data()
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = build_cnn_model(num_classes)
    
    print("Starting model training...")
    history = model.fit(X_train, y_train,
                        epochs=15,
                        batch_size=32,
                        validation_data=(X_val, y_val))
    
    print("Training finished.")

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")

    # Save the trained model
    model.save('asl_cnn_model.h5')
    print("Model saved as asl_cnn_model.h5")

if __name__ == '__main__':
    main()