Logistic Regression Model

This project demonstrates the implementation of a simple logistic regression model using Python. The code includes functions for data preprocessing, model initialization, training through gradient descent, and evaluating the accuracy of predictions.
Table of Contents

    Features
    Requirements
    Installation
    Usage
    Project Structure
    Future Improvements

Features

    Data Loading: Handles datasets in HDF5 format.
    Normalization: Scales features to a [0, 1] range for improved performance.
    Gradient Descent: Optimizes the model parameters using custom gradient calculations.
    Log-Loss Calculation: Computes error metrics to evaluate model performance.
    Prediction and Evaluation: Predicts test data and calculates model accuracy.

Requirements

To run this project, you need the following Python libraries:

    numpy
    scikit-learn
    h5py
    tqdm

You can install these dependencies with:

pip install numpy scikit-learn h5py tqdm

Installation

    Clone this repository:

git clone <repository_url>

Navigate to the project directory:

    cd <project_directory>

    Ensure you have the dataset files (trainset.hdf5 and testset.hdf5) in a datasets/ folder.

Usage

Run the main script to train the logistic regression model and evaluate its accuracy:

python test.py

Sample Output

The script outputs the training progress and the final accuracy score of the model on the test set.
Project Structure

.
├── test.py                 # Main script for the model
├── utilities.py            # Helper functions for data loading
├── datasets/
│   ├── trainset.hdf5       # Training dataset
│   └── testset.hdf5        # Testing dataset
└── README.md               # Project documentation

Key Functions
test.py

    init_function: Initializes weights and biases.
    linear_function: Implements the linear component of the model.
    sigmoid_function: Applies the sigmoid activation function.
    log_loss_function: Calculates the logistic loss.
    gradients_function: Computes gradients for optimization.
    update_value_w_b: Updates weights and biases.
    gradients_descent_function: Optimizes parameters using gradient descent.
    normalize_min_max_method: Normalizes data to a [0, 1] range.
    flatten_train_test: Reshapes input datasets.
    predict_function: Generates predictions based on trained weights.

utilities.py

    load_data: Loads training and testing datasets from HDF5 files.

Future Improvements

    Add support for more advanced optimization techniques (e.g., Adam, RMSprop).
    Extend the project to handle multi-class classification.
    Include a more detailed progress report with visualization of training metrics.