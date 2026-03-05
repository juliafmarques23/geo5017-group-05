# GEO5017 - Machine Learning (2026) Assignment 1

**Group 05:**  
Chaeyeon Moon (6477453)  
Evangelia Palli (6435939)  
Julia Fossa Marques (6550975)  

### Project: Linear Regression / Drone Trajectory Analysis

#### Description: 

This project implements polynomial regression using gradient descent to model the motion of a drone in 3D space. Two motion models are considered:

Constant velocity
Constant acceleration

The program fits the models to observed positions and predicts the drone's next position (t=7) based on the constant acceleration model. It also plots the 3D trajectory using Plotly.

The main function was specifically designed to answer the questions of this assignment, using the data provided for it; it is not necessary to provide any aditional arguments to reproduce the results shown in the report.

Setup

The program works with minimal setup; the only modules required, besides Python's standard library, are numpy and plotly.

Install numpy (directly from the command prompt)

pyhton -m pip install numpy

Install plotly (directly from the command prompt)

python -m pip install plotly

Parameters:
t : array_like
Timestamps of the position measurements.

positions : array_like
Measured positions in relation to an specific axis. Must have the same lenght as t.

learning_rate : float, optional
Learning rate of the gradient descent algorithm. If it is not specified, the default value (0.001) is used.

Setup
Install numpy

pip install numpy

Install plotly

pip install plotly

Parameters:
t : array_like
Timestamps of the position measurements.

positions : array_like
Measured positions in relation to an specific axis. Must have the same lenght as t.

learning_rate : float, optional
Learning rate of the gradient descent algorithm. If it is not specified, the default value (0.001) is used.

max_iter : int, optional
Maximum number of times the parameters optimization is performed. If it is not specified, the default value (10,000) is used. If it is not specified, the default value (1e-8) is used.*

GEO5017 A1 – Group 05
Project: Linear Regression / Drone Trajectory Analysis

Authors:

Julia Marques
Chaeyeon Moon
Evangelia Palli
Description: This project implements polynomial regression using gradient descent to model the motion of a drone in 3D space. Two motion models are considered:

Constant velocity
Constant acceleration
The program fits the models to observed positions and predicts the drone's next position (t=7) based on the constant acceleration model. It also plots the 3D trajectory using Plotly.

File Structure:

code/
main.py # Main script containing gradient descent solver and plotting function
ReadMe.txt # Instructions and project info
Dependencies:

Python >= 3.*
NumPy
Plotly
Installation:

Install Python and pip (if not already installed)
Install required packages: pip install numpy plotly
How to Run:

Navigate to the 'code' folder
Run: python main.py
The program will output:
Velocity vector for constant velocity model
Residual errors for constant velocity and acceleration models
3D plot showing:
Observed points (t=1-6)
Fitted trajectory using constant acceleration model
Predicted position at t=7
Notes:

Learning rate and max iterations for gradient descent can be modified in main.py by changing the parameters 'learning_rate' and 'max_iter'
The code is fully reproducible w

ith the provided data points.
