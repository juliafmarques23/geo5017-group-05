# GEO5017 - Machine Learning (2026) Assignment 1

**Group 05**  
Chaeyeon Moon (6477453)  
Evangelia Palli (6435939)  
Julia Fossa Marques (6550975)  


## Project: Linear Regression / Drone Trajectory Analysis

### Description
This project implements polynomial regression using gradient descent to model the motion of a drone in 3D space. Two motion models are considered:

- Constant velocity
- Constant acceleration

The program fits the models to observed positions and predicts the drone's next position (t=7) based on the constant acceleration model. It also plots the 3D trajectory using Plotly.

The main function was specifically designed to answer the questions of this assignment, using the data provided for it; it is not necessary to provide any aditional arguments to reproduce the results shown in the report.

#### Repository Structure
```
/code/
   └── main.py  # Main script containing gradient descent solver and plotting function
ReadMe.txt      # Instructions and project info
```

### Setup
The program works with minimal setup; the only modules required, besides Python's standard library, are **numpy** and **plotly**.

**Install Numpy** 
```
pyhton -m pip install numpy
```
**Install Plotly** 
```
python -m pip install plotly
```

The following **parameters** are tunable:
- _learning_rate_ : float
Learning rate of the gradient descent algorithm. If it is not specified, the default value (**0.001**) is used.

- _max_iter_ : int
Maximum number of times the parameters optimization is performed. If it is not specified, the default value (**10,000**) is used.

### How to Run

1. Navigate to the `/code/` folder
2. Run `main.py`

### Expected output
- Velocity vector for constant velocity model
- Residual errors for constant velocity and acceleration models
- 3D plot showing:
  - Observed points ($t$ = 1 ~ 6)
  - Fitted trajectory using constant acceleration model
  - Predicted position at $t$ = 7
 
**Notes**
The code is fully reproducible with the provided data points.
