# GEO5017 A1 – Group 05

Project: Linear Regression / Drone Trajectory Analysis

Authors:
- Julia Marques
- Chaeyeon Moon
- Evangelia Palli

Description:
This project implements polynomial regression using gradient descent 
to model the motion of a drone in 3D space. Two motion models are considered:
1) Constant velocity
2) Constant acceleration

The program fits the models to observed positions and predicts
the drone's next position (t=7) based on the constant acceleration model.
It also plots the 3D trajectory using Plotly.

File Structure:
- code/
    - main.py            # Main script containing gradient descent solver and plotting function
- ReadMe.txt             # Instructions and project info

Dependencies:
- Python >= 3.*
- NumPy
- Plotly

Installation:
1) Install Python and pip (if not already installed)
2) Install required packages:
   pip install numpy plotly

How to Run:
1) Navigate to the 'code' folder
2) Run:
   python main.py
3) The program will output:
   - Velocity vector for constant velocity model
   - Residual errors for constant velocity and acceleration models
   - 3D plot showing:
       - Observed points (t=1-6)
       - Fitted trajectory using constant acceleration model
       - Predicted position at t=7

Notes:
- Learning rate and max iterations for gradient descent can be modified
  in main.py by changing the parameters 'learning_rate' and 'max_iter'
- The code is fully reproducible with the provided data points.
