# geo5017-group-05
GEO5017 A1 – Linear Regression

This project implements polynomial regression models to analyze and predict the 3D motion of a drone using gradient descent.

Folder structure:
- code/
  - main.py
  - trajectory.py

Requirements:
- Python 3.x
- numpy
- plotly (for visualization)

How to run:
1. Open a terminal and navigate to the 'code' directory.
2. Run the main script using:
   python main.py

The script will:
- Estimate the constant velocity model
- Estimate the constant acceleration model
- Compute residual errors
- Predict the next drone position
- Plot the observed and predicted trajectories

Learning rate and number of iterations can be adjusted directly in the function calls inside main.py.
The code is self-contained and reproduces all reported results without modification.
