# GEO5017 - Machine Learning (2026) Assignment 2

**Group 05**  
Chaeyeon Moon (6477453)  
Evangelia Palli (6435939)  
Julia Fossa Marques (6550975)  


## Project: Objects Classification from AirBorne LiDAR Data

### Description
This project aims to classify a dataset of 500 pre-segmented urban point clouds into five classes: building, car, fence, pole, and tree. To achieve this, we developed a feature-based classification code using two supervised learning classifiers: Support Vector Machine (SVM) and Random Forest (RF).


to briefly explain how to run the code and
reproduce the results, e.g., dependence on external libraries/packages (in-
cluding the commands for installing them), 
the path to data, 


The main function was specifically designed to answer the questions of this assignment, using the data provided for it; it is not necessary to provide any aditional arguments to reproduce the results shown in the report.

#### Repository Structure
```
/A2/
  └── /code/
         └── main.py       # Main script containing gradient descent solver and plotting function
         └── ReadMe.txt    # Instructions and project info
```

### Setup
The program works with minimal setup; Besides Python's standard library, the following 5 libraries are required. **matplotlib**, **numpy**, **sklearn**, **scipy**, and **tqdm**.

**Command for Installation** 
```
python -m pip install matplotlib numpy scikit-learn scipy tqdm
```

### How to Run

1. Navigate to the `/code/` folder
2. Run `main.py` with the path to input data
```
CHANGE ME - INPUT DATA PATH
```

### Expected Output
- CHANGE ME
 
**Notes**
The code is fully reproducible with the provided data points.
