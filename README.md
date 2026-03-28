# GEO5017 - Machine Learning (2026) Assignment 2

**Group 05**  
Chaeyeon Moon (6477453)  
Evangelia Palli (6435939)  
Julia Fossa Marques (6550975)  


## Project: Objects Classification from AirBorne LiDAR Data

### Description
This project aims to classify a dataset of 500 pre-segmented urban point clouds into five classes: building, car, fence, pole, and tree. To achieve this, we developed a feature-based classification code using two supervised learning classifiers: Support Vector Machine (SVM) and Random Forest (RF).

The main function was specifically designed to answer the questions of this assignment, using the data provided for it; it is not necessary to provide any additional arguments to reproduce the results shown in the report.

#### Repository Structure
```
/A1/
/A2/
  └── /code/
         └── main.py       # Main script 
  └── ReadMe.txt    # Instructions and project info
```

### Setup
The program works with minimal setup; Besides Python's standard library, the following 5 libraries are required: **matplotlib**, **numpy**, **sklearn**, **scipy**, and **tqdm**.

**Command for Installation** 
```
python -m pip install matplotlib numpy scikit-learn scipy tqdm
```

### How to Run

1. Navigate to the `/code/` folder
2. Run `main.py` after changing the input data path

### Expected Output

The following results will be printed:
- Selected features
- Optimal hyperparameters
- Accuracy scores and confusion matrices for both classifiers

The following figures will be displayed:
- Plot 1. Learning curve for the SVM classifier
- Plot 2. Learning curve for the RF classifier
- Plot 3. Comparison of learning curves: SVM vs. RF
 
**Notes**
The code is fully reproducible with the provided data points.

