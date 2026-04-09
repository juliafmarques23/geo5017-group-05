# GEO5017 - Machine Learning (2026) Assignment 3

**Group 05**  
Chaeyeon Moon (6477453)  
Evangelia Palli (6435939)  
Julia Fossa Marques (6550975)  


## Project: Waste Detection in the Built Environment

#### Repository Structure
```
/A1/
/A2/
/A3/
  └── /code/
         └── classification_main.py      # main script for image clasification
         └── object_detection_main.py    # main script for object detection
         └── undersampling.py            # user can run this script to undersample a training dataset
  └── ReadMe.txt                         # Instructions and project info
```

### Setup
Besides Python's standard library, the following 5 libraries are required: **matplotlib**, **numpy**, **sklearn**, **scipy**, and **tqdm**.

**Command for Installation** 
```
python -m pip install ultralytics opencv-python torch
```

### How to Run

1. Navigate to the `/code/` folder
2. Run `classification_main.py` and `object_detection_main.py` after changing the configuration (e.g., data path)
 
**Notes**
The code is fully reproducible with the provided data points.
