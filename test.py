"""
This demo shows how to visualize the designed features. Currently, only 2D feature space visualization is supported.
I use the same data for A2 as my input.
Each .xyz file is initialized as one urban object, from where a feature vector is computed.
6 features are defined to describe an urban object.
Required libraries: numpy, scipy, scikit learn, matplotlib, tqdm
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.spatial import ConvexHull
from tqdm import tqdm
from os.path import exists, join
from os import listdir
from sklearn.preprocessing import StandardScaler


class urban_object:
    """
    Define an urban object
    """
    def __init__(self, filenm):
        """
        Initialize the object
        """
        # obtain the cloud name
        self.cloud_name = filenm.split('/\\')[-1][-7:-4]

        # obtain the cloud ID
        self.cloud_ID = int(self.cloud_name)

        # obtain the label
        self.label = math.floor(1.0*self.cloud_ID/100)

        # obtain the points
        self.points = read_xyz(filenm)

        # initialize the feature vector
        self.feature = []

        # initialize the list for feature names
        self.feature_names = []

    def compute_features(self):
        """
        Compute the features, here we provide two example features. You're encouraged to design your own features
        """
        # [feature 1] height
        height = np.amax(self.points[:, 2])
        self.feature.append(height)
        self.feature_names.append('height')

        # get the root point and top point
        root = self.points[[np.argmin(self.points[:, 2])]]
        top = self.points[[np.argmax(self.points[:, 2])]]

        # construct the 2D and 3D kd tree
        kd_tree_2d = KDTree(self.points[:, :2], leaf_size=5)
        kd_tree_3d = KDTree(self.points, leaf_size=5)

        # [feature 2] root_density
        radius_root = 0.2
        count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
        root_density = 1.0*count[0] / len(self.points)
        self.feature.append(root_density)
        self.feature_names.append('root_density')

        # [feature 3] area
        # compute the 2D footprint and calculate its area
        hull_2d = ConvexHull(self.points[:, :2])
        hull_area = hull_2d.volume
        self.feature.append(hull_area)
        self.feature_names.append('hull_area')

        # [feature 4] (hull) shape_index  # compactness
        hull_perimeter = hull_2d.area
        shape_index = 1.0 * hull_area / hull_perimeter
        self.feature.append(shape_index)
        self.feature_names.append('shape_index')

        # obtain the point cluster near the top area
        k_top = max(int(len(self.points) * 0.005), 100)
        idx = kd_tree_3d.query(top, k=k_top, return_distance=False)
        idx = np.squeeze(idx, axis=0)
        neighbours = self.points[idx, :]

        # obtain the covariance matrix of the top points
        cov = np.cov(neighbours.T)
        w, _ = np.linalg.eig(cov)
        w.sort()  # w2: principal component, w1: second component, w3: third component

        # [feature 5] linearity
        # [feature 6] sphericity
        linearity = (w[2]-w[1]) / (w[2] + 1e-5)  # 0 ~ 1
        sphericity = w[0] / (w[2] + 1e-5)        # 0 ~ 1
        self.feature += [linearity, sphericity]
        self.feature_names.append('linearity')
        self.feature_names.append('sphericity')

        # [+ feature 7] planarity
        cov_global = np.cov(self.points.T)
        w_global, _ = np.linalg.eig(cov_global)
        w_global.sort()
        planarity = (w_global[2] - w_global[0]) / (w_global[2] + 1e-5)
        self.feature.append(planarity)
        self.feature_names.append('planarity')

        # [+ feature 8] 3d bounding box volume
        dx = np.amax(self.points[:, 0]) - np.amin(self.points[:, 0])
        dy = np.amax(self.points[:, 1]) - np.amin(self.points[:, 1])
        dz = np.amax(self.points[:, 2]) - np.amin(self.points[:, 2])
        bbox_vol = dx * dy * dz
        self.feature.append(bbox_vol)
        self.feature_names.append('bbox_vol')

        # [+ feature 9] 3d bounding box density
        bbox_density = len(self.points) / (bbox_vol + 1e-5)
        self.feature.append(bbox_density)
        self.feature_names.append('bbox_density')

        # [+ feature 10] upper half ratio
        top_z = top[0, 2]
        median_z = (top_z) / 2.0  # note: consider the ground elevation is 0m
        above_median_z = np.sum(self.points[:, 2] >= median_z)
        total_points = len(self.points)
        upper_half_ratio = above_median_z / total_points
        self.feature.append(upper_half_ratio)
        self.feature_names.append('upper_half_ratio')

        # [+ feature 11] minimum z (root)
        root_z = root[0, 2]
        self.feature.append(root_z)
        self.feature_names.append('root_z')

        # [+ feature 12] z variance
        z_var = np.var(self.points[:, 2])
        self.feature.append(z_var)
        self.feature_names.append('z_var')

        # [+ feature 13] total number of points
        self.feature.append(total_points)
        self.feature_names.append('total_points')


def read_xyz(filenm):
    """
    Reading points
        filenm: the file name
    """
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points


def feature_preparation(data_path):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    data_file = 'data.txt'

    # obtain the files in the folder
    files = sorted(listdir(data_path))

    # initialize the data
    input_data = []

    # initialize the list for feature names
    ft_names = []

    # retrieve each data object and obtain the feature vector
    for file_i in tqdm(files, total=len(files)):
        if file_i.startswith('.'):
            continue
        # obtain the file name
        file_name = join(data_path, file_i)

        # read data
        i_object = urban_object(filenm=file_name)

        # calculate features
        i_object.compute_features()

        # add the data to the list
        i_data = [i_object.cloud_ID, i_object.label] + i_object.feature
        input_data += [i_data]

        # obtain the feature name list
        if ft_names == i_object.feature_names:
            pass
        else:
            ft_names = i_object.feature_names

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    # write the output to a local file
    features = ",".join(ft_names)
    data_header = f'ID,label,{features}'
    np.savetxt(data_file, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)

    return outputs, ft_names

# def data_loading(data_file='data.txt'):
#     """
#     Read the data with features from the data file
#         data_file: the local file to read data with features and labels
#     """
#     # load data
#     data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')
#
#     # extract object ID, feature X and label Y
#     ID = data[:, 0].astype(np.int32)
#     y = data[:, 1].astype(np.int32)
#     X = data[:, 2:].astype(np.float32)
#
#     return ID, X, y

def feature_selection(outputs, ft_names):
    ft_idx_name = {}  # {feature_index : feature_name}
    for name in ft_names:
        ft_idx_name[ft_names.index(name)] = name

    # obtain the unique label (class) list (e.g., [0, 1, 2, 3, 4]
    class_col = outputs[:, 1]
    unique_class = np.unique(class_col)

    # make the array that contains label (class) column and feature columns
    ft_matrix = outputs[:, 1:]  # # [label, ft1, ft2, ...]

    # obtain the total number of the features
    total_feature_count = len(ft_names)

    # initialize the list for the selected 4 features
    selected_ft_idx = []

    for i in range(1, 5):
        # initialize the temporary J value and candidate feature set
        best_j = -np.inf
        best_ft_set = None

        for ft_idx in range(0, total_feature_count):
            test_ft_set = selected_ft_idx.copy()
            if ft_idx in test_ft_set:
                continue

            test_ft_set.append(ft_idx)

            col_num = [int(x + 1) for x in test_ft_set]
            col_num.insert(0, 0)  # add the label column

            test_ft_matrix = ft_matrix[:, col_num]  # [label, candidate_ft1, candidate_ft2, ...]

            n = ft_matrix.shape[0]                                 # total number of data samples
            test_ft_mean = np.mean(test_ft_matrix[:, 1:], axis=0)  # mean of all data samples (by feature)

            # initialize the metrics
            sw = np.zeros((i, i))  # within-class scatter matrix
            sb = np.zeros((i, i))  # between-class scatter matrix

            for k in unique_class:  # k == class index
                k_test_ft_matrix = test_ft_matrix[test_ft_matrix[:, 0].astype(int) == int(k)][:, 1:]  # [candidate_ft1, candidate_ft2, ...]

                nk = k_test_ft_matrix.shape[0]                      # the number of data samples of class k
                k_test_ft_mean = np.mean(k_test_ft_matrix, axis=0)  # mean of class k samples (by feature)

                diff = k_test_ft_matrix - k_test_ft_mean
                cov_k = (diff.T @ diff) / nk                        # covariance matrix of the data samples of class k using a given feature set

                mean_diff = (k_test_ft_mean - test_ft_mean).reshape(-1, 1)

                sw += nk * cov_k / n
                sb += nk * (mean_diff @ mean_diff.T) / n

            # calculate J value
            j = np.trace(sb) / (np.trace(sw) + 1e-5)

            # update the best J value and best feature set if the J value is larger
            if j > best_j:
                best_j = j
                best_ft_set = test_ft_set

        selected_ft_idx = best_ft_set.copy()
        selected_ft_name = []
        for idx in selected_ft_idx:
            selected_ft_name.append(ft_idx_name[idx])

        print(f'total number of selected feature: {i}')                            # temp
        print(f'> selected feature: {selected_ft_name}\t(J Value: {best_j:.4f})')  # temp

    filter_cols = selected_ft_idx.copy()
    filter_cols = [int(x + 2) for x in filter_cols]
    filter_cols.insert(0, 0)  # add the cloud_ID column
    filter_cols.insert(1, 1)  # add the label column

    filtered_outputs = outputs[:, filter_cols]

    return filtered_outputs

def SVM_classification(X, y, test_size=0.4, kernel='rbf', C=1.0, gamma='scale', degree=3, verbose=False):
    """
    Conduct SVM classification
        X: features
        y: labels
    """
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


    # Scale the features (important for SVM!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)
    acc = accuracy_score(y_test, y_preds)

    if verbose:
        print(f"SVM ({kernel} kernel) accuracy: {acc:.2f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_preds))

    return acc

def learning_curve(X, y, **best_params):
    test_percentages = []
    accuracies = []

    for i in range(5, 100, 5):
        # Pass i/100 (e.g., 0.05, 0.10...)
        acc = SVM_classification(X, y, test_size=i / 100.0, **best_params)

        test_percentages.append(i)
        accuracies.append(acc)

    plt.figure(figsize=(10, 6))
    plt.plot(test_percentages, accuracies, marker='o', linestyle='-')
    plt.title('SVM Accuracy vs. Test Set Size')
    plt.xlabel('Test Set Percentage (%)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def tune_svm_hyperparameters(X, y):
    """
    Tests Linear, RBF, and Poly kernels with different C, gamma, and degree
    to find the best parameters and compare the different kernels for SVM.
    """
    best_acc = 0
    best_config = {}

    # Test Linear Kernel (Only needs C)
    print("Tuning Linear Kernel...")
    for c in [0.1, 1, 10]:
        acc = SVM_classification(X, y, kernel='linear', C=c)
        if acc > best_acc:
            best_acc = acc
            best_config = {'kernel': 'linear', 'C': c}

    # Test RBF Kernel (Needs C and Gamma)
    print("Tuning RBF Kernel...")
    for c in [0.1, 1, 10]:
        for g in [0.1, 0.5, 'scale']:
            acc = SVM_classification(X, y, kernel='rbf', C=c, gamma=g)
            if acc > best_acc:
                best_acc = acc
                best_config = {'kernel': 'rbf', 'C': c, 'gamma': g}

    # Test Polynomial kernel (C, Gamma, and Degree)
    # The notes mention degree=3 in the example
    print("Tuning Polynomial Kernel...")
    for c in [0.1, 1, 10]:
        for d in [2, 3]:
            acc = SVM_classification(X, y, kernel='poly', C=c, degree=d)
            if acc > best_acc:
                best_acc = acc
                best_config = {'kernel': 'poly', 'C': c, 'degree': d}

    print(f"\nOptimization Complete! Best Config: {best_config} | Accuracy: {best_acc:.4f}")
    return best_config


if __name__=='__main__':
    # specify the data folder
    """"Here you need to specify your own path"""
    path = r"C:\Users\evang\Documents\Q3\GEO5017\ass02\GEO5017-A2-Classification\pointclouds-500\pointclouds-500"

    # conduct feature preparation
    print('Start preparing features')
    outputs, ft_names = feature_preparation(data_path=path)

    # conduct feature selection
    print('Start selecting 4 best features')
    filtered_outputs = feature_selection(outputs=outputs, ft_names=ft_names)
    # [could_ID, class_label, selected_ft1, selected_ft2, selected_ft3, selected_ft4]

    y = filtered_outputs[:, 1]
    X = filtered_outputs[:, 2:]

    # Run the tuning function
    best_params = tune_svm_hyperparameters(X, y)

    # Final evaluation of the Recommended Model
    print('\n--- Final Recommended SVM Model ---')
    SVM_classification(X, y, **best_params, verbose=True)

    learning_curve(X, y, **best_params)



    # # Hyperparameter analysis
    # print('Hyperparameter Analysis')
    # # Test Kernels for SVM
    # for k in ['linear', 'rbf', 'poly']:
    #     acc = SVM_classification(X, y, kernel=k)
    #     print(f"Kernel: {k} | Accuracy: {acc:.4f}")
    #
    #     # 2. Since RBF is likely best, tune C and Gamma specifically for it
    #     print("\n--- Tuning RBF Hyperparameters (C and Gamma) ---")
    #     best_rbf_acc = 0
    #     best_params = {}
    #
    #     for c_val in [0.1, 1.0, 10]:
    #         for g_val in [0.1, 0.5, 1.0, 'scale']:
    #             acc = SVM_classification(X, y, kernel='rbf', C=c_val, gamma=g_val)
    #             print(f"RBF Kernel | C: {c_val} | Gamma: {g_val} | Accuracy: {acc:.4f}")
    #
    #             if acc > best_rbf_acc:
    #                 best_rbf_acc = acc
    #                 best_params = {'C': c_val, 'gamma': g_val}
    #
    #     print(f"\nRecommended RBF Params: {best_params} with Accuracy: {best_rbf_acc:.4f}")
    #
    #
    #
    # print('Final Recommended SVM Model')
    # # Set verbose=True here so it prints the one matrix you need for your report
    # SVM_classification(X, y, kernel='rbf', C=1.0, verbose=True)
    #
    # print("\nGenerating Learning Curves...")
    # learning_curve(X, y)

    # # 1. Hyperparameter Analysis: Test Kernels (Requirement 2.2)
    # print('\n--- Hyperparameter Analysis: Kernel Comparison ---')
    # kernel_results = {}
    # for k in ['linear', 'rbf', 'poly']:
    #     # verbose=False keeps this quiet
    #     acc = SVM_classification(X, y, kernel=k, verbose=False)
    #     kernel_results[k] = acc
    #     print(f"Kernel: {k} | Accuracy: {acc:.4f}")
    #
    # # 2. Tuning the Recommended Kernel (RBF)
    # # Based on notes, RBF usually outperforms Poly [cite: 367]
    # print('\n--- Tuning RBF Hyperparameters (C and Gamma) ---')
    # best_rbf_acc = 0
    # best_params = {}
    #
    # for c_val in [0.1, 1.0, 10]:
    #     for g_val in [0.1, 0.5, 1.0, 'scale']:
    #         acc = SVM_classification(X, y, kernel='rbf', C=c_val, gamma=g_val, verbose=False)
    #         if acc > best_rbf_acc:
    #             best_rbf_acc = acc
    #             best_params = {'C': c_val, 'gamma': g_val}
    #
    # print(f"Best RBF Params Found: {best_params} with Accuracy: {best_rbf_acc:.4f}")
    #
    # # 3. Final Model Evaluation (Requirement 3: Error Analysis)
    # print('\n--- Final Recommended SVM Model for Report ---')
    # # This is the ONLY one that prints the Confusion Matrix
    # SVM_classification(X, y, kernel='rbf', **best_params, verbose=True)

    # # 1. Tuning the RBF (The Recommended Kernel) first
    # print('\n--- Tuning RBF Hyperparameters (C and Gamma) ---')
    # best_rbf_acc = 0
    # best_params = {}
    #
    # for c_val in [0.1, 1.0, 10]:
    #     for g_val in [0.1, 0.5, 1.0, 'scale']:
    #         acc = SVM_classification(X, y, kernel='rbf', C=c_val, gamma=g_val)
    #         if acc > best_rbf_acc:
    #             best_rbf_acc = acc
    #             best_params = {'C': c_val, 'gamma': g_val}
    #
    # # 2. Now do the "Final Kernel Comparison" using the best C found
    # print('\n--- Final Comparison: Optimized RBF vs. Others ---')
    # # Use the best C for all of them to make it fairer
    # best_c = best_params['C']
    # for k in ['linear', 'rbf', 'poly']:
    #     if k == 'rbf':
    #         acc = best_rbf_acc
    #     else:
    #         acc = SVM_classification(X, y, kernel=k)
    #     print(f"Kernel: {k:7}  | Accuracy: {acc:.4f}")
    #
    # print("\nGenerating Learning Curves ")




    """
    # load the data
    #print('Start loading data from the local file')
    #ID, X, y = data_loading()

    # visualize features
    #print('Visualize the features')
    #feature_visualization(X=X)

    # SVM classification
    #print('Start SVM classification')
    #SVM_classification(X, y)

    # RF classification
    #print('Start RF classification')
    #RF_classification(X, y)
    """