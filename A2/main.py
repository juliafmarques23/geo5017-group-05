import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.ensemble import RandomForestClassifier
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

        # [+ feature 8] Volume of the Axis-Aligned Bounding Box (AABB)
        dx = np.amax(self.points[:, 0]) - np.amin(self.points[:, 0])
        dy = np.amax(self.points[:, 1]) - np.amin(self.points[:, 1])
        dz = np.amax(self.points[:, 2]) - np.amin(self.points[:, 2])
        AABB_vol = dx * dy * dz
        self.feature.append(AABB_vol)
        self.feature_names.append('AABB_vol')

        # [+ feature 9] AABB density
        AABB_density = len(self.points) / (bbox_vol + 1e-5)
        self.feature.append(AABB_density)
        self.feature_names.append('AABB_density')

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

def feature_selection(outputs, ft_names):
    """This function selects the optimal features"""
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

        if i == 4:
            print(f'> selected feature: {selected_ft_name}')

    filter_cols = selected_ft_idx.copy()
    filter_cols = [int(x + 2) for x in filter_cols]
    filter_cols.insert(0, 0)  # add the cloud_ID column
    filter_cols.insert(1, 1)  # add the label column

    filtered_outputs = outputs[:, filter_cols]

    return filtered_outputs

def SVM_classification(X, y, test_size=0.4, kernel='rbf', C=1.0, gamma='scale', degree=3, verbose=False):
    """
    Conduct SVM classification
        Parameters:
        X: Feature matrix (e.g., planarity, root_density).
        y: Target labels (urban object classes).
        test_size: Proportion of the dataset to include in the test split (0.0 to 1.0).
        kernel: Kernel function ('linear', 'rbf', or 'poly') to map data to high-dimensional space.
        C: Regularization parameter, balances margin width vs. classification error.
        gamma: Kernel coefficient for RBF, defines the reach of a single training example's influence.
        degree: Degree of the polynomial kernel function (ignored by other kernels).
        verbose: If True, prints accuracy and the confusion matrix.
    """
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=42)
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)
    acc = accuracy_score(y_test, y_preds)

    if verbose:
        print(f"SVM ({kernel} kernel) accuracy: {acc:.2f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_preds))

    return acc

def RF_classification(X, y, test_size=0.4):
    """
    Conduct RF classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    rfc = RandomForestClassifier(n_estimators=200, min_samples_split=2, min_samples_leaf=1,
                                 bootstrap=True, max_samples=0.5, max_leaf_nodes=400,
                                 max_features=2, max_depth=20, criterion='gini', random_state=42)
    rfc.fit(X_train, y_train)
    y_preds = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_preds)
    cmatrix = confusion_matrix(y_test, y_preds)
    print("RF accuracy: %5.2f" % accuracy)
    print("Confusion Matrix:")
    print(cmatrix)


def learning_curve(X, y, svm_params, rf_params):
    """
    Generates three separate plots for the final report:
    1. SVM Train vs. Test (Generalization check)
    2. RF Train vs. Test (Generalization check)
    3. SVM Test vs. RF Test (Model Performance Comparison)
    """
    test_percentages = []
    svm_train_scores, svm_test_scores = [], []
    rf_train_scores, rf_test_scores = [], []

    for i in range(5, 100, 5):
        test_size = i / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # SVM Path (with scaling)
        scaler = StandardScaler()
        X_train_svm = scaler.fit_transform(X_train)
        X_test_svm = scaler.transform(X_test)
        clf_svm = svm.SVC(**svm_params)
        clf_svm.fit(X_train_svm, y_train)

        svm_train_scores.append(accuracy_score(y_train, clf_svm.predict(X_train_svm)))
        svm_test_scores.append(accuracy_score(y_test, clf_svm.predict(X_test_svm)))

        #  RF Path
        clf_rf = RandomForestClassifier(**rf_params, random_state=42)
        clf_rf.fit(X_train, y_train)

        rf_train_scores.append(accuracy_score(y_train, clf_rf.predict(X_train)))
        rf_test_scores.append(accuracy_score(y_test, clf_rf.predict(X_test)))

        test_percentages.append(i)

    # FIGURE 1: SVM Train vs. Test
    plt.figure(figsize=(10, 6))
    plt.plot(test_percentages, svm_train_scores, label='SVM Train', marker='o', linestyle='--', color='blue', alpha=0.5)
    plt.plot(test_percentages, svm_test_scores, label='SVM Test (Linear, C=10)', marker='o', linestyle='-',
             color='blue')
    plt.title('SVM Learning Curve: Train vs. Test')
    plt.xlabel('Test Set Percentage (%)')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()

    # FIGURE 2: RF Train vs. Test
    plt.figure(figsize=(10, 6))
    plt.plot(test_percentages, rf_train_scores, label='RF Train', marker='s', linestyle='--', color='green', alpha=0.5)
    plt.plot(test_percentages, rf_test_scores, label='RF Test', marker='s', linestyle='-', color='green')
    plt.title('RF Learning Curve: Train vs. Test')
    plt.xlabel('Test Set Percentage (%)')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()

    # FIGURE 3: SVM Test vs. RF Test
    plt.figure(figsize=(10, 6))
    plt.plot(test_percentages, svm_test_scores, label='SVM Test Accuracy', marker='o', color='blue')
    plt.plot(test_percentages, rf_test_scores, label='RF Test Accuracy', marker='s', color='green')
    plt.title('Final Comparison: SVM vs. RF Test Accuracy')
    plt.xlabel('Test Set Percentage (%)')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()

def tune_svm_hyperparameters(X, y):
    """
    Tests Linear, RBF, and Poly kernels with different C, gamma, and degree
    to find the optimal parameters for each kernel
    and then compare them to find the best one for SVM (for our features).
    """
    best_acc = 0
    best_config = {}

    # Test Linear Kernel (Only needs C)
    for c in [0.1, 1, 10, 100]:
        acc = SVM_classification(X, y, kernel='linear', C=c)
        if acc > best_acc:
            best_acc = acc
            best_config = {'kernel': 'linear', 'C': c}

    # Test RBF Kernel (Needs C and Gamma)
    for c in [0.1, 1, 10, 100]:
        for g in [0.01, 0.1, 0.5, 'scale']:
            acc = SVM_classification(X, y, kernel='rbf', C=c, gamma=g)
            if acc > best_acc:
                best_acc = acc
                best_config = {'kernel': 'rbf', 'C': c, 'gamma': g}

    # Test Polynomial kernel (C, Gamma, and Degree)
    # The notes mention degree=3 in the example
    for c in [0.1, 1, 10, 100]:
        for d in [2, 3]:
            acc = SVM_classification(X, y, kernel='poly', C=c, degree=d)
            if acc > best_acc:
                best_acc = acc
                best_config = {'kernel': 'poly', 'C': c, 'degree': d}

    print(f"> optimization complete! Best Config: {best_config} | Accuracy: {best_acc:.4f}")
    return best_config


if __name__=='__main__':
    # specify the data folder
    """"Here you need to specify your own path"""
    path = r"C:\Users\evang\Documents\Q3\GEO5017\ass02\GEO5017-A2-Classification\pointclouds-500\pointclouds-500"

    # conduct feature preparation
    print('>> Start preparing features')
    outputs, ft_names = feature_preparation(data_path=path)

    # conduct feature selection
    print('>> Start selecting 4 best features')
    filtered_outputs = feature_selection(outputs=outputs, ft_names=ft_names)
    # [could_ID, class_label, selected_ft1, selected_ft2, selected_ft3, selected_ft4]

    X, y = filtered_outputs[:, 2:], filtered_outputs[:, 1]
    # Run the tuning function for SVM
    print('>> Start tuning SVM')
    svm_params = tune_svm_hyperparameters(X, y)

    # Final evaluation of the Recommended Model
    print('>> Start SVM classification')
    SVM_classification(X, y, **svm_params, verbose=True)

    # For her RF
    rf_params = {
        'n_estimators': 200, 'max_features': 2, 'max_samples': 0.5,
        'criterion': 'gini', 'max_depth': 20
    }

    print('>> Start RF classification')
    RF_classification(X, y)

    learning_curve(X, y, svm_params, rf_params)
