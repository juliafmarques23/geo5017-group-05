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

    def compute_features(self):
        """
        Compute the features, here we provide two example features. You're encouraged to design your own features
        """
        # [feature 1] height
        height = np.amax(self.points[:, 2])
        self.feature.append(height)

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

        # [feature 3] area
        # compute the 2D footprint and calculate its area
        hull_2d = ConvexHull(self.points[:, :2])
        hull_area = hull_2d.volume
        self.feature.append(hull_area)

        # [feature 4] (hull) shape_index  # compactness
        hull_perimeter = hull_2d.area
        shape_index = 1.0 * hull_area / hull_perimeter
        self.feature.append(shape_index)

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

        # [+ feature 7] planarity
        cov_global = np.cov(self.points.T)
        w_global, _ = np.linalg.eig(cov_global)
        w_global.sort()
        planarity = (w_global[2] - w_global[0]) / (w_global[2] + 1e-5)
        self.feature.append(planarity)

        # [+ feature 8] 3d bounding box volume
        dx = np.amax(self.points[:, 0]) - np.amin(self.points[:, 0])
        dy = np.amax(self.points[:, 1]) - np.amin(self.points[:, 1])
        dz = np.amax(self.points[:, 2]) - np.amin(self.points[:, 2])
        bbox_vol = dx * dy * dz
        self.feature.append(bbox_vol)

        # [+ feature 9] 3d bounding box density
        bbox_density = len(self.points) / (bbox_vol + 1e-5)
        self.feature.append(bbox_density)

        # [+ feature 10] upper half ratio
        top_z = top[0, 2]
        median_z = (top_z) / 2.0  # note: consider the ground elevation is 0m
        above_median_z = np.sum(self.points[:, 2] >= median_z)
        total_points = len(self.points)
        upper_half_ratio = above_median_z / total_points
        self.feature.append(upper_half_ratio)

        # [+ feature 11] minimum z (root)
        root_z = root[0, 2]
        self.feature.append(root_z)

        # [+ feature 12] z variance
        z_var = np.var(self.points[:, 2])
        self.feature.append(z_var)

        # [+ feature 13] total number of points
        self.feature.append(total_points)


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
    # check if the current data file exist
    data_file = 'data.txt'
    #if exists(data_file):
    #    return

    # obtain the files in the folder
    files = sorted(listdir(data_path))

    # initialize the data
    input_data = []

    # retrieve each data object and obtain the feature vector
    for file_i in tqdm(files, total=len(files)):
        if file_i.startswith('.'): #or file_i.startswith('163') or file_i.startswith('060'):
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

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    # write the output to a local file
    data_header = 'ID,label,\
                   height,root_density,area,shape_index,linearity,sphericity,\
                   planarity,bbox_vol,bbox_density,upper_half_ratio,min_z,z_var,total_points'
    np.savetxt(data_file, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)


def data_loading(data_file='data.txt'):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)

    return ID, X, y


def feature_visualization(X):
    """
    Visualize the features
        X: input features. This assumes classes are stored in a sequential manner
    """
    # initialize a plot
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("feature subset visualization of 5 classes", fontsize="small")

    # define the labels and corresponding colors
    colors = ['firebrick', 'grey', 'darkorange', 'dodgerblue', 'olivedrab']
    labels = ['building', 'car', 'fence', 'pole', 'tree']

    # plot the data with first two features
    for i in range(5):
        ax.scatter(X[100*i:100*(i+1), 9], X[100*i:100*(i+1), 11], marker="o", c=colors[i], edgecolor="k", label=labels[i])

    # show the figure with labels
    """
    Replace the axis labels with your own feature names
    """
    #ax.set_xlabel('x1:height')  # 0
    #ax.set_ylabel('x2:root density')  # 1
    # ax.set_xlabel('x1:hull area')  # 2
    # ax.set_ylabel('x2:shape index')  # 3
    # ax.set_xlabel('x1:linearity')  # 4
    #ax.set_ylabel('x2:sphericity')  # 5
    #ax.set_xlabel('x1:planarity')  # 6
    #ax.set_xlabel('x1:bbox_vol')  # 6
    #ax.set_ylabel('x2:bbox_density')  # 7
    ax.set_ylabel('x2:z_var')  # 8
    #ax.set_ylabel('x2:root_z')  # 9
    ax.set_xlabel('x1:upper_half_ratio')  # 10
    # ax.set_ylabel('x2:shape index')  # 3
    ax.legend()
    plt.show()


def SVM_classification(X, y):
    """
    Conduct SVM classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)
    acc = accuracy_score(y_test, y_preds)
    print("SVM accuracy: %5.2f" % acc)
    print("confusion matrix")
    conf = confusion_matrix(y_test, y_preds)
    print(conf)


def RF_classification(X, y):
    """
    Conduct RF classification
        X: features
        y: labels
    """
    pass


if __name__=='__main__':
    # specify the data folder
    """"Here you need to specify your own path"""
    path = '/Users/moonchaeyeon/PycharmProjects/q3_ml/GEO5017-A2-Classification/pointclouds-500/pointclouds-500'

    # conduct feature preparation
    print('Start preparing features')
    feature_preparation(data_path=path)

    # load the data
    print('Start loading data from the local file')
    ID, X, y = data_loading()

    # visualize features
    print('Visualize the features')
    feature_visualization(X=X)

    # SVM classification
    print('Start SVM classification')
    SVM_classification(X, y)

    # RF classification
    print('Start RF classification')
    RF_classification(X, y)
