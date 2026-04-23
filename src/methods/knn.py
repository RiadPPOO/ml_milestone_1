import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification", distance="l2"):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.distance = distance
        self.w = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data 
        self.training_labels = training_labels

        pred_labels = self.predict(training_data)
        
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """

        N_test = test_data.shape[0]
        test_labels = np.zeros(N_test)

        for i in range(N_test): 
            if self.distance == "l2":
                distances = np.sqrt(np.sum(np.square(self.training_data - test_data[i]), axis=1))
            elif self.distance == "l1":
                distances = np.sum(np.abs(self.training_data - test_data[i]), axis=1)
            else:
                raise ValueError(f"Unknown distance: {self.distance}")
            
            sorted_indices = np.argsort(distances)
            k_nearest_indices = sorted_indices[:self.k]
            k_nearest_labels = self.training_labels[k_nearest_indices]

            if self.task_kind == "classification" :
                test_labels[i] = np.bincount(k_nearest_labels.astype(int)).argmax()
            elif self.task_kind == "regression" :
                test_labels[i] = np.mean(k_nearest_labels)
            else:
                raise ValueError(f"Unknown task_kind: {self.task_kind}")

        return test_labels
