import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    """
    An implementation of a random forest ensemble learning algorithm built from multiple DecisionTrees.
    """
    def __init__(self, x, y, n_trees=100, max_depth=None, min_gain=1e-7, n_subset_features=None):
        """
        Constructor for RandomForest

        Parameters
        ----------
        x : np.ndarray
            2D array of features

        y : np.ndarray
            Array of labels

        n_trees : int
            Number of trees to use

        max_depth : int
            The maximum depth of any one decision tree

        min_gain : float
            The minimum entropy reduction required to make a Decision Node.

        n_subset_features : int | None
            The number of features to use in each decision tree.
            
        """
        self.__trees = [] # A list of tuples, each containing a decision tree, and its relevant feature indices.
        self.__labels = np.unique(y) # An array of sorted labels generate

        n_samples, n_features = x.shape
        # Each subset used to build a decision tree will use some (likely not all) of the input features.
        # If no value for this is provided, a commonly accepted default is the square root of the total number of features.
        if n_subset_features is None:
            n_subset_features = int(np.sqrt(n_features))

        for _ in range(n_trees):
            print(f'Generating tree {_}')
            self.__trees.append(
                RandomForest.__generate_tree(
                    x, 
                    y, 
                    max_depth, 
                    min_gain, 
                    n_subset_features
                )
            )

    # =============== Public Interface =============== 
    def predict_prob(self, x):
        """ 
        Predict a probability distribution for each sample in the provided features.

        Parameters
        ----------
        x : np.ndarray
            2D array of features
        
        Returns
        -------
        np.ndarray
            2D array of probability distributions
        """
        n_samples, n_features = x.shape
        # Gather the raw predictions from each tree - a 2D array of shape (n_trees, n_samples)
        raw_preds = self.__gather_raw_preds(x)
        # Compute the probability distribution
        prob = self.__compute_prob(raw_preds)
        return prob

    def predict(self, x):
        """ 
        Convenience function that uses majority voting to provide direct labels rather than a probability distribution. 
        
        Parameters 
        ----------
        x : np.ndarray
            2D array of features

        Returns
        -------
        np.ndarray
            Array of predicted labels
        
        """
        prob = self.predict_prob(x)
        return self.__labels[np.argmax(prob, axis=1)]


    # =============== Private Methods =============== 
    @staticmethod
    def __generate_tree(x, y, max_depth, min_gain, n_subset_features):
        n_samples, n_features = x.shape
        # Typically in Bootstrap Aggregation, samples are generated with replacement.
        # Additionally, we generate the same number of samples as in the dataset, empirically found to optimise Bias-Variance tradeoff.
        # On average, this leads to (1 - (1 - 1/N)^N) of the dataset being included (which tends to 1 - 1/e or ~63.2%)
        indices = np.random.choice(n_samples, n_samples, replace=True)

        # Extract samples based on indices
        x_sample, y_sample = x[indices], y[indices]

        # Generate the feature indices to be used in the tree
        feature_indices = np.random.choice(n_features, n_subset_features, replace=False)

        # Construct decision tree
        tree = DecisionTree(
            x_sample[:, feature_indices], 
            y_sample, 
            max_depth, 
            min_gain
        )

        return tree, feature_indices

    def __gather_raw_preds(self, x):
        return np.array([
            tree.predict(x[:, feature_indices])
            for tree, feature_indices in self.__trees
        ])

    def __compute_prob(self, raw_preds):
        n_samples = raw_preds.shape[1]

        prob = np.zeros((n_samples, len(self.__labels)))
        for i, label in enumerate(self.__labels):
            # Summing over the boolean values in numpy's masking, collapsing the rows (per-tree counts), to produce a count
            prob[:, i] = np.sum(raw_preds == label, axis=0)

        # Divide elementwise by the number of trees to normalise.
        return prob / len(self.__trees)

    # Getters
    def get_trees_and_features(self):
        return self.__trees
    
    def get_labels(self):
        return self.__labels