import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Vectorizer:
    """
        Transform raw data into feature vectors. Support ordinal, numerical and categorical data.
        Also implements feature normalization and scaling.

        TODO: Support numerical, ordinal, categorical, histogram features.
    """
    def __init__(self, feature_config, num_bins=5):
        self.feature_config = feature_config
        self.feature_transforms = {}
        self.is_fit = False

    def get_numerical_vectorizer(self, values, verbose=False):
        """
        :return: function to map numerical x to a zero mean, unit std dev normalized score.
        """

        #mean, std = None, None
        #raise NotImplementedError("Numerical vectorizer not implemented yet")
        values = [int(num) for num in values]
        values = np.array(values)
        mean, std = np.mean(values), np.std(values)

        def vectorizer(x):
            """
            :param x: numerical value
            Return transformed score

            Hint: this fn knows mean and std from the outer scope
            """
            x = float(x)
            x = (x - mean.item())/std.item()

            return [x]

        return vectorizer

    def get_histogram_vectorizer(self, values):

        values = [int(num) for num in values]
        values = np.array(values)

        # Create 10 bins based on the range of the values
        bins = np.linspace(min(values), max(values), 10)

        def vectorizer(x):
            x = float(x)
            # Digitize the value to find its bin
            bin_idx = np.digitize(x, bins) - 1
            
            # One-hot encode the bin index
            one_hot = [0] * 10
            one_hot[bin_idx] = 1
            
            return one_hot

        return vectorizer
        #raise NotImplementedError("Histogram vectorizer not implemented yet")

    def get_categorical_vectorizer(self, values):
        """
        :return: function to map categorical x to one-hot feature vector
        """
        #raise NotImplementedError("Categorical vectorizer not implemented yet")
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  # handle_unknown='ignore' will return all zeros for unknown categories.

        #values_reshaped = [[item] for item in values]  # Reshape to fit encoder's expected input shape
        encoder.fit(np.array(values).reshape(-1, 1))

        def vectorizer(x):
            return encoder.transform([[x]])[0].tolist()
        
        return vectorizer

        # Create a DataFrame for the unique values
        # dummy_df = pd.get_dummies(values)
        
        # def vectorizer(x):
        #     if x in dummy_df.columns:
        #         return dummy_df.loc[x].tolist()
        #     # If the value doesn't exist in the original values list, return a vector of zeros
        #     return [0] * len(dummy_df.columns)

    def fit(self, X):
        """
            Leverage X to initialize all the feature vectorizers (e.g. compute means, std, etc)
            and store them in self.

            This implementation will depend on how you design your feature config.
        """
        for feature_name, feature_type in self.feature_config.items():
            values = [x[feature_name] for x in X]

            if feature_type == 'numerical':
                values = [float(x) for x in values]
                self.feature_transforms[feature_name] = self.get_numerical_vectorizer(values)
            elif feature_type == 'categorical':
                self.feature_transforms[feature_name] = self.get_categorical_vectorizer(values)
            elif feature_type == 'histogram':
                self.feature_transforms[feature_name] = self.get_histogram_vectorizer(values)
            else:
                raise Exception(f"Unsupported feature type: {feature_type}")

        self.is_fit = True
        # raise NotImplementedError("Not implemented yet")
        # self.feature_transforms = { "transform_name": None}


    def transform(self, X):
        """
        For each data point, apply the feature transforms and concatenate the results into a single feature vector.

        :param X: list of dicts, each dict is a datapoint
        """

        if not self.is_fit:
            raise Exception("Vectorizer not intialized! You must first call fit with a training set" )

        transformed_data = []
        for x_dict in X:
            feature_vector = []
            for feature_name, vectorizer in self.feature_transforms.items():
                feature_vector.extend(vectorizer(x_dict[feature_name]))
            transformed_data.append(feature_vector)

        return np.array(transformed_data)
        #raise NotImplementedError("Not implemented yet")