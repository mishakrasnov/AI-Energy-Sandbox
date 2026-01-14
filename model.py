import numpy as np
import joblib
class Model:
    def __init__(self, checkpoint_path: str):
        """
        Initialization of the model.
        """
        self.checkpoint_path = checkpoint_path
        self.coefs = []
        self.bias = 0
        self.means = [318.136037, 318.137625, 318.190841, 318.210136, 318.244376]

    def load_checkpoint(self):
        """
        Loads weights from checkpoint. 
        Must be implemented. 
        """
        checkpoint = joblib.load(self.checkpoint_path)
        self.coefs = np.array(checkpoint[:-1])
        self.bias = checkpoint[-1]
        
    def predict(self, df, target_index: int):
        """
        Makes prediction based on the dataframe df with input features. Returns predictions for speicific target_index
        of shape (len(df),). Must be implemented. 
        """
        # Impute missing values with the mean for each feature using self.means
        for i, mean in enumerate(self.means):
            df.iloc[:, i] = df.iloc[:, i].fillna(mean)
        answer = df.iloc[:, :5] @ self.coefs + self.bias
        return answer + np.random.normal(0, 20, size=answer.shape)
