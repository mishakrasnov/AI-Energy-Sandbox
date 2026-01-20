import xgboost as xgb
class Model:
    def __init__(self, checkpoint_path: str):
        """
        Initialization of the model.
        """
        self.checkpoint_path = checkpoint_path
        self.model = xgb.Booster()

    def load_checkpoint(self):
        """
        Loads weights from checkpoint. 
        """
        self.model.load_model(self.checkpoint_path)
        
    def predict(self, df, target_index: int):
        """
        Makes prediction based on the dataframe df with input features. Returns predictions for speicific target_index
        of shape (len(df),).
        """
        # remove the feature_ from the beginign of the columns names
        return self.model.predict(xgb.DMatrix(df)).reshape(len(df), -1)[:, target_index]