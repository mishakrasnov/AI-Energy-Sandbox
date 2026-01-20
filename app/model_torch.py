import torch
class Model:
    def __init__(self, checkpoint_path: str):
        """
        Initialization of the model.
        """
        self.checkpoint_path = checkpoint_path

    def load_checkpoint(self):
        """
        Loads weights from checkpoint. 
        """
        self.model = torch.jit.load(self.checkpoint_path)
        
    def predict(self, df, target_index: int):
        """
        Makes prediction based on the dataframe df with input features. Returns predictions for speicific target_index
        of shape (len(df),).
        """
        data = torch.tensor(df.values, dtype=torch.float32)
        preds = self.model(data).reshape(len(df), -1).detach().numpy()
        return preds[:, target_index]