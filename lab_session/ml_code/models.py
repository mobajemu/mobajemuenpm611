from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class ModelFactory:
    """Factory class to get the model object."""
    @staticmethod
    def get_model(model_type):
        """Get the model object based on the model type."""
        if model_type == "logistic_regression":
            return LogisticRegression()
        elif model_type == "decision_tree":
            return DecisionTreeClassifier()
        elif model_type == "random_forest":
            return RandomForestClassifier()
        else:
            raise ValueError(f"Unknown model type: {model_type}")