# utils/metrics.py
# This is a simple helper file that helps calculate the final metrics that we require for our model results
# This includes MSE, MAE, RMSE
# It is imported in train.py such that the metrics can be calculated after every epoch and at the end of the training sequence as well for better statistical analysis

# We use numpy and the very useful sklearn in order to use their more accurate metric calculations
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# The main function that actually calculates the metrics
# Takes in an array of predicted values as input and the ground truth array of these values
# Returns a dictionary that holds the 3 metrics that we calculate
def calculate_metrics(predictions, targets):
    # Ensure that both these arrays are in numpy form, we need it to use the sklearn metrics
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # Calculate the 3 values using sklearn and sqrt(mse) for the rmse
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    
    # Return the values in dictionary format
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse)
    }

# Print out the metrics in an easy to read format. This is especially useful to ensure clean output when debugging
def print_metrics(metrics, prefix=""):
    if prefix:
        prefix = prefix + " "
    print(f"{prefix}MSE: {metrics['mse']:.4f}, "
          f"MAE: {metrics['mae']:.4f}, "
          f"RMSE: {metrics['rmse']:.4f}")

# This is just to test out the metrics.py function to make sure everything is working smoothly
if __name__ == '__main__':
    import numpy as np
    
    # Create dummy data in the correct format
    targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = np.array([1.3, 2.5, 2.4, 4.8, 6.3])
    
    # Test the metric calculation and the easy to read print format on the dummy data to ensure everything is working perfectly and cleanly.
    # Good for debugging
    metrics = calculate_metrics(predictions, targets)
    print("Test metrics:")
    print_metrics(metrics)
