import pickle
import sklearn.metrics as metrics
import math
import pandas as pd


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def evaluate_model(r, x, y):
    # Check model using all relevant metrics
    metric_map = {
        'MSE': metrics.mean_squared_error,
        'RMSE': lambda yt, yp: math.sqrt(metrics.mean_squared_error(yt, yp)),
        'ABS ERR': metrics.mean_absolute_error,
        '% ERR': metrics.mean_absolute_percentage_error,
        'R^2': metrics.r2_score
    }
    return {f: r.score(x, y, metric_map[f]) for f in metric_map}


def load_and_test_regressor(fp="housing.csv"):
    r = load_regressor()
    data = pd.read_csv(fp)
    n = data.shape[0]
    output_label = "median_house_value"
    x_test = data.loc[:, data.columns != output_label]
    y_test = data.loc[:, [output_label]]

    print(evaluate_model(r, x_test, y_test))
