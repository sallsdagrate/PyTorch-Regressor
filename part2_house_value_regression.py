from utility import *
from hyperparameters import RegressorHyperParameterSearch, RandomHyperparameterSearch
from Regressor import *

def example_main():
    load_and_test_regressor()


if __name__ == "__main__":
    # example_main()
    RegressorHyperParameterSearch(random_searches=10, folds=8)
