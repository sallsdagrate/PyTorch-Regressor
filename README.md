## Usage

# Part 1

To run the part 1 library, run the command

```
python3 part1_nn_lib.py

```

# Part 2

In order to perform hyperparameter tuning, we can use the function which performs a 9/10, 1/10 split for the cross-validation and test data,
uses 50 folds and generates 200 random searches on the hyperparameter search space. This function returns the best regressor as well as saving it
 in the pickle file.
This can be overriden to use 5 folds and 100 searches by passing in the parameters as follows:

```
RegressorHyperParameterSearch()
RegressorHyperParameterSearch((0.8, 0.1), 100, 5)
```

A new evaluate model was built which we can use to see how a given model performed over a given test set
 using a variety of different metrics such as MSE, RMSE, ABS error, % error and R^2. Can be called as follows:

```
evaluate_model(regressor, test_x, test_y)
```

An easier way to test a model is by calling load_and_test_regressor which uses housing.csv as a default file and makes use of the
 given load_regressor function to evaluate the model on given data. If the data was in a file called "test.csv" we would call it as follows:

```
load_and_test_regressor("test.csv")
```

Given a regressor we can easily predict the output as a tensor:

```
prediction = regressor.predict(x)
```

Given a set of unseen test data we have a metric score that calculates its performs on any given metric;
 the default metric is set to MSE, however, it is possible to pass in other metrics or even a custom metric as a function reference
 to the parameter metric_f.

```
output = regressor.score(x, actuals)
```
or
```
output = regressor.score(x, actuals, metric_f = custom_metric)
```