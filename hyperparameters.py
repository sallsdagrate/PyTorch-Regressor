from Regressor import Regressor
from utility import save_regressor, evaluate_model
import sklearn.metrics as metrics
import random
import pandas as pd
import matplotlib.pyplot as plt
def nFoldCrossValidationParameterSearch(x, y, ps, folds, metric_f):
    # find best regressor in n fold cross val
    scores = []
    best_r_for_ps = None
    best_score_for_ps = float('inf')
    fold_size = x.shape[0] // folds
    for f in range(folds):
        # training data: x1 + x2, validation data: vx
        x1 = x.iloc[:f * fold_size]
        vx = x.iloc[f * fold_size:(f + 1) * fold_size]
        x2 = x.iloc[(f + 1) * fold_size:]

        # training data: y1 + y2, validation data: vy
        y1 = y.iloc[:f * fold_size]
        vy = y.iloc[f * fold_size:(f + 1) * fold_size]
        y2 = y.iloc[(f + 1) * fold_size:]

        # concatenate training data and pass into regressor
        x_train = x1._append(x2, ignore_index=True)
        y_train = y1._append(y2, ignore_index=True)

        r = Regressor(x_train, nb_epoch=300, params=ps)
        r.fit(x_train, y_train)
        # calculate score and hold onto the model if its the best so far
        current_r_score = r.score(vx, vy, metric_f=metric_f)
        scores.append(current_r_score)
        if current_r_score < best_score_for_ps:
            best_score_for_ps = current_r_score
            best_r_for_ps = r

    # return average score and best performing regressor for these parameters
    return sum(scores) / folds, best_r_for_ps


def RandomHyperparameterSearch(training,
                               folds, random_searches,
                               bounds=None,
                               metric_f=metrics.mean_squared_error):
    # Define bounds of hyperparameter search space
    if bounds is None:
        # default bounds
        bounds = {"learning": (-2, -0.5), "nodes": (3, 50), "mini-batch_size": (100, 300)}

    x_train_and_val, y_train_and_val = training

    llb, lub = bounds['learning']
    nlb, nub = bounds['nodes']
    mlb, mub = bounds['mini-batch_size']

    best_reg_overall, best_score_overall = None, float('inf')
    # points = [[], [], []]
    for n in range(random_searches):
        # generate random hyperparameters and perform crossValidation
        l = random.uniform(llb, lub)
        ps = {
            'learning': 10 ** l,
            'nodes': 20,  # random.randint(nlb, nub),
            'mini-batch_size': random.randint(mlb, mub)
        }
        score, r = nFoldCrossValidationParameterSearch(x_train_and_val,
                                                       y_train_and_val,
                                                       ps, folds,
                                                       metric_f=metric_f)

        # For visualisation
        # r = Regressor(x_train, nb_epoch=300, params=ps)
        # r.fit(x_train, y_train)
        # score = r.score(x_val, y_val, metric_f=metric_f)

        # points[0].append(l)
        # points[1].append(ps['nodes'])
        # points[2].append(score)
        if score < best_score_overall:
            best_score_overall = score
            best_reg_overall = r
            save_regressor(r)
            print(f'{n}: found new best score {score}, {ps}')
        else:
            print(f'{n}: {score}, {ps}')

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(points[0], points[1], points[2])
    # ax.set_xlabel('learning rate')
    # ax.set_ylabel('nodes')
    # ax.set_zlabel('score')
    # plt.show()

    return best_reg_overall


def RegressorHyperParameterSearch(train_valid_test_split=(9 / 10, 1 / 10),
                                  random_searches=200,
                                  folds=50):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """
    output_label = "median_house_value"

    data = pd.read_csv("housing.csv")
    data = data.sample(frac=1).reset_index(drop=True)
    n = data.shape[0]

    train, _ = train_valid_test_split
    train_ratio = (n * train) // 1

    x_train = data.loc[:train_ratio, data.columns != output_label]
    y_train = data.loc[:train_ratio, [output_label]]

    x_test = data.loc[train_ratio:, data.columns != output_label]
    y_test = data.loc[train_ratio:, [output_label]]

    best_r = RandomHyperparameterSearch((x_train, y_train),
                                        folds, random_searches)
    print(f'Model evaluation on test data -- parameters={best_r.params}')
    print(evaluate_model(best_r, x_test, y_test))
    return best_r