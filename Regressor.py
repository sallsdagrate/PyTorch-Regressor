import torch
import torch.nn as nn
import pandas as pd
from DataLoader import DataLoader
import sklearn.metrics as metrics


class Net(nn.Module):

    def __init__(self, inputs, outputs=1, hidden_nodes=20):
        super(Net, self).__init__()
        # Adjust sizes of fully connected layers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputs, hidden_nodes),
            nn.Sigmoid(),
            nn.Linear(hidden_nodes, outputs)
        )

    def forward(self, x):
        return self.linear_relu_stack(x.float())


class Regressor:
    def __init__(self, x, nb_epoch=1000, params=None):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """
        if params is None:
            params = {"learning": 0.001, "nodes": 20, 'mini-batch_size': -1}
        self.params = params

        # plot_ls used for visualisation
        # self.plot_ls = []
        # Norm values store mean and std for each column so that we can "denormalise" the output
        self.norm_values = {}

        X, Y = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1

        # setting up network
        self.learning_rate = params['learning']
        self.nb_epoch = nb_epoch

        # initialise network
        self.net = Net(inputs=X.shape[1], hidden_nodes=params['nodes'])

        if torch.cuda.is_available():
            self.net.to('cuda')

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)
        return

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """
        # Return preprocessed x and y, return None for y if it was None
        x_scaled = x.copy()
        for col in x.columns:
            x_scaled = self.__normalise_col__(x_scaled, x, col, training)

        y_scaled = None
        if y is not None:
            y_scaled = y.copy()
            for col in y:
                self.y_col = col
                y_scaled = self.__normalise_col__(y_scaled, y, col, training)

        return torch.tensor(x_scaled.values), y if y is None else torch.tensor(y_scaled.values)

    def __normalise_col__(self, out, unscaled, col, training):
        # if float normalise on a normal distribution otherwise turn into one hot vector
        if unscaled[col].dtype == float:
            if training:
                scaled_col, col_mean_and_std = self.__z_sample_col__(col, unscaled)
                out[col] = scaled_col
                self.norm_values[col] = col_mean_and_std
            else:
                try:
                    out[col] = (unscaled[col].fillna(self.norm_values[col]['mean']) - self.norm_values[col]['mean']) / \
                               self.norm_values[col]['std']
                except KeyError as k:
                    print(f'Model is trying to normalise test data column which is has not seen before')
                    print(k)
        else:
            if training:
                out, categories = self.__one_hot_col__(unscaled[col], out)
                self.norm_values[col] = categories
            else:
                for cat in self.norm_values[col]['categories']:
                    # PyTorch network requires float32
                    out[cat] = (unscaled[col] == cat).astype('float32')
            out = out.drop(columns=[col])
        return out

    # Normal normalising
    def __z_sample_col__(self, col, data):
        vs = {}
        col_copy = data[col]
        m = col_copy.mean()
        data.loc[:, col] = col_copy.fillna(m)
        vs['mean'] = data[col].mean()
        vs['std'] = data[col].std()
        return ((data[col] - vs['mean']) / vs['std']).astype('float32'), vs

    # Alternative form of normalise we experimented with
    def __scale_by_max__(self, col):
        vs = {}
        col = col.fillna(col.mean())
        vs['max'] = col.max()
        return (col / vs['max']).astype('float32'), vs

    # Turning into one hot vector
    def __one_hot_col__(self, col, x_scaled):
        cats = pd.Categorical(col).categories
        for cat in cats:
            x_scaled[cat] = (col == cat).astype('float32')
        return x_scaled, {'categories': cats}

    def __train_loop__(self, dataloader):
        # for each loop get another batch from dataloader and perform gradient descent on it
        mini_x, mini_y = dataloader.get()
        loss = float('inf')
        # first = True
        while mini_x is not None and mini_y is not None:
            self.optimizer.zero_grad()
            x_tensor = mini_x.requires_grad_(True)
            y_true = mini_y.requires_grad_(True)

            y_pred = self.net(x_tensor)
            loss = self.criterion(y_pred, y_true)

            # This was used for visualisation
            # if first:
            #     self.plot_ls.append(loss.tolist())
            #     first = False

            loss.backward()
            self.optimizer.step()

            mini_x, mini_y = dataloader.get()
        # self.plot_ls.append(loss.tolist())
        return loss

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y=y, training=True)
        dl = DataLoader(X, Y, miniBatchSize=self.params['mini-batch_size'])
        for epoch in range(self.nb_epoch):
            loss = self.__train_loop__(dl)

        # Below code for visualisation
        # plt.plot(self.plot_ls, markersize=0.2)
        # plt.title(f'lr:{self.learning_rate}, hidden nodes:{self.params["nodes"]}, epochs:{self.nb_epoch}')
        # plt.show()
        # self.plot_ls = []
        return self
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """
        X, _ = self._preprocessor(x, training=False)  # Do not forget
        self.net.eval()
        with torch.no_grad():
            # evaluate and denormalising
            return (self.net(X) * self.norm_values[self.y_col]['std'] +
                    self.norm_values[self.y_col]['mean']).numpy()
        # * self.norm_values['median_house_value']['max']).tolist

    def score(self, x, y, metric_f=metrics.mean_squared_error):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        X, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
        self.net.eval()
        with torch.no_grad():
            net_output = self.net(X)
            test_preds = [n[0] for n in net_output.tolist()]
            test_actuals = [n[0] for n in Y.tolist()]
            m = metric_f(test_actuals, test_preds)
            return m
