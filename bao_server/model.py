import json
import numpy as np
import torch
import torch.optim
import joblib
import os
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
import net
import time
from featurize import TreeFeaturizer

CUDA = torch.cuda.is_available()

def _nn_path(base):
    return os.path.join(base, "nn_weights")

def _x_transform_path(base):
    return os.path.join(base, "x_transform")

def _y_transform_path(base):
    return os.path.join(base, "y_transform")

def _channels_path(base):
    return os.path.join(base, "channels")

def _n_path(base):
    return os.path.join(base, "n")


def _inv_log1p(x):
    return np.exp(x) - 1

class BaoData:
    def __init__(self, data):
        assert data
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return (self.__data[idx]["tree"],
                self.__data[idx]["target"])

def collate(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.tensor(targets)
    return trees, targets

class BaoRegression:
    def __init__(self, verbose=False, have_cache_data=False):
        self.__net = None
        self.__verbose = verbose

        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p,
            validate=True)
        scale_transformer = preprocessing.MinMaxScaler()

        self.__pipeline = Pipeline([("log", log_transformer),
                                    ("scale", scale_transformer)])
        
        self.__tree_transform = TreeFeaturizer()
        self.__have_cache_data = have_cache_data
        self.__in_channels = None
        self.__n = 0
        
    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def num_items_trained_on(self):
        return self.__n
            
    def load(self, path):
        with open(_n_path(path), "rb") as f:
            self.__n = joblib.load(f)
        with open(_channels_path(path), "rb") as f:
            self.__in_channels = joblib.load(f)
            
        self.__net = net.BaoNet(self.__in_channels)
        # The map_location argument is required to use a CPU-based PyTorch version.
        self.__net.load_state_dict(torch.load(_nn_path(path), map_location=torch.device('cpu')))
        self.__net.eval()
        
        with open(_y_transform_path(path), "rb") as f:
            self.__pipeline = joblib.load(f)
        with open(_x_transform_path(path), "rb") as f:
            self.__tree_transform = joblib.load(f)

    def save(self, path):
        # try to create a directory here
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.__net.state_dict(), _nn_path(path))
        with open(_y_transform_path(path), "wb") as f:
            joblib.dump(self.__pipeline, f)
        with open(_x_transform_path(path), "wb") as f:
            joblib.dump(self.__tree_transform, f)
        with open(_channels_path(path), "wb") as f:
            joblib.dump(self.__in_channels, f)
        with open(_n_path(path), "wb") as f:
            joblib.dump(self.__n, f)

    def fit(self, X, y):
        if isinstance(y, list):
            y = np.array(y)

        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        self.__n = len(X)
            
        # transform the set of trees into feature vectors using a log
        # (assuming the tail behavior exists, TODO investigate
        #  the quantile transformer from scikit)
        y = self.__pipeline.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        
        self.__tree_transform.fit(X)
        X = self.__tree_transform.transform(X)

        pairs = list(zip(X, y))
        dataset = DataLoader(pairs,
                             batch_size=16,
                             shuffle=True,
                             collate_fn=collate)

        # determine the initial number of channels
        for inp, _tar in dataset:
            in_channels = inp[0][0].shape[0]
            break

        self.__log("Initial input channels:", in_channels)

        if self.__have_cache_data:
            assert in_channels == self.__tree_transform.num_operators() + 3
        else:
            assert in_channels == self.__tree_transform.num_operators() + 2

        self.__net = net.BaoNet(in_channels)
        # Raise flag to the NN that it is being trained so that it won't 
        # store the predictions.
        self.__net.set_being_trained()
        self.__in_channels = in_channels
        if CUDA:
            self.__net = self.__net.cuda()

        optimizer = torch.optim.Adam(self.__net.parameters())
        loss_fn = torch.nn.MSELoss()
        
        losses = []
        for epoch in range(100):
            loss_accum = 0
            for x, y in dataset:
                if CUDA:
                    y = y.cuda()
                y_pred = self.__net(x)
                loss = loss_fn(y_pred, y)
                loss_accum += loss.item()
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum /= len(dataset)
            losses.append(loss_accum)
            if epoch % 15 == 0:
                self.__log("Epoch", epoch, "training loss:", loss_accum)

            # stopping condition
            if len(losses) > 10 and losses[-1] < 0.1:
                last_two = np.min(losses[-2:])
                if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
                    self.__log("Stopped training from convergence condition at epoch", epoch)
                    # Raise flag to the NN to begin storing the predictions.
                    self.__net.set_ready()
                    break
        else:
            self.__log("Stopped training after max epochs")
            self.__net.set_ready()

    def predict(self, X):
        # self.__net.set_being_trained() # TODO: comment back for pgd

        if not isinstance(X, list):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]

        #print("Before Tree Transformation = {}\n".format(X))
        X = self.__tree_transform.transform(X)
        #print("AFTER Tree Transformation = {}\n".format(X))
        
        self.__net.eval()
        pred = self.__net(X).cpu().detach().numpy()
        #print(pred)
        # return self.__pipeline.inverse_transform(pred)[0][0] # ONLY USE FOR PGD
        return self.__pipeline.inverse_transform(pred)
    
    def featurize_vector(self, X):
        if not isinstance(X, list):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]

        return self.__tree_transform.transform(X)


    def predict_vector(self, X):
        # For testing poisoning at the lowest level: vectorized plans
        self.__net.set_being_trained() # TODO: delete later
        self.__net.eval()
        pred = self.__net(X).cpu().detach().numpy()
        #print(pred)
        return self.__pipeline.inverse_transform(pred)[0][0]


    
    def compute_gradients_vectorized(self, x, y, layer=10, return_loss=False):
        # Ensure y is a PyTorch tensor and has the same shape as y_pred
        y = torch.tensor(y, dtype=torch.float32)
        if len(y.shape) == 1:  # If y is a flat array, add a batch dimension
            y = y.unsqueeze(0)

        self.__net.zero_grad()
        self.__net.set_being_trained()

        loss_fn = torch.nn.MSELoss()
        self.__net.train()

        # Forward pass to compute predictions and gradients
        y_pred, trees = self.__net(x, return_trees=True, layer=layer)
        #print(y_pred)

        # Compute the loss (negating to maximize the original loss)
        # loss = loss_fn(y_pred, y)  TODO: maybe uncomment later
        loss = y_pred - y


        #print(f"loss == {loss.item()}")
        
        # Retain gradients for the input trees
        trees.retain_grad()
        loss.backward()

        # Detach gradients to avoid side effects
        gradients = trees.grad.detach()

        # Clear gradients from the network
        self.__net.zero_grad()

        if return_loss:
            return gradients, loss.item()
        return gradients


