import torch.nn as nn
from TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from TreeConvolution.tcnn import TreeActivation, DynamicPooling
from TreeConvolution.util import prepare_trees
import storage

def left_child(x):
    if len(x) != 3:
        return None
    return x[1]

def right_child(x):
    if len(x) != 3:
        return None
    return x[2]

def features(x):
    return x[0]

class BaoNet(nn.Module):
    def __init__(self, in_channels):
        super(BaoNet, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = False
        self.__is_being_trained = False

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def in_channels(self):
        return self.__in_channels
        
    def forward(self, x, return_trees=False, layer=10):
        trees = prepare_trees(x, features, left_child, right_child, cuda=self.__cuda)


        if return_trees:
            #assert(layer in [0, 1, 4, 7, 10, 12]) (for debugging output from different layers)
            # Get the output of the first tree convolution layer
            entry = self.tree_conv[:layer](trees)
            if isinstance(entry, tuple):
                grad = entry[0]
                grad.requires_grad_(True) # for computing gradients
            else:
                grad = entry
            output = self.tree_conv[layer:](entry)
            return output, grad
        

        # Get the output from the 10th layer:
        penultimate = self.tree_conv[:10](trees)
        if self.__is_being_trained == False:
            # We must not store predictions when the model is being trained.
            storage.record_penultimate_representation(penultimate)

        # Output:
        output = self.tree_conv[10:](penultimate)
            
        return output # self.tree_conv(trees)

    def cuda(self):
        self.__cuda = True
        return super().cuda()

    def set_being_trained(self):
        self.__is_being_trained = True
    
    def set_ready(self):
        self.__is_being_trained = False