import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        
        self.fc1 = nn.Linear(state_size,(state_size+action_size)/2)
        self.fc2 = nn.Linear((state_size+action_size)/2,action_size)

        self.dropout = nn.Dropout(p=0.3)

        nn.init.uniform_(self.fc1.weights,a=0,b=1)
        nn.init.uniform_(self.fc2.weights,a=0,b=1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        output = self.fc2(x)

        return output
