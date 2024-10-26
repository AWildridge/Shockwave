import torch
import torch.nn as nn

#TODO: Implement model architectures
class MaskPredMLP(nn.Module):
    '''
    An MLP to predict masked portions of its input.
    '''
    def __init__(self, d_input, hidden_layers, d_output, dropout=0.0, **kwargs):
        '''

        '''
        super().__init__()
        
        layers = [nn.Linear(d_input, hidden_layers[0]), nn.GELU(), nn.Dropout(dropout)]
        for idx in range(len(hidden_layers)-1):
            layers.extend([nn.Linear(hidden_layers[idx], hidden_layers[idx+1]), nn.GELU(), nn.Dropout(dropout)])

        layers.append(nn.Linear(hidden_layers[-1], d_output))

        self.network = nn.Sequential(*layers)
    
    def forward(self, src):
        return self.network(src)