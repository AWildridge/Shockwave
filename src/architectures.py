import torch
import torch.nn as nn

#TODO: Implement model architectures
class MLPBlock(nn.Module):
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

class Regurgitator(nn.Module):
    '''

    '''
    def __init__(self, d_input=2, d_model=64, n_heads=4, num_encoder_layers=4, num_decoder_layers=4, dropout=0.0, dim_feedforward=96, device='cpu', **kwargs):
        '''

        '''
        super().__init__()

        # Embed inputs using a single linear layer
        self.embedding = nn.Linear(d_input, d_model)

        # Seemed easier to write the args first than have a huge line passing them all
        transformer_args = {
            'd_model': d_model,
            'nhead': n_heads,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'activation': "gelu", # trying this one out
            'batch_first': True,
            'device': device,
        }
        self.transformer = nn.Transformer(**transformer_args)

        self.final_transform = nn.Linear(d_model, d_input)
    
    def forward(self, src, src_mask):
        return src
        # Work in progress!

