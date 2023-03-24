
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reorder(nn.Module):
    def forward(self, input):
        return input.permute((0, 2, 1))
    

class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.conv1 = nn.Conv1d(
            in_channels=hid_size,
            out_channels=hid_size,
            kernel_size=2) 
        
        self.rel_pol =  nn.Sequential( nn.ReLU(),
                                    nn.AdaptiveAvgPool1d(output_size=1))
        
        self.lin_res =  nn.Linear(in_features=hid_size, out_features=1)
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        
        self.category_out =  nn.Sequential(nn.Linear(n_cat_features, hid_size),
                                           nn.ReLU(),
                                           nn.Linear(hid_size, 32),
                                           nn.ReLU(),
                                           nn.Linear(32, 1))

        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.conv1(title_beg)
        title = self.rel_pol(title)
        title = title.view(title.size(0), -1)
        title = self.lin_res(title)

        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.conv1(full_beg)
        full = self.rel_pol(full)
        full = full.view(full.size(0), -1)
        full = self.lin_res(full)      
        
        category = self.category_out(input3)        
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        out = self.inter_dense(concatenated)
        out = self.final_dense(out)
        
        return out