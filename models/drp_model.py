# models/drp_model.py
# This file is the heart of our project. It holds our DRP models and the corresponding implementation
# We chose to use two feed-forward neural encoders that then feed into a main model as our implementation
# We have a Cell Encoder, a Drug Encoder, and the main DRP model that concatenates both encoders into one final prediction model

import torch
import torch.nn as nn

# This class handles the cell encoding -> it is a feed-forward neural encoder that compresses the cell line feature vector into a low-dimensional embedding
class CellEncoder(nn.Module):
    # We take in the input features, hidden dimension, output dimension, and dropout (for regularization purposes) as our args
    def __init__(self, in_features=7107, hidden_dims=[1024, 512], out_dim=256, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = in_features
        
        # This builds the MLP which is repeated for each hidden layer
        # It is a fairly simple and common implementation that follows the format: Linear layer -> BN -> ReLU -> Dropout
        # Documentation: 
        #   https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        #   https://www.geeksforgeeks.org/deep-learning/what-is-batch-normalization-in-deep-learning/
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # This is the final projection to the embedding space, which just follows a simple linear layer followed by ReLU
        layers.append(nn.Linear(prev_dim, out_dim))
        layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
    
    # Self explanatory ML function, we simply take in a tensor x of shape (batch_size, 7107) containing all the cell features and return the cell embedding (batch_size, 256)
    def forward(self, x):
        return self.encoder(x)

# This class handles the drug encoding -> It compresses the Morgan fingerprint that we represent our drugs as and turns it into a 256 dimensional drug embedding
class DrugEncoder(nn.Module):
    # The arguments are essentially the same as for the cell encoder, except for the hidden_dims
    def __init__(self, in_features=2048, hidden_dims=[512], out_dim=256, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = in_features
        
        # Similar to the cell encoding, we follow the same process for the MLP 
        # See documentation provided in CellEncoder for reasoning behind this implementation
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Once again similar to the cell encoder, we finish with a linear layer and ReLU to project a 256-D embedding
        layers.append(nn.Linear(prev_dim, out_dim))
        layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)

    # Another self explanatory forward function that takes in the tensor x and returns the drug embedding
    def forward(self, x):
        return self.encoder(x)

# This is the main DRP model that encodes the cell line embedding and the drug embedding and concatenates the two before predicting log(IC50).
class DRPModel(nn.Module):
    def __init__(self, cell_in=7107, drug_in=2048, embed_dim=256, dropout=0.3):
        super().__init__()
        
        # First we encode the cell line information using an MLP -> this calls our function that handles the encoding process (see documentation for that function)
        self.cell_encoder = CellEncoder(
            in_features=cell_in,
            hidden_dims=[1024, 512],
            out_dim=embed_dim,
            dropout=dropout
        )
        
        # Next, we encode the drug information using an MLP -> this calls our function that handles the encoding process (see documentation for that function)
        self.drug_encoder = DrugEncoder(
            in_features=drug_in,
            hidden_dims=[512],
            out_dim=embed_dim,
            dropout=dropout
        )
        
        # This is the prediction head that takes the concatenated cell and drug embeddings and predicts a singular response
        self.predictor = nn.Sequential(
            # Once again we follow the same process that we use for the other MLP's (see CellEncoder documentation as to why)
            nn.Linear(embed_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Notably, we alter the dropout here because we don't want as large of a dropout in the deeper parts of our network -> this is good practice
            nn.Dropout(dropout * 0.7), 
            
            # This final linear layer ensures a singular output which is our log(IC50) predicted value
            nn.Linear(128, 1)  
        )
    
    # Another pretty self explanatory forward function with some more moving pieces this time
    def forward(self, cell_features, drug_features):
        # Naturally, we first encode both the cell features and the drug features
        cell_emb = self.cell_encoder(cell_features)  
        drug_emb = self.drug_encoder(drug_features)  
        
        # Here is where we concatenate both of them which is necessary in order to feed into our main model and predict a singular value
        fused = torch.cat([cell_emb, drug_emb], dim=1)  
        
        # This calls the model predictor that obtains a singular value based on the cell and drug information
        pred = self.predictor(fused) 
        
        # We return the value but squeezed because we need to change (batch_size, 1) format to (batch_size, )
        return pred.squeeze(-1)  
    
    # Small helper function for debugging purposes -> gets the embeddings to understand model behavior and ensure the model is working properly
    def get_embeddings(self, cell_features, drug_features):
        with torch.no_grad():
            cell_emb = self.cell_encoder(cell_features)
            drug_emb = self.drug_encoder(drug_features)
        return cell_emb, drug_emb

# Another helper function for debugging purposes -> we want to know how many parameters we have in our model to understand model  behavior slightly better 
# We simply use the best practices approach for obtaining model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Briefly test our model for debugging purposes and ensure everything is working properly before a full training of the model
    print("Testing DRP Model...")
    
    # See how many parameters our model has
    model = DRPModel()
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Test a minor forward pass just to make sure everything is working smoothly
    batch_size = 32
    cell_feat = torch.randn(batch_size, 7107)
    drug_feat = torch.randn(batch_size, 2048)
    
    # Output certain metrics for debugging purposes to ensure the trial forward pass works smoothly
    # If this is a success, we can move on to actually training the model
    pred = model(cell_feat, drug_feat)
    print(f"\nInput shapes:")
    print(f"  Cell features: {cell_feat.shape}")
    print(f"  Drug features: {drug_feat.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Output range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    
    cell_emb, drug_emb = model.get_embeddings(cell_feat, drug_feat)
    print(f"\nEmbedding shapes:")
    print(f"  Cell embeddings: {cell_emb.shape}")
    print(f"  Drug embeddings: {drug_emb.shape}")
