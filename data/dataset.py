# data/dataset.py
# This is an imperative file for the project. 
# It essentially acts as a bridge file between our data and our PyTorch model
# It converts the data into tensors so that we can train the model effectively in terms it can understand

# Documentation used to help with implementing this DataLoader
#   https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
#   https://www.geeksforgeeks.org/deep-learning/pytorch-dataloader/

# This is why we import torch
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

# Main class that handles the conversion of data -> tensors
class DRPDataset(Dataset):
    # This function's args are as follows
    #   response_list: List of dicts that we previously created of the cell ID, drug ID, and IC50 ground truth value
    #   cell_features: the dictionary we previously created mapping cell ID to the cell's feature vector
    #   drug_features: the dictionary we previously created mapping drug ID to the drug's fingerprint vector
    def __init__(self, response_list, cell_features, drug_features):
        # Initialize the vars
        self.responses = response_list
        self.cell_features = cell_features
        self.drug_features = drug_features
        
        # The following lines are a precautionary measure to filter out any pairs that don't have feature vectors
        # We already address this issue in preprocessing but to ensure the user can start the pipeline at any point, we address it again
        # The implementation, however, is the same
        self.valid_indices = []
        missing_cells = 0
        missing_drugs = 0
        
        for idx, resp in enumerate(response_list):
            if resp['cell_id'] not in cell_features:
                missing_cells += 1
                continue
            if resp['drug_id'] not in drug_features:
                missing_drugs += 1
                continue
            self.valid_indices.append(idx)
        
        # Output if any filtering was needed for debugging purposes
        if missing_cells > 0 or missing_drugs > 0:
            print(f"Warning: Filtered out {missing_cells} pairs with missing cells, "
                  f"{missing_drugs} pairs with missing drugs")
        
        # Output metrics of this new dataset for debugging purposes
        print(f"Dataset created: {len(self.valid_indices)} valid pairs")
    
    # Helper function that PyTorch REQUIRES (see documentation) for length
    def __len__(self):
        return len(self.valid_indices)
    
    # Another helper function that PyTorch REQUIRES (see documentation) to get one training sample in a tensor format
    def __getitem__(self, idx):
        # Obtain response data from a specified index
        resp = self.responses[self.valid_indices[idx]]
        
        # Get the feature vectors for that specific index
        cell_feat = self.cell_features[resp['cell_id']]  
        drug_feat = self.drug_features[resp['drug_id']] 
        log_ic50 = resp['log_ic50']
        
        # Convert to tensor format (this is critical)
        cell_feat = torch.FloatTensor(cell_feat)
        drug_feat = torch.FloatTensor(drug_feat)
        log_ic50 = torch.FloatTensor([log_ic50])
        
        # Return sample
        return {
            'cell_features': cell_feat,   
            'drug_features': drug_feat,   
            'log_ic50': log_ic50,         
            'cell_id': resp['cell_id'],
            'drug_id': resp['drug_id']
        }

# Loads the preprocessed dataset which are stored as .pkl files. Our training class will use these functions in order to properly function
def load_data(data_dir='data/processed'):
    print("Loading preprocessed data...")
    
    # Load the features from their .pkl files 
    with open(f'{data_dir}/cell_features.pkl', 'rb') as f:
        cell_features = pickle.load(f)
    
    with open(f'{data_dir}/drug_features.pkl', 'rb') as f:
        drug_features = pickle.load(f)
    
    with open(f'{data_dir}/response_data.pkl', 'rb') as f:
        responses = pickle.load(f)
    
    # Output metrics for debugging purposes
    print(f"Loaded: {len(cell_features)} cells, {len(drug_features)} drugs, "
          f"{len(responses)} response pairs")
    
    return cell_features, drug_features, responses

# Loads the cross validation splits that we stored as .pkl files. Our training class will use these functions in order to properly function
# Notably, we allow them to choose the split type as the training can be done on one specific type at a time (see train.py for further documentation)
def load_splits(split_type, data_dir='data/processed'):
    with open(f'{data_dir}/splits_{split_type}.pkl', 'rb') as f:
        splits = pickle.load(f)
    return splits


if __name__ == '__main__':
    # Quick test to ensure the dataset loads properly
    # For debugging purposes
    # Essentially, we load the data and create a dataset to see if it works properly and then output 1 sample to ensure accuracy
    cell_features, drug_features, responses = load_data()
    dataset = DRPDataset(responses, cell_features, drug_features)
    sample = dataset[0]
    print("\nSample data:")
    print(f"  Cell features shape: {sample['cell_features'].shape}")
    print(f"  Drug features shape: {sample['drug_features'].shape}")
    print(f"  Log IC50: {sample['log_ic50'].item():.4f}")
    print(f"  Cell ID: {sample['cell_id']}")
    print(f"  Drug ID: {sample['drug_id']}")