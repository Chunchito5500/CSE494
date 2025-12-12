# data/splits.py
# Used to generate the 4 data splits we need
#   Random, cell-blind, drug-blind, cell-drug-blind

# Documentation used to help with proper, best practices Kfold generation: 
#   https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
#   https://www.geeksforgeeks.org/machine-learning/cross-validation-using-k-fold-with-scikit-learn/

# Import necessary libraries such as sklearn's KFold as well as other data-based libraries such as numpy and pickle
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict
import pickle

# Main class that handles splitting the data
class DataSplitter:
    
    def __init__(self, responses, n_folds=5, random_state=42):
        # Initialize the necessary args that we need
        #   responses (the list of dicts that we obtained in the preprocessing)
        #   number of folds in our cross-validation (specifications tell us to use 5-fold)
        #   random state for reproducability purposes (this is good practice)
        self.responses = responses
        self.n_folds = n_folds
        self.random_state = random_state
    
    # Handles the random K-fold split that tests on random cell and drug pairs
    def random_split(self):
        print(f"\nCreating random split ({self.n_folds}-fold CV)...")
        
        # Initialize the Kfold splitter with the appropriate parameters
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Create the array of sample indices that the Kfold will use to partition as well as the container for all fold splits
        indices = np.arange(len(self.responses))
        splits = []
        
        # Iterate through the folds produced by Kfold
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            # For each current fold we save the indices
            splits.append({
                'train': train_idx.tolist(),
                'val': val_idx.tolist()
            })

            # Print the fold summary for debugging purposes
            print(f"  Fold {fold_idx+1}: {len(train_idx)} train, {len(val_idx)} val")
        
        return splits
    
    # This function handles the cell-blind Kfold split
    # Care is taken to ensure that no cell line appearing in the validation set appears in the training set.
    # In essence, the model is evaluated on entirely unseen cell lines
    def cell_blind_split(self):
        print(f"\nCreating cell-blind split ({self.n_folds}-fold CV)...")
        
        # The first step is to group all the sample indices by cell line, because we will need to be able to distinguish the cell lines
        cell_to_indices = defaultdict(list)
        for idx, resp in enumerate(self.responses):
            cell_to_indices[resp['cell_id']].append(idx)
        
        # This is where we get the list of all unique cell lines that exist in the dataset, we will use this in our fold logic to implement cell-blind
        cells = list(cell_to_indices.keys())
        print(f"  Total unique cells: {len(cells)}")
        
        # Kfold splitter with appropriate parameters
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        splits = []

        # This loop is slightly different from the random split, as we split based on cells and not indices
        for fold_idx, (train_cells_idx, val_cells_idx) in enumerate(kf.split(cells)):
            # This converts the indices to actual cell IDs so that we actually know which cell lines belong to each split
            # This makes it so that we will be able to append the splits accordingly based on unique cells later on.
            train_cells = [cells[i] for i in train_cells_idx]
            val_cells = [cells[i] for i in val_cells_idx]
            
            # Initialize the index lists
            # In essence, we iterate over the train and validation cells 
            # Any sample belonging to a training cell will go into the training set
            # Likewise, any sample belonging to a validation cell will go into a validation set
            # IN DOING SO, we ensure that no cell line in the validation set appears in the training set, thus enforcing cell-blind
            train_idx = []
            val_idx = []
            for cell in train_cells:
                train_idx.extend(cell_to_indices[cell])
            for cell in val_cells:
                val_idx.extend(cell_to_indices[cell])
            
            splits.append({
                'train': train_idx,
                'val': val_idx
            })
            
            # Output metrics before returning
            print(f"  Fold {fold_idx+1}: {len(train_cells)} train cells, "
                  f"{len(val_cells)} val cells → "
                  f"{len(train_idx)} train pairs, {len(val_idx)} val pairs")
        
        return splits
    
    # This function handles the drug-blind split
    # We ensure that no drugs in the validation set are seen in the training set
    # In essence, the model is evaluated entirely on unseen drugs
    # This is a similar implementation to cell-blind split, we just replace cells with drugs
    # As such, we don't repeat the same comments for cleaner code
    def drug_blind_split(self):
        print(f"\nCreating drug-blind split ({self.n_folds}-fold CV)...")
        
        drug_to_indices = defaultdict(list)
        for idx, resp in enumerate(self.responses):
            drug_to_indices[resp['drug_id']].append(idx)
        
        drugs = list(drug_to_indices.keys())
        print(f"  Total unique drugs: {len(drugs)}")
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        splits = []
        for fold_idx, (train_drugs_idx, val_drugs_idx) in enumerate(kf.split(drugs)):
            train_drugs = [drugs[i] for i in train_drugs_idx]
            val_drugs = [drugs[i] for i in val_drugs_idx]
            
            train_idx = []
            val_idx = []
            for drug in train_drugs:
                train_idx.extend(drug_to_indices[drug])
            for drug in val_drugs:
                val_idx.extend(drug_to_indices[drug])
            
            splits.append({
                'train': train_idx,
                'val': val_idx
            })
            
            print(f"  Fold {fold_idx+1}: {len(train_drugs)} train drugs, "
                  f"{len(val_drugs)} val drugs → "
                  f"{len(train_idx)} train pairs, {len(val_idx)} val pairs")
        
        return splits
    
    # This function handles the cell-drug-blind Kfold split
    # This is the hardest to implement split as we have to ensure that both cell AND drugs that appear
    #   in the validation set don't appear in the training set
    # In essence, the model will be evaluated on both unseen cell lines and unseen drug lines
    def cell_drug_blind_split(self):
        print(f"\nCreating cell-drug-blind split ({self.n_folds}-fold CV)...")
        
        # For this implementation, we have to get the unique cell ID's AND drug ID's
        # Additionally including output logging for debugging purposes
        cells = list(set(r['cell_id'] for r in self.responses))
        drugs = list(set(r['drug_id'] for r in self.responses))
        print(f"  Total unique cells: {len(cells)}")
        print(f"  Total unique drugs: {len(drugs)}")
        
        # We do 2 Kfold splitters, one for cells and one for drugs
        kf_cell = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        kf_drug = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state + 1)  

        # Since we have 2 different splitters, we can't exactly do the same for loop iteration that we did previously
        # As a result, we obtain the split cells and drugs here itself so that in the upcoming iteration,
        #   we can simply use this list in order to get the cell and drug fold splits        
        cell_splits = list(kf_cell.split(cells))
        drug_splits = list(kf_drug.split(drugs)) 
        splits = []

        # Iterate through the number of folds and manually construct each of the train/val sets from the fold
        for fold_idx in range(self.n_folds):
            # Obtain train/val sets for both cells and drugs
            train_cells = set(cells[i] for i in cell_splits[fold_idx][0])
            val_cells = set(cells[i] for i in cell_splits[fold_idx][1])
            train_drugs = set(drugs[i] for i in drug_splits[fold_idx][0])
            val_drugs = set(drugs[i] for i in drug_splits[fold_idx][1])
            
            # Initialize indices, the main difference is that we have a discarded list
            # This is because there is a real chance that a case does not fit our cell-drug-blind condition
            # In this case, we will have to simply discard that sample
            train_idx = []
            val_idx = []
            discarded = 0
            
            # This iteration handles assigning each sample to the appropriate list
            # It is more similar to the previous split implementations in terms of assignment
            for idx, resp in enumerate(self.responses):
                cell = resp['cell_id']
                drug = resp['drug_id']
                
                # The main difference is that we need to ensure that the cell and drugs samples 
                #   are BOTH in the respective validation sets, not just the cells and drugs
                if cell in val_cells and drug in val_drugs:
                    val_idx.append(idx)
                elif cell in train_cells and drug in train_drugs:
                    train_idx.append(idx)
                # Handles discards
                else:
                    discarded += 1
            
            # Basic assignment to splits list
            splits.append({
                'train': train_idx,
                'val': val_idx
            })
            
            # Output metrics and stats for debugging purposes
            print(f"  Fold {fold_idx+1}: "
                  f"{len(train_cells)} train cells, {len(val_cells)} val cells, "
                  f"{len(train_drugs)} train drugs, {len(val_drugs)} val drugs → "
                  f"{len(train_idx)} train pairs, {len(val_idx)} val pairs "
                  f"({discarded} discarded)")
        
        return splits
    
    # Main function that handles the creation of all the splits
    # Essentially calls all the functions to create specific splits and saves it to the correct destination
    def create_all_splits(self, save_dir='data/processed'):
        # Distinguishes splits creation from any previous terminal process which helps when debugging sections
        print("="*60)
        print("Creating All CV Splits")
        print("="*60)
        
        # Create a dictionary of all the splits
        splits_dict = {
            'random': self.random_split(),
            'cell_blind': self.cell_blind_split(),
            'drug_blind': self.drug_blind_split(),
            'cell_drug_blind': self.cell_drug_blind_split()
        }
        
        # Save all the splits to the correct destination in data/processed
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save as a .pkl file 
        for split_type, splits in splits_dict.items():
            save_path = f'{save_dir}/splits_{split_type}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(splits, f)
            print(f"\n✓ Saved {save_path}")
        
        # More fancy formatting to signify end of splits, useful for debugging
        print("\n" + "="*60)
        print("All splits created and saved!")
        print("="*60)
        
        return splits_dict

# Helper function to load_splits, useful in other files that need to access the splits and import splits.py to use it
def load_splits(split_type, data_dir='data/processed'):
    """Load a specific split type"""
    with open(f'{data_dir}/splits_{split_type}.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # Load preprocessed data
    print("Loading preprocessed data...")
    with open('data/processed/response_data.pkl', 'rb') as f:
        responses = pickle.load(f)
    
    print(f"Loaded {len(responses)} response pairs")
    
    # Create all splits
    splitter = DataSplitter(responses, n_folds=5, random_state=42)
    splitter.create_all_splits(save_dir='data/processed')