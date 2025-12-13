# data/preprocessing.py
# Used to preprocess the data and get it into the format we need to feed it into the models

# Import necessary packages
# We use pandas to get the data into dataframes, numpy to aid in concatenation, and rdkit to transform the drug files
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path

# Main class to preprocess the data, uses the csv files stored in data/raw folder
class DataPreprocessor:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
    
    # This function is used to load the cell data and then concatenate it all into one feature vector that we can easily feed into the model
    # Debugging print statements throughout to ensure proper workflow
    def load_cell_features(self):
        print("Loading cell line data...")
        
        # Load the three cell files using pandas
        exp_df = pd.read_csv(self.data_dir / 'CCLE_2369_EXP.csv')
        mut_df = pd.read_csv(self.data_dir / 'CCLE_2369_hotspot_mut.csv')
        cnv_df = pd.read_csv(self.data_dir / 'CCLE_2369_binary_cnv.csv')
        
        # Here we set the index of the files to the cell line ID. 
        # After further inspection, the first column was titled "Unnamed: 0" which is why we used that to identify the column
        exp_df = exp_df.set_index('Unnamed: 0')
        mut_df = mut_df.set_index('Unnamed: 0')
        cnv_df = cnv_df.set_index('Unnamed: 0')
        
        # This is a precaution step that also ensures expansion to other datasets. We want to ensure that the cell lines and genes have the same order across the files.
        # This allows us to work through the datasets much more easily.
        assert list(exp_df.columns) == list(mut_df.columns) == list(cnv_df.columns), "Gene columns don't match across files"
        assert list(exp_df.index) == list(mut_df.index) == list(cnv_df.index), "Cell line indices don't match across files"
        
        # Outputs a metric telling us how many cell lines and genes exist
        print(f"Loaded {len(exp_df)} cell lines with {len(exp_df.columns)} genes")
        
        # Now we combine all of the features for each cell line. 
        # It starts with each file and after that we eventually combine that combination to get one master dictionary as a feature vector
        cell_features = {}
        for cell_id in exp_df.index:
            exp_vals = exp_df.loc[cell_id].values  
            mut_vals = mut_df.loc[cell_id].values  
            cnv_vals = cnv_df.loc[cell_id].values  
            
            # Here we concatenate the 3 combinations into one master vector
            # We additionally ensure that all of the features are an appropriate data type that will allow proper evaluation
            features = np.concatenate([exp_vals, mut_vals, cnv_vals])
            cell_features[cell_id] = features.astype(np.float32)
        
        # Outputs the length of the cell features as another metric to see what we are working with before returning
        print(f"Created cell features: {len(cell_features)} cells × 7107 features")

        return cell_features
    
    # Helper function that actually converts the SMILES to the Morgan fingerprint using the rdkit library
    def smiles_to_fingerprint(self, smiles, radius=2, n_bits=2048):
        # Convert to fingerprint with appropriate data types as well as debugging measures in place in case it doesn't work
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES: {smiles}")
            return np.zeros(n_bits, dtype=np.float32)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    
    # Main function to load the drug features from the drug csv as well as converting it to the fingerprints (we use helper function above for this)
    # Returns a dictionary of all the drug features in one vector
    def load_drug_features(self):
        print("Loading drug data...")
        
        # Read in the data and initialize the feature vector as well as a variable to track failures if any
        drug_df = pd.read_csv(self.data_dir / 'drug_smiles.csv')
        
        drug_features = {}
        failed_count = 0
        
        # We use a slightly different format for getting the features because the drug dataset has a different format. 
        # In essence , we try to get the CanonicalSMILES data from each drug, and if that doesn't work, we use the IsomericSMILES
        # We then call the helper function to convert it to a fingerprint and then add that to the feature vector
        # Appropriate print statements for debugging exist as well
        for _, row in drug_df.iterrows():
            drug_name = row['drug_name']
            
            smiles = row['CanonicalSMILES']
            if pd.isna(smiles):
                smiles = row['IsomericSMILES']
            
            if pd.isna(smiles):
                print(f"Warning: No SMILES for drug {drug_name}")
                failed_count += 1
                continue
            
            fp = self.smiles_to_fingerprint(smiles)
            drug_features[drug_name] = fp
        
        # Metrics to display the length of the drug feature vector as well as to identify any failures before returning
        print(f"Created drug features: {len(drug_features)} drugs × 2048 features")
        if failed_count > 0:
            print(f"Warning: Failed to process {failed_count} drugs")
        
        return drug_features
    
    # This function loads the IC50 response ground truth data and returns a list of dictionaries
    # Each dictionary has the format {cell id, drug id, log_ic50} and there is a list of it for each combination of cells and drugs
    def load_responses(self):
        print("Loading response data...")
        
        response_df = pd.read_csv(self.data_dir / 'sorted_IC50_82833_580_170.csv')
        
        # We extract the relevant columns after examining the .csv file to see the head column name which is seen below
        # Cell ID: "DepMap_ID"
        # Drug name: "Drug name" 
        # IC50: "IC50" 
        
        # Initialize list and a variable to track any skipped/invalid values
        responses = []
        skipped = 0
        
        # This is a similar iterative process to the drug file
        for _, row in response_df.iterrows():
            cell_id = row['DepMap_ID']
            drug_name = row['Drug name']
            log_ic50 = row['IC50']
            
            # Skip if IC50 is missing or invalid
            if pd.isna(log_ic50):
                skipped += 1
                continue
            
            # We append the list with the appropriate dictionary
            responses.append({
                'cell_id': cell_id,
                'drug_id': drug_name,  
                'log_ic50': float(log_ic50)
            })
        
        # Metric to display the length of the response vector and if any values were skipped/invalid
        print(f"Loaded {len(responses)} cell-drug response pairs")
        if skipped > 0:
            print(f"Skipped {skipped} pairs with missing IC50 values")
        
        return responses
    
    # This is a validation function that ensures all of the cell ids and drug ids in our ground truth dataset also exist in our feature data. 
    # This is imperative to ensure before we actually determine data splits or model creation or training or anything
    def verify_data_integrity(self, cell_features, drug_features, responses):
        print("\nVerifying data integrity...")
        
        # Get the cell and drug IDS from the ground truth vector
        response_cells = set(r['cell_id'] for r in responses)
        response_drugs = set(r['drug_id'] for r in responses)
        
        # We now get the IDS from both feature vectors and use simple differences to validate our data
        cells_in_features = set(cell_features.keys())
        drugs_in_features = set(drug_features.keys())
        
        missing_cells = response_cells - cells_in_features
        missing_drugs = response_drugs - drugs_in_features
        
        # Debugging print statements to determine the validity of the data
        print(f"Cells in responses: {len(response_cells)}")
        print(f"Cells with features: {len(cells_in_features)}")
        print(f"Missing cells: {len(missing_cells)}")
        
        print(f"\nDrugs in responses: {len(response_drugs)}")
        print(f"Drugs with features: {len(drugs_in_features)}")
        print(f"Missing drugs: {len(missing_drugs)}")
        
        # Additional print statements to display some of the missing IDS if they exist
        if missing_cells:
            print(f"Example missing cells: {list(missing_cells)[:5]}")
        
        if missing_drugs:
            print(f"Example missing drugs: {list(missing_drugs)[:5]}")
        
        # This is how we address missing data, by filtering it out of the ground truth dataset so that the only ground truth data that exists is something that exists in the feature data
        valid_responses = [
            r for r in responses 
            if r['cell_id'] in cells_in_features and r['drug_id'] in drugs_in_features
        ]
        
        # Prints out the number of valid responses as an additional debugging metric before returning
        print(f"\nValid response pairs: {len(valid_responses)}/{len(responses)}")
        
        return valid_responses
    
    # This is essentially the main function that runs all of the functions and additionally saves the results to pkl files in the appropriate destination
    def preprocess_all(self, save_dir='data/processed'):
        # Set the output destination if it does not exist
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Call the functions to process all of the data
        cell_features = self.load_cell_features()
        drug_features = self.load_drug_features()
        responses = self.load_responses()
        
        # Validate the data and filter accordingly if needed
        valid_responses = self.verify_data_integrity(
            cell_features, drug_features, responses
        )
        
        # Save the data as a .pkl file in the correct destination with print statements so that user knows if it was a successful save
        print(f"\nSaving preprocessed data to {save_dir}...")
        
        with open(save_dir / 'cell_features.pkl', 'wb') as f:
            pickle.dump(cell_features, f)
        print(f"✓ Saved cell_features.pkl")
        
        with open(save_dir / 'drug_features.pkl', 'wb') as f:
            pickle.dump(drug_features, f)
        print(f"✓ Saved drug_features.pkl")
        
        with open(save_dir / 'response_data.pkl', 'wb') as f:
            pickle.dump(valid_responses, f)
        print(f"✓ Saved response_data.pkl")
        
        # Print final output metrics of the preprocessing with fancy formatting separators to distinguish it from the previous logging
        print(f"\n{'='*60}")
        print("Preprocessing Complete.")
        print(f"{'='*60}")
        print(f"Cell features: {len(cell_features)} cells × 7107 features")
        print(f"  (2369 expression + 2369 mutation + 2369 CNV)")
        print(f"Drug features: {len(drug_features)} drugs × 2048 features")
        print(f"Responses: {len(valid_responses)} valid pairs")
        print(f"{'='*60}")


if __name__ == '__main__':
    preprocessor = DataPreprocessor(data_dir='data/raw')
    preprocessor.preprocess_all(save_dir='data/processed')