# train.py
# This is one of the most imperative files in the entire project
# It handles the training and evaluation of the model all in one file. 
# Additionally allows for CLI arguments so that the user has some customization with what split they want to train and evaluate

# Mostly self explanatory imports
# Notably, we must import the classes from our dataset.py, drp_model.py, and metrics.py in order to use those functions
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import os

from data.dataset import DRPDataset, load_data, load_splits
from models.drp_model import DRPModel
from utils.metrics import calculate_metrics, print_metrics

# The main class that handles both training and evaluation
class Trainer:
    # Takes in the arguments of the model, device, learning rate, and weight decay
    # Notably, we push to use the GPU during training, as this speeds up the process immensely
    # We HIGHLY recommend the utilization of ASU's SOL clusters as it finishes all training in around ~30-40 minutes with a properly setup environment
    def __init__(self, model, device='cuda', lr=1e-3, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        # We choose to use the Adam optimizer as it is reliable and trusted
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
    
    # This function handles one full training epoch over the dataset. 
    def train_epoch(self, train_loader):
        # This enables dropout as well as gradient updates
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # We loop through the batches and follow a pretty rudimentary training script utilizing tqdm for a progress bar
        # This is essentially how all manual training scripts are designed
        for batch in tqdm(train_loader, desc='Training', leave=False):
            # First, we move the batch data to the device, aka the cell line features, drug features, and the target data
            cell_feat = batch['cell_features'].to(self.device)
            drug_feat = batch['drug_features'].to(self.device)
            target = batch['log_ic50'].squeeze().to(self.device)
            
            # Next, we run a simple forward pass on the data and obtain our loss
            pred = self.model(cell_feat, drug_feat)
            loss = self.criterion(pred, target)
            
            # Now, we utilize backpropagation, aka the backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update our counter variables as we will need this to eventually return average loss over the epoch
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    # This function handles the evaluation phase on the validation set with no gradient computation
    def evaluate(self, val_loader):
        # Enable eval mode
        self.model.eval()
        predictions = []
        targets = []
        
        # Again, we do NOT utilize gradient updates to speed up the model
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating', leave=False):
                # We load the batch data to the device again
                cell_feat = batch['cell_features'].to(self.device)
                drug_feat = batch['drug_features'].to(self.device)
                target = batch['log_ic50'].squeeze().to(self.device)
                
                # Obtain the prediction
                pred = self.model(cell_feat, drug_feat)
                
                # Add it to our lists that we initialized earlier in the proper formats
                predictions.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # Convert these lists to an array (see utils.metrics.py -> we need np arrays to run our metric calculati ons)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Obtain the metrics and return it (see utils.metrics.py for documentation on calculate_metrics(predictions, targets) function)
        metrics = calculate_metrics(predictions, targets)

        return metrics

# This function runs a singular fold of our cross-validation
# As such, we need the fold index, train/val indices, as well as our data and save directory
def run_fold(fold_idx, train_idx, val_idx, cell_features, drug_features, responses, config, save_dir):
    # For debugging purposes, we use separation print formatting so that it is clear that a specific fold is being run
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/{config['n_folds']}")
    print(f"{'='*60}")
    
    # Recall in data/dataset.py, we store the data in triples (cell, drug, ic50)
    # However, the response data was never split in data/splits.py
    # As such, we need to slice into the train and val subsets in order to get the appropriate response data for train/val
    train_responses = [responses[i] for i in train_idx]
    val_responses = [responses[i] for i in val_idx]
    
    # Now that we have the responses, we can officially create the DRP dataset objects 
    train_dataset = DRPDataset(train_responses, cell_features, drug_features)
    val_dataset = DRPDataset(val_responses, cell_features, drug_features)
    
    # Now that we have those datasets, we can create the Dataloaders which deals with batching and shuffling
    # This is already in data/dataset.py but here is the documentation again to help setup these parameters
    # Documentation used to help with implementing this DataLoader
    #   https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    #   https://www.geeksforgeeks.org/deep-learning/pytorch-dataloader/
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  
        pin_memory=True if config['device'] == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Refer to the model documentation in models/drp_model.py
    # Here we simply call the mdoel with the appropriate args
    model = DRPModel(
        cell_in=7107,
        drug_in=2048,
        embed_dim=config['embed_dim'],
        dropout=config['dropout']
    )
    
    # We create the trainer with the apropriate args (see above for more on Trainer documentation)
    trainer = Trainer(
        model,
        device=config['device'],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # This is the training loop that we use along with some important initialized metric variables
    # We allow early stopping, for instance, when the model converges quicker than our max_epochs (set to 100)
    best_val_mse = float('inf')
    patience_counter = 0
    train_losses = []
    val_metrics_history = []
    
    # Main loop
    for epoch in range(config['max_epochs']):
        # We call our earlier functions that deal with training one individual epoch and evaluating said epoch
        train_loss = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        
        # We append the metrics to our earlier initialized variables. 
        train_losses.append(train_loss)
        val_metrics_history.append(val_metrics)
        
        # Useful debugging output that shows the metric stats at each epoch (we can manually examine to see if there is constant improvement)
        print(f"Epoch {epoch+1}/{config['max_epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val ", end="")
        print_metrics(val_metrics)
        
        # This conditional helps stop the training if we determine that the performance is not getting better after any epoch
        # Essentially, if the model's MSE is not improving after some epochs, we can simply end the training there itself as it is evident the model won't get any better results
        # Documentation for early stopping and why it is useful: 
        #   https://www.geeksforgeeks.org/deep-learning/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/

        # If we get a better MSE val, we update the tracker variable and save the model stats as a checkpoint in the appropriate destination
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            patience_counter = 0
            checkpoint_path = save_dir / f"fold_{fold_idx}_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics
            }, checkpoint_path)
        # However, if we do not notice improvement, we will increase the patience counter.
        # If this counter reaches our threshold that we set in the hyperparameter config, the model will stop being trained and incur early stopping
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model that we had that we saved in the appropriate destination
    checkpoint = torch.load(save_dir / f"fold_{fold_idx}_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model = model.to(config['device'])
    
    # Obtain the final metrics based on this best model
    final_metrics = trainer.evaluate(val_loader)
    
    # Print out the metrics for debugging purposes
    print(f"\nFold {fold_idx+1} Final Results:")
    print_metrics(final_metrics, prefix="  ")
    
    # Save the fold results in a json friendly format.
    fold_results = {
        'fold_idx': fold_idx,
        'best_epoch': checkpoint['epoch'],
        'final_metrics': final_metrics,
        'train_losses': train_losses,
        'val_metrics_history': val_metrics_history
    }
    
    # Those results will now we saved in the appropriate destination as a json file enabling further analysis
    with open(save_dir / f"fold_{fold_idx}_results.json", 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    return final_metrics

# This function handles all folds of a specific split type, with the default set to random
def run_cross_validation(split_type='random', config=None):
    # Precautionary measure just in case hyperparameters were not defined
    if config is None:
        config = {
            'batch_size': 256,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'max_epochs': 100,
            'patience': 10,
            'embed_dim': 256,
            'dropout': 0.3,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'n_folds': 5
        }
    
    # For debugging purposes, we use fancy formatting when outputting the training process for a specific split type. 
    # This helps separate it from any other files that were run near it in the terminal
    print(f"\n{'#'*60}")
    print(f"Running {split_type.upper()} split")
    print(f"{'#'*60}")
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Max epochs: {config['max_epochs']}")
    print(f"Early stopping patience: {config['patience']}")
    
    # First, we load the data using the helper functions obtained from data/dataset.py
    print("\nLoading data...")
    cell_features, drug_features, responses = load_data()
    
    # next, we have to load the splits using the helper functions obtained from data/dataset.py
    print(f"Loading {split_type} splits...")
    splits = load_splits(split_type)
    
    # Create save directory for the outputs to go in
    save_dir = Path(f'results/{split_type}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint directory for model states to go in
    checkpoint_dir = Path(f'checkpoints/{split_type}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Run each fold using the appropriate args and obtain all the metrics for each fold in a list. This will then be saved to output
    all_metrics = []
    for fold_idx, split in enumerate(splits):
        fold_metrics = run_fold(
            fold_idx,
            split['train'],
            split['val'],
            cell_features,
            drug_features,
            responses,
            config,
            checkpoint_dir
        )
        all_metrics.append(fold_metrics)
    
    # Display the results across folds using fancy formatting for debugging purposes. Makes it easy to spot when scrolling up a large terminal
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {split_type.upper()} SPLIT")
    print(f"{'='*60}")
    for metric_name in ['mse', 'mae', 'rmse']:
        values = [m[metric_name] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric_name.upper()}: {mean:.4f} ± {std:.4f}")
    
    # Now we put these results and metrics in a json friendly format as we will store it in the appropriate destination in results/ which enables further statistical analysis
    # We decided on storing the mean and std dev for each metric type as well as the actual values themselves to enable richer insight
    aggregated_results = {
        'split_type': split_type,
        'config': config,
        'fold_metrics': all_metrics,
        'aggregated': {
            'mse': {
                'mean': float(np.mean([m['mse'] for m in all_metrics])),
                'std': float(np.std([m['mse'] for m in all_metrics])),
                'values': [float(m['mse']) for m in all_metrics]
            },
            'mae': {
                'mean': float(np.mean([m['mae'] for m in all_metrics])),
                'std': float(np.std([m['mae'] for m in all_metrics])),
                'values': [float(m['mae']) for m in all_metrics]
            },
            'rmse': {
                'mean': float(np.mean([m['rmse'] for m in all_metrics])),
                'std': float(np.std([m['rmse'] for m in all_metrics])),
                'values': [float(m['rmse']) for m in all_metrics]
            }
        }
    }
    
    # Save it to the appropriate destination in a json format
    with open(save_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    print(f"\nResults saved to {save_dir}")
    
    return aggregated_results

# CLI arguments
# This is something that we decided on implementing mainly because it makes the debugging process much easier. 
# With the implementation of CLI arguments we no longer have to train every split at once and then wait to see results. 
# We can simply train one at a time to make quick adjustments and examine results much easier

# Documentation for implementation help:
#   https://docs.python.org/3/library/argparse.html
#   https://www.geeksforgeeks.org/python/command-line-option-and-argument-parsing-using-argparse-in-python/
if __name__ == '__main__':
    import argparse
    
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Train drug response prediction model')
    
    # We allow arguments for a custom split type, batch size, learning rate, number of epochs, and patience
    # This allows for custom hyperparameter tuning from the command line itself, without having to mess with the code config
    # We have stable default values set but it really is up to the user
    parser.add_argument('--split_type', type=str, default='random', choices=['random', 'cell_blind', 'drug_blind', 'cell_drug_blind'], help='Type of data split for evaluation')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Parse the args
    args = parser.parse_args()
    
    # For the hyperparameter config, we let the user's argument determine many of them
    # The rest are just values that we agreed on after fine tuning
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'max_epochs': args.epochs,
        'patience': args.patience,
        'embed_dim': 256,
        'dropout': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_folds': 5
    }
    
    # Call the main function that runs the entire cross validation for a split type
    run_cross_validation(split_type=args.split_type, config=config)