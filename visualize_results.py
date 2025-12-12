# visualize_results.py
# This is a nice file we have that enables analysis of our results
# Along with the json results that we have, we also generate a plot and command line table to help determine how good our performance was

# Documentation for visualization help:
#   https://seaborn.pydata.org/
#   https://www.geeksforgeeks.org/python/introduction-to-seaborn-python/
#   https://matplotlib.org/stable/index.html
#   https://www.w3schools.com/python/matplotlib_intro.asp

# We need the json files as well as certain visualization helpers, such as matplotlib and seaborn
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set the global plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Helper function that loads the results for a split type. We need this in order to get the actual data that we plot
def load_results(split_type):
    results_path = Path(f'results/{split_type}/aggregated_results.json')
    with open(results_path, 'r') as f:
        return json.load(f)

# This first plot is for comparing the split types in terms of their metrics
def plot_comparison_bar():
    # Obtain the split types as well as the split labels for graph purposes
    split_types = ['random', 'cell_blind', 'drug_blind', 'cell_drug_blind']
    split_labels = ['Random', 'Cell-Blind', 'Drug-Blind', 'Cell-Drug-Blind']
    
    # Load the results using our helper function
    results = {st: load_results(st) for st in split_types}
    
    # Obtain the metrics as well as the labels for graphing purposes
    metrics = ['mse', 'mae', 'rmse']
    metric_labels = ['MSE', 'MAE', 'RMSE']
    
    # We will need 3 subplots because we have 3 metrics to visualize (MSE, MAE, RMSE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loop that goes through the metrics and populates the graph
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # To enable richer analysis, we extract both the mean and std for this graph (it will become clear why later on)
        means = [results[st]['aggregated'][metric]['mean'] for st in split_types]
        stds = [results[st]['aggregated'][metric]['std'] for st in split_types]
        
        # Create bars using the appropriate parameters (view documentation links for specific parameters used)
        bars = axes[idx].bar(split_labels, means, yerr=stds, capsize=5, alpha=0.7, color=["#8bc34a", '#03a9f4', '#f44336', '#9c27b0'])
        
        # This is why we extract the std values earlier on
        # To enable higher-quality analysis, our results will display in the form "mean ± std"
        # The actual code is essentially just adding a text above the bar but that text in particular will be in the above format so that we can see how much the metric values varied per fold
        # View documentation for specifics on axes implementation
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height, f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Basic labeling on the graph parameters
        axes[idx].set_ylabel(label, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Split Type', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{label} Across Split Types (Baseline - 7.87)', fontsize=13, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=15)
        
        # To highlight the improvement we got per split, we add a baseline comparison for MSE
        # The baseline MSE is, of course, just the average
        # In other words, a barebones implementation of a "prediction" model would just be "predict the same value regardless of cell or drug" 
        #   and that is done by calculating the mean
        # Our model calculates things differently, however, and we want to be able to effectively compare how much better this approach is compared to our baseline
        if metric == 'mse':
            # The baseline value is 7.87, so we draw a line at this value and label it as the baseline. 
            # The further any split's MSE is from this baseline, the better it is at predicting IC50 value.
            axes[idx].axhline(y=7.87, color='red', linestyle='--', linewidth=2, label='Baseline')
    
    # Save the plot at the appropriate destination
    plt.tight_layout()
    plt.savefig('results/comparison_bar_plot.png', dpi=300, bbox_inches='tight')
    print("Saved: results/comparison_bar_plot.png")
    plt.close()

# Alongside the graph visualization, we also generate a summary table that outputs in the terminal using fancy formatting.
# In all honesty, SOL isn't the best place to view graph visualizations, so this table is the main way of checking final performance metrics on SOL
# However, on a personal machine, the graph visualization is a nice and effective way at analyzing performance
def create_summary_table():
    # Same setup as before regarding types and labels
    split_types = ['random', 'cell_blind', 'drug_blind', 'cell_drug_blind']
    split_labels = ['Random', 'Cell-Blind', 'Drug-Blind', 'Cell-Drug-Blind']
    
    # Same loading of results using the helper function
    results = {st: load_results(st) for st in split_types}
    
    # Once again using the baseline to analyze split performance
    baseline_mse = 7.87
    
    # Fancy formatting once again helps separate the final performance metrics from any other pipeline phase that displays in the command line
    print("\n" + "="*100)
    print("SUMMARY TABLE: Model Performance Across All Split Types")
    print("="*100)
    print(f"{'Split Type':<20} {'MSE':<20} {'MAE':<15} {'RMSE':<10} {'Improvement'}")
    print("-"*100)
    
    # Loop that populates the table with the appropriate values and labels
    for st, label in zip(split_types, split_labels):
        mse = results[st]['aggregated']['mse']['mean']
        mse_std = results[st]['aggregated']['mse']['std']
        mae = results[st]['aggregated']['mae']['mean']
        mae_std = results[st]['aggregated']['mae']['std']
        rmse = results[st]['aggregated']['rmse']['mean']
        rmse_std = results[st]['aggregated']['rmse']['std']
        # We additionally calculate improvement over the baseline
        improvement = (baseline_mse - mse) / baseline_mse * 100
        
        # Fancy formatting to print in the "mean ± std" format
        print(f"{label:<20} {mse:.2f}±{mse_std:.2f}        "
              f"{mae:.2f}±{mae_std:.2f}        "
              f"{rmse:.2f}±{rmse_std:.2f}        "
              f"{improvement:.1f}%")
    
    # More fancy formatting for debugging and separation purposes
    # Sometimes causes issue depending on terminal size but worked for the team's purposes
    print("-"*100)
    print(f"{'Baseline (Mean)':<20} {baseline_mse:.2f}")
    print("="*100 + "\n")

# Main function that generates the visualizations and summaries
def main():
    # Fancy separation for debugging purposes
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60 + "\n")
    
    # The results path should already exist but if not, this creates it. 
    # Once again, this enforces the ability for our project to be ran at any phase of the pipeline without issues
    Path('results').mkdir(exist_ok=True)
    
    # Generate the graph visualization
    plot_comparison_bar()
    
    # Generate and print the command line table
    create_summary_table()
    
    # More fancy separation for debugging purposes
    # Essentially just tells the user that visualization has been generated and the files that it generates
    print("\n" + "="*60)
    print("All visualizations generated!")
    print("="*60)
    print("\nGenerated files:")
    print("  - results/comparison_bar_plot.png")
    print("\n")

if __name__ == '__main__':
    main()
