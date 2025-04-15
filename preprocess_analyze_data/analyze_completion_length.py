import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd

def analyze_code_lengths(json_file_path):
    """
    Analyze code completion lengths from a JSON dataset using the Qwen2.5-Coder tokenizer.
    
    Args:
        json_file_path (str): Path to the JSON file containing prompt-completion pairs
        
    Returns:
        dict: Statistics of code lengths in tokens
    """
    # Load tokenizer for Qwen2.5-Coder-7B-Instruct
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    
    # Load the dataset
    print(f"Loading dataset from {json_file_path}")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Calculate token lengths for all completions
    print(f"Analyzing {len(data)} samples...")
    token_lengths = []
    
    for item in tqdm(data):
        completion = item["completion"]
        # Tokenize the completion
        tokens = tokenizer.encode(completion)
        token_lengths.append(len(tokens))
    
    # Convert to numpy array for statistics
    token_lengths = np.array(token_lengths)
    
    # Calculate statistics
    stats = {
        "count": len(token_lengths),
        "mean": token_lengths.mean(),
        "median": np.median(token_lengths),
        "min": token_lengths.min(),
        "max": token_lengths.max(),
        "std": token_lengths.std(),
        "25th_percentile": np.percentile(token_lengths, 25),
        "75th_percentile": np.percentile(token_lengths, 75),
        "90th_percentile": np.percentile(token_lengths, 90),
        "95th_percentile": np.percentile(token_lengths, 95),
        "99th_percentile": np.percentile(token_lengths, 99),
    }
    
    # Create a DataFrame for the token lengths
    df = pd.DataFrame({"token_length": token_lengths})
    
    # Generate visualizations
    print("Generating visualizations...")
    visualize_token_length_distribution(df, stats)
    
    return stats, df

def visualize_token_length_distribution(df, stats):
    """
    Create visualizations of token length distribution.
    
    Args:
        df (pd.DataFrame): DataFrame containing token lengths
        stats (dict): Statistics calculated from the token lengths
    """
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Code Completion Token Length Analysis', fontsize=16)
    
    # 1. Histogram with KDE
    ax1 = axes[0, 0]
    sns.histplot(df["token_length"], kde=True, ax=ax1)
    ax1.set_title('Distribution of Token Lengths')
    ax1.set_xlabel('Number of Tokens')
    ax1.set_ylabel('Frequency')
    
    # Add vertical lines for statistics
    ax1.axvline(stats["mean"], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.1f}')
    ax1.axvline(stats["median"], color='green', linestyle='--', label=f'Median: {stats["median"]:.1f}')
    ax1.axvline(stats["95th_percentile"], color='purple', linestyle='--', label=f'95th %: {stats["95th_percentile"]:.1f}')
    ax1.legend()
    
    # 2. Box plot
    ax2 = axes[0, 1]
    sns.boxplot(y=df["token_length"], ax=ax2)
    ax2.set_title('Token Length Box Plot')
    ax2.set_ylabel('Number of Tokens')
    
    # 3. Cumulative distribution function
    ax3 = axes[1, 0]
    values, base = np.histogram(df["token_length"], bins=40)
    cumulative = np.cumsum(values) / len(df["token_length"])
    ax3.plot(base[:-1], cumulative, color='blue')
    
    # Add vertical lines for percentiles
    percentiles = [50, 90, 95, 99]
    percentile_values = [stats["median"], stats["90th_percentile"], 
                         stats["95th_percentile"], stats["99th_percentile"]]
    percentile_colors = ['green', 'orange', 'purple', 'red']
    
    for p, val, color in zip(percentiles, percentile_values, percentile_colors):
        ax3.axvline(val, color=color, linestyle='--', 
                   label=f'{p}th percentile: {val:.1f}')
    
    ax3.set_title('Cumulative Distribution Function')
    ax3.set_xlabel('Number of Tokens')
    ax3.set_ylabel('Cumulative Proportion')
    ax3.legend()
    
    # 4. Log-scale histogram for the long tail
    ax4 = axes[1, 1]
    sns.histplot(df["token_length"], log_scale=(False, True), ax=ax4)
    ax4.set_title('Token Length Distribution (Log Scale)')
    ax4.set_xlabel('Number of Tokens')
    ax4.set_ylabel('Frequency (Log Scale)')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('token_length_analysis.png', dpi=300)
    print("Visualization saved as 'token_length_analysis.png'")
    
    # Create a summary figure for the most important statistics
    plt.figure(figsize=(10, 6))
    stats_to_show = ["mean", "median", "75th_percentile", "90th_percentile", 
                     "95th_percentile", "99th_percentile"]
    labels = ["Mean", "Median", "75th %", "90th %", "95th %", "99th %"]
    values = [stats[s] for s in stats_to_show]
    
    plt.bar(labels, values, color=sns.color_palette("viridis", len(labels)))
    for i, v in enumerate(values):
        plt.text(i, v + 5, f"{v:.1f}", ha='center')
    
    plt.title('Key Token Length Statistics')
    plt.ylabel('Number of Tokens')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('token_length_key_stats.png', dpi=300)
    print("Key statistics visualization saved as 'token_length_key_stats.png'")

def print_recommendation(stats):
    """
    Print recommendations for context window size based on the statistics.
    
    Args:
        stats (dict): Statistics calculated from the token lengths
    """
    print("\n===== Context Window Size Recommendation =====")
    print(f"Mean token length: {stats['mean']:.2f}")
    print(f"Median token length: {stats['median']:.2f}")
    print(f"95th percentile: {stats['95th_percentile']:.2f}")
    print(f"99th percentile: {stats['99th_percentile']:.2f}")
    
    # Consider headroom for prompts and other elements
    headroom_factor = 1.5  # Extra space for prompts and model overhead
    recommended_min = stats["95th_percentile"] * headroom_factor
    recommended_comfortable = stats["99th_percentile"] * headroom_factor
    
    print(f"\nMinimum recommended context window: {int(recommended_min)} tokens")
    print(f"Comfortable recommended context window: {int(recommended_comfortable)} tokens")
    print(f"This includes a {headroom_factor}x multiplier for prompt space and overhead.")
    
    # Check if any extreme outliers exist
    if stats["max"] > stats["99th_percentile"] * 2:
        print("\nWarning: Your dataset contains extreme outliers!")
        print(f"Maximum token length: {stats['max']}")
        print("Consider examining these outliers or setting a maximum sequence length cap.")

if __name__ == "__main__":
    # Replace with your actual JSON file path
    json_file_path = "/ephemeral/grpo-data/dojo_sft_js_grpo-format.json"
    
    # Run the analysis
    stats, df = analyze_code_lengths(json_file_path)
    
    # Print detailed statistics
    print("\n===== Token Length Statistics =====")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    # Print recommendations
    print_recommendation(stats)
    
    # Optional: Save the dataframe with token lengths for further analysis
    df.to_csv("token_lengths.csv", index=False)
    print("\nToken lengths saved to 'token_lengths.csv'")