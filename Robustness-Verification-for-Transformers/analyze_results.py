import pandas as pd
import glob
import os
import re

def analyze():
    # 1. Find all benchmark files
    files = glob.glob("results/bench_eps*_keep*.csv")
    
    if not files:
        print("No benchmark files found in results/ folder.")
        return

    # 2. Store data structure: data[epsilon][tokens] = (avg_diff_lower, avg_diff_upper)
    experiments = {}

    print(f"Found {len(files)} files. Processing...\n")

    for filepath in files:
        filename = os.path.basename(filepath)
        
        # Regex to extract Epsilon and Keep count from filename
        # Expects format: bench_eps0.001_keep5.csv
        match = re.search(r"eps([\d\.]+)_keep(\d+)", filename)
        if not match:
            continue
            
        eps = float(match.group(1))
        tokens = int(match.group(2))
        
        try:
            df = pd.read_csv(filepath)
            
            # Calculate Differences (Unpruned - Pruned)
            # Positive value = Bound got looser/worse (Pruned < Unpruned)
            # Negative value = Bound got tighter/better (Pruned > Unpruned)
            
            # Lower Bound Diff: (L_unpruned - L_pruned)
            # If P=20, P'=15 -> Diff=5 (Loss of 5)
            diff_lower = (df['L_real_unpruned'] - df['L_real_pruned']).mean()
            
            # Upper Bound Diff: (U_unpruned - U_pruned)
            # If P=20, P'=15 -> Diff=5
            diff_upper = (df['U_real_unpruned'] - df['U_real_pruned']).mean()
            
            if eps not in experiments:
                experiments[eps] = {}
            
            experiments[eps][tokens] = (diff_lower, diff_upper)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # 3. Print Output in Requested Format
    
    # Sort Epsilons
    sorted_eps = sorted(experiments.keys())
    
    for eps in sorted_eps:
        print(f"--- Epsilon: {eps} ---")
        print(f"{'Tokens Kept':<12} | {'Avg Diff Lower Bound':<22} | {'Avg Diff Upper Bound'}")
        print("-" * 60)
        
        # Sort Tokens
        token_counts = sorted(experiments[eps].keys())
        
        for k in token_counts:
            d_low, d_up = experiments[eps][k]
            
            # Format nicely
            print(f"{k:<12} | {d_low:<22.5f} | {d_up:.5f}")
        
        print("\n")

if __name__ == "__main__":
    analyze()
