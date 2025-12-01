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

    # 2. Store data structure
    experiments = {}

    print(f"Found {len(files)} files. Processing...\n")

    for filepath in files:
        filename = os.path.basename(filepath)
        
        match = re.search(r"eps([\d\.]+)_keep(\d+)", filename)
        if not match:
            continue
            
        eps = float(match.group(1))
        tokens = int(match.group(2))
        
        try:
            df = pd.read_csv(filepath)
            
            # --- CORRECT INTERVAL SUBTRACTION ---
            # We want the bounds of (Unpruned - Pruned)
            # Min(Diff) = Min(Unpruned) - Max(Pruned)
            # Max(Diff) = Max(Unpruned) - Min(Pruned)
            
            # 1. Lower Bound of the Difference
            # (How much smaller could Unpruned be compared to Pruned?)
            diff_lower = (df['L_real_unpruned'] - df['U_real_pruned']).mean()
            
            # 2. Upper Bound of the Difference
            # (How much larger could Unpruned be compared to Pruned?)
            diff_upper = (df['U_real_unpruned'] - df['L_real_pruned']).mean()
            
            if eps not in experiments:
                experiments[eps] = {}
            
            experiments[eps][tokens] = (diff_lower, diff_upper)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # 3. Print Output
    sorted_eps = sorted(experiments.keys())
    
    for eps in sorted_eps:
        print(f"--- Epsilon: {eps} ---")
        # Adjust column widths for clarity
        print(f"{'Tokens Kept':<12} | {'Avg Diff Lower (L_P - U_P\')':<28} | {'Avg Diff Upper (U_P - L_P\')'}")
        print("-" * 75)
        
        token_counts = sorted(experiments[eps].keys())
        
        for k in token_counts:
            d_low, d_up = experiments[eps][k]
            print(f"{k:<12} | {d_low:<28.5f} | {d_up:.5f}")
        
        print("\n")

if __name__ == "__main__":
    analyze()
