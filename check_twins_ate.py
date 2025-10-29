import numpy as np

# Load TWINS data
data = np.load('C:/Users/0702ty/OneDrive/Desktop/DRLECB/twins_1-10.train.npz')

print("="*80)
print("TWINS Dataset Analysis - Checking True ATE")
print("="*80)

print("\nAvailable keys:", list(data.keys()))
print("Has ycf:", 'ycf' in data)
print("Has mu0/mu1:", 'mu0' in data and 'mu1' in data)

# Use first experiment
i_exp = 0
t = data['t'][:,i_exp]
yf = data['yf'][:,i_exp]
ycf = data['ycf'][:,i_exp] if 'ycf' in data else None

print("\n" + "="*80)
print("Basic Statistics")
print("="*80)
print(f"Sample size: {len(t)}")
print(f"Treatment rate: {np.mean(t):.3f}")
print(f"Number treated (t=1): {np.sum(t>0)}")
print(f"Number control (t=0): {np.sum(t<1)}")

print("\n" + "="*80)
print("Observed Outcomes")
print("="*80)
print(f"Mean outcome for treated (t=1): {np.mean(yf[t>0]):.4f}")
print(f"Mean outcome for control (t=0): {np.mean(yf[t<1]):.4f}")
print(f"Naive ATE (treated - control): {np.mean(yf[t>0]) - np.mean(yf[t<1]):.4f}")

if ycf is not None:
    print("\n" + "="*80)
    print("TRUE Causal Effects (from counterfactuals)")
    print("="*80)
    
    # Calculate ITE correctly
    # ITE = Y(1) - Y(0)
    # For control (t=0): ITE = ycf - yf (ycf is Y(1), yf is Y(0))
    # For treated (t=1): ITE = yf - ycf (yf is Y(1), ycf is Y(0))
    ite = np.zeros_like(yf)
    ite[t < 1] = ycf[t < 1] - yf[t < 1]  # Control: Y(1) - Y(0)
    ite[t > 0] = yf[t > 0] - ycf[t > 0]  # Treated: Y(1) - Y(0)
    
    print(f"True ATE (Average Treatment Effect): {np.mean(ite):.6f}")
    print(f"True ATT (Average Treatment Effect on Treated): {np.mean(ite[t>0]):.6f}")
    print(f"True ATC (Average Treatment Effect on Control): {np.mean(ite[t<1]):.6f}")
    
    print("\n" + "="*80)
    print("Interpretation")
    print("="*80)
    if np.mean(ite) < 0:
        print(f"✓ Treatment REDUCES the outcome by {abs(np.mean(ite)):.6f}")
        print("  (For TWINS dataset, outcome is mortality, so negative is GOOD)")
    else:
        print(f"✗ Treatment INCREASES the outcome by {np.mean(ite):.6f}")
    
    print("\n" + "="*80)
    print("Detailed Breakdown")
    print("="*80)
    print("For TREATED individuals (t=1):")
    print(f"  Observed outcome (with treatment): {np.mean(yf[t>0]):.4f}")
    print(f"  Counterfactual (without treatment): {np.mean(ycf[t>0]):.4f}")
    print(f"  Effect (observed - counterfactual): {np.mean(yf[t>0] - ycf[t>0]):.6f}")
    
    print("\nFor CONTROL individuals (t=0):")
    print(f"  Observed outcome (without treatment): {np.mean(yf[t<1]):.4f}")
    print(f"  Counterfactual (with treatment): {np.mean(ycf[t<1]):.4f}")
    print(f"  Effect (counterfactual - observed): {np.mean(ycf[t<1] - yf[t<1]):.6f}")
    
    # Check all 10 experiments
    print("\n" + "="*80)
    print("ATE across all 10 experiments")
    print("="*80)
    ates = []
    for i in range(10):
        t_i = data['t'][:,i]
        yf_i = data['yf'][:,i]
        ycf_i = data['ycf'][:,i]
        
        ite_i = np.zeros_like(yf_i)
        ite_i[t_i < 1] = ycf_i[t_i < 1] - yf_i[t_i < 1]
        ite_i[t_i > 0] = yf_i[t_i > 0] - ycf_i[t_i > 0]
        
        ate_i = np.mean(ite_i)
        ates.append(ate_i)
        print(f"Experiment {i+1}: ATE = {ate_i:.6f}")
    
    print(f"\nMean ATE across experiments: {np.mean(ates):.6f} ± {np.std(ates):.6f}")

else:
    print("\n[WARNING] No counterfactual data available!")

print("\n" + "="*80)
