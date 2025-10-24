"""
TWINS Dataset Experiment Runner for DeR-CFR

TWINS Dataset Configuration:
- Binary outcome (mortality: 0 or 1)
- Has counterfactual ground truth (ycf available)
- Does NOT have mu0/mu1 (unlike IHDP)
- Uses evaluate_cont_ate (modified to handle missing mu0/mu1)
- Metrics: PEHE, Bias_ATE, RMSE_fact, RMSE_cfact
"""
import subprocess
import sys
import os

# TWINS dataset configuration
TWINS_CONFIG = {
    # Data configuration
    'datadir': 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/',
    'dataform': 'twins_1-10.train.npz',
    'data_test': 'twins_1-10.test.npz',
    'outdir': 'results/example_twins/',
    
    # Training configuration
    'experiments': 10,         # TWINS has 10 experiments
    'iterations': 300,         # Training iterations
    'batch_size': 0,           # 0 = full batch
    'lrate': 1e-3,            # Initial learning rate
    'lrate_decay': 0.97,      # Learning rate decay
    'val_part': 0.3,          # Validation set ratio
    
    # Network architecture
    'n_in': 7,                # Representation layers
    'n_out': 7,               # Output layers
    'n_t': 3,                 # Treatment layers
    'dim_in': 64,             # Representation dimension
    'dim_out': 64,           # Output dimension
    
    # Loss function weights (TWINS optimized configuration)
    'p_coef_y': 1.0,          # Outcome regression loss
    'p_coef_alpha': 1e-2,     # α: Adjustment decomposition loss
    'p_coef_beta': 1e-3,         # β: Instrumental variable decomposition loss
    'p_coef_gamma': 1e-3,     # γ: Balance loss
    'p_coef_mu': 5,           # μ: Orthogonality regularization
    'p_coef_lambda': 5,    # λ: L2 regularization
    
    # TWINS specific configuration
    'loss': 'log',            # Binary classification uses log loss
    'ycf_result': 1,          # TWINS has counterfactual ground truth
    'batch_norm': 1,          # Use batch normalization
    'autoWeighting': 1,       # Enable auto-weighting (same as Jobs)
    'constrainedLayer': 0,    # 0 = all layers (per paper recommendation for TWINS)
    
    # Other configuration
    'seed': 1,
    'optimizer': 'Adam',
    'imb_fun': 'mmd_lin',
    'output_delay': 100,
    'pred_output_delay': 30,
}

# Python interpreter path
PYTHON_EXE = r'C:\tool\Anaconda3\envs\DeR_CFR_tf1\python.exe'

def run_twins():
    """Run TWINS experiment"""
    
    print("=" * 80)
    print("Running DeR-CFR TWINS Dataset Experiment")
    print("=" * 80)
    print(f"\nPython Interpreter: {PYTHON_EXE}")
    print(f"Dataset: {TWINS_CONFIG['dataform']}")
    print(f"Experiments: {TWINS_CONFIG['experiments']}")
    print(f"Iterations: {TWINS_CONFIG['iterations']}")
    print(f"Batch Size: {TWINS_CONFIG['batch_size']}")
    print("\nHyperparameter Configuration:")
    print(f"  alpha (p_coef_alpha):  {TWINS_CONFIG['p_coef_alpha']}")
    print(f"  beta  (p_coef_beta):   {TWINS_CONFIG['p_coef_beta']}")
    print(f"  gamma (p_coef_gamma):  {TWINS_CONFIG['p_coef_gamma']}")
    print(f"  mu    (p_coef_mu):     {TWINS_CONFIG['p_coef_mu']}")
    print(f"  lambda(p_coef_lambda): {TWINS_CONFIG['p_coef_lambda']}")
    print("\nTWINS Special Settings:")
    print(f"  Loss Function: log (binary classification)")
    print(f"  Counterfactual Truth: Yes (ycf_result=1)")
    print(f"  Note: TWINS has ycf but NO mu0/mu1 (different from IHDP)")
    print("=" * 80)
    print()
    
    # Build command line arguments
    cmd = [PYTHON_EXE, 'main.py']
    
    for key, value in TWINS_CONFIG.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key}={value}')
    
    # Run command
    try:
        # Get current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(cmd, check=True, cwd=script_dir)
        print("\n" + "=" * 80)
        print("[SUCCESS] TWINS experiment completed!")
        print("=" * 80)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"[ERROR] Experiment failed! Return code: {e.returncode}")
        print("=" * 80)
        return e.returncode
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"[ERROR] Error occurred: {e}")
        print("=" * 80)
        return 1

if __name__ == '__main__':
    exit_code = run_twins()
    sys.exit(exit_code)
