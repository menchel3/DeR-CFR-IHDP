"""
Jobs Dataset Experiment Runner for DeR-CFR
Using specified Python interpreter and hyperparameter configuration
"""
import subprocess
import sys
import os

# Jobs dataset configuration
JOBS_CONFIG = {
    # Data configuration
    'datadir': 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/data/',
    'dataform': 'jobs_DW_bin.new.10.train.npz',
    'data_test': 'jobs_DW_bin.new.10.test.npz',
    'outdir': 'results/example_jobs/',
    
    # Training configuration
    'experiments': 10,         # Jobs has 10 experiments
    'iterations': 300,         # Training iterations
    'batch_size': 0,           # Jobs has more samples, use batch training
    'lrate': 1e-3,            # Initial learning rate
    'lrate_decay': 0.97,      # Learning rate decay
    'val_part': 0.3,          # Validation set ratio
    
    # Network architecture
    'n_in': 5,                # Representation layers
    'n_out': 4,               # Output layers
    'n_t': 1,                 # Treatment layers
    'dim_in': 32,             # Representation dimension
    'dim_out': 128,           # Output dimension
    
    # Loss function weights (Jobs optimized configuration)
    'p_coef_y': 1.0,          # Outcome regression loss
    'p_coef_alpha': 1e-2,     # α: Adjustment decomposition loss
    'p_coef_beta': 1,         # β: Instrumental variable decomposition loss
    'p_coef_gamma': 1e-2,     # γ: Balance loss
    'p_coef_mu': 5,           # μ: Orthogonality regularization
    'p_coef_lambda': 1e-3,    # λ: L2 regularization
    
    # Jobs specific configuration
    'loss': 'log',            # Binary classification uses log loss
    'ycf_result': 0,          # Jobs does not have counterfactual ground truth
    'batch_norm': 1,          # Use batch normalization
    'autoWeighting': 1,       # Must set to 0 when batch_size>0
    'constrainedLayer': 2,    # Constrained layers
    
    # Other configuration
    'seed': 1,
    'optimizer': 'Adam',
    'imb_fun': 'mmd_lin',
    'output_delay': 100,
    'pred_output_delay': 30,
}

# Python interpreter path
PYTHON_EXE = r'C:\tool\Anaconda3\envs\DeR_CFR_tf1\python.exe'

def run_jobs():
    """Run Jobs experiment"""
    
    print("=" * 80)
    print("Running DeR-CFR Jobs Dataset Experiment")
    print("=" * 80)
    print(f"\nPython Interpreter: {PYTHON_EXE}")
    print(f"Dataset: {JOBS_CONFIG['dataform']}")
    print(f"Experiments: {JOBS_CONFIG['experiments']}")
    print(f"Iterations: {JOBS_CONFIG['iterations']}")
    print(f"Batch Size: {JOBS_CONFIG['batch_size']}")
    print("\nHyperparameter Configuration:")
    print(f"  alpha (p_coef_alpha):  {JOBS_CONFIG['p_coef_alpha']}")
    print(f"  beta  (p_coef_beta):   {JOBS_CONFIG['p_coef_beta']}")
    print(f"  gamma (p_coef_gamma):  {JOBS_CONFIG['p_coef_gamma']}")
    print(f"  mu    (p_coef_mu):     {JOBS_CONFIG['p_coef_mu']}")
    print(f"  lambda(p_coef_lambda): {JOBS_CONFIG['p_coef_lambda']}")
    print("\nJobs Special Settings:")
    print(f"  Loss Function: log (binary classification)")
    print(f"  Counterfactual Truth: No (ycf_result=0)")
    print("=" * 80)
    print()
    
    # Build command line arguments
    cmd = [PYTHON_EXE, 'main.py']
    
    for key, value in JOBS_CONFIG.items():
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
        print("[SUCCESS] Jobs experiment completed!")
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
    exit_code = run_jobs()
    sys.exit(exit_code)
