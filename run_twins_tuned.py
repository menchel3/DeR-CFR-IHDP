
import subprocess
import sys
import os

# TWINS dataset configuration - FINE-TUNED
TWINS_CONFIG = {
    # Data configuration
    'datadir': 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/',
    'dataform': 'twins_1-10.train.npz',
    'data_test': 'twins_1-10.test.npz',
    'outdir': 'results/twins_tuned16/',
    
    # Training configuration - MORE ITERATIONS
    'experiments': 10,
    'iterations': 500,
    'batch_size': 0,
    'lrate': 1e-3,
    'lrate_decay': 0.995,
    'val_part': 0.3,
    
    # Network architecture - SMALLER (proven effective)
    'n_in': 3,
    'n_out': 3,
    'n_t': 2,
    'dim_in': 64,
    'dim_out': 64,
    
    # Loss function weights - OPTIMAL (unchanged)
    'p_coef_y': 50.0,
    'p_coef_alpha': 1e-2,
    'p_coef_beta': 1e-3,
    'p_coef_gamma': 1e-4,
    'p_coef_mu': 0.1,
    'p_coef_lambda': 0.1,
    
    # TWINS specific configuration
    'loss': 'log',
    'ycf_result': 1,
    'batch_norm': 1,
    'autoWeighting': 1,
    'constrainedLayer': 0,
    
    # Other configuration
    'seed': 1,
    'optimizer': 'Adam',
    'imb_fun': 'mmd_lin',
    'output_delay': 50,
    'pred_output_delay': 10,
}

# Python interpreter path
PYTHON_EXE = r'C:\tool\Anaconda3\envs\DeR_CFR_tf1\python.exe'

def run_twins_tuned():
    """Run TWINS experiment with fine-tuned configuration"""
    
    print("=" * 80)
    print("Running DeR-CFR TWINS Dataset Experiment - TRAINING PARAMS TUNING v4")
    print("=" * 80)
    print(f"\nPrevious Results:")
    print(f"  Optimal (5-5-3, dim=100):               Bias_ATE = 0.017")
    print(f"  v1 (4-4-2, dim=80, original train):     Bias_ATE = 0.019")
    print(f"  v2 (4-4-2, dim=80, tuned loss):         Bias_ATE = 0.030")
    print(f"  v3 (4-4-2, dim=80, faster decay):       Bias_ATE = 0.021 ❌")
    print("\n" + "=" * 80)
    print("Strategy: MORE ITERATIONS instead of training param adjustments")
    print("=" * 80)
    print("  Network: 4-4-2, dim=80")
    print("  Loss weights: OPTIMAL (p_y=50, gamma=1e-4, mu=0.1)")
    print("  Training strategy:")
    print("    - iterations: 300 → 500 (more training)")
    print("    - Restore original: decay=0.995, val_part=0.3")
    print("\n  Rationale: Small network needs more iterations, not faster decay")
    print("=" * 80)
    print("Key Parameters:")
    print("=" * 80)
    print(f"  Network: 4-4-2, dim=80")
    print(f"  Loss: OPTIMAL (p_y=50, γ=1e-4, μ=0.1)")
    print(f"  Training (MORE ITERATIONS):")
    print(f"    lrate:      {TWINS_CONFIG['lrate']}")
    print(f"    decay:      {TWINS_CONFIG['lrate_decay']} (restored)")
    print(f"    iterations: {TWINS_CONFIG['iterations']} (increased)")
    print(f"    val_part:   {TWINS_CONFIG['val_part']} (restored)")
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(cmd, check=True, cwd=script_dir)
        print("\n" + "=" * 80)
        print("[SUCCESS] TWINS tuned v4 experiment completed!")
        print("=" * 80)
        print("\n📊 Next steps:")
        print("  1. Check if bias < 0.019 (v1 baseline)")
        print("  2. Verify if more iterations help small network")
        print("  3. Compare: v1=0.019 vs v4=?")
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
    exit_code = run_twins_tuned()
    sys.exit(exit_code)
