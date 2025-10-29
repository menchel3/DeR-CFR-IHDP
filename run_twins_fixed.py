"""
TWINS Dataset Experiment Runner for DeR-CFR - 修复版

根据诊断结果，模型输出全是0.5（sigmoid(0)），说明：
1. 学习率可能太小
2. 正则化太强
3. 需要调整超参数让模型能够学习

修复策略：
1. 增大学习率
2. 减小正则化权重
3. 增加迭代次数
4. 调整损失函数权重
"""
import subprocess
import sys
import os

# TWINS dataset configuration - FIXED
TWINS_CONFIG = {
    # Data configuration
    'datadir': 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/',
    'dataform': 'twins_1-10.train.npz',
    'data_test': 'twins_1-10.test.npz',
    'outdir': 'results/twins_fixed/',
    
    # Training configuration
    'experiments': 10,         # TWINS has 10 experiments
    'iterations': 300,        # ↑ 增加迭代次数 (原来300)
    'batch_size': 0,           # 0 = full batch
    'lrate': 5e-3,             # ↑ 增大学习率 (原来1e-3)
    'lrate_decay': 0.99,       # ↑ 降低衰减速度 (原来0.97)
    'val_part': 0.3,           # Validation set ratio
    
    # Network architecture
    'n_in': 7,                 # Representation layers
    'n_out': 7,                # Output layers
    'n_t': 3,                  # Treatment layers
    'dim_in': 64,              # Representation dimension
    'dim_out': 64,             # Output dimension
    
    # Loss function weights - ADJUSTED
    'p_coef_y': 10.0,          # ↑ 增大outcome loss (原来1.0)
    'p_coef_alpha': 1e-2,      # α: Adjustment decomposition loss
    'p_coef_beta': 1e-3,       # β: Instrumental variable decomposition loss
    'p_coef_gamma': 1e-2,      # ↑ 增大balance loss (原来1e-3)
    'p_coef_mu': 1.0,          # ↓ 减小正则化 (原来5.0)
    'p_coef_lambda': 1.0,      # ↓ 减小L2正则化 (原来5.0)
    
    # TWINS specific configuration
    'loss': 'log',             # Binary classification uses log loss
    'ycf_result': 1,           # TWINS has counterfactual ground truth
    'batch_norm': 1,           # Use batch normalization
    'autoWeighting': 1,        # ✓ 保持auto-weighting开启
    'constrainedLayer': 0,     # 0 = all layers
    
    # Other configuration
    'seed': 1,
    'optimizer': 'Adam',
    'imb_fun': 'mmd_lin',
    'output_delay': 100,
    'pred_output_delay': 30,
}

# Python interpreter path
PYTHON_EXE = r'C:\tool\Anaconda3\envs\DeR_CFR_tf1\python.exe'

def run_twins_fixed():
    """Run TWINS experiment with fixed hyperparameters"""
    
    print("=" * 80)
    print("Running DeR-CFR TWINS Dataset Experiment - FIXED VERSION")
    print("=" * 80)
    print(f"\nPython Interpreter: {PYTHON_EXE}")
    print(f"Dataset: {TWINS_CONFIG['dataform']}")
    print(f"Experiments: {TWINS_CONFIG['experiments']}")
    print(f"Iterations: {TWINS_CONFIG['iterations']}")
    print("\nKey Changes from Original:")
    print("  1. Learning rate: 1e-3 → 5e-3 (5x increase)")
    print("  2. Learning rate decay: 0.97 → 0.99 (slower decay)")
    print("  3. Iterations: 300 → 3000 (10x increase)")
    print("  4. Outcome loss weight: 1.0 → 10.0 (10x increase)")
    print("  5. Regularization: 5.0 → 1.0 (5x decrease)")
    print("\nHyperparameter Configuration:")
    print(f"  alpha (p_coef_alpha):  {TWINS_CONFIG['p_coef_alpha']}")
    print(f"  beta  (p_coef_beta):   {TWINS_CONFIG['p_coef_beta']}")
    print(f"  gamma (p_coef_gamma):  {TWINS_CONFIG['p_coef_gamma']}")
    print(f"  mu    (p_coef_mu):     {TWINS_CONFIG['p_coef_mu']}")
    print(f"  lambda(p_coef_lambda): {TWINS_CONFIG['p_coef_lambda']}")
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
    exit_code = run_twins_fixed()
    sys.exit(exit_code)
