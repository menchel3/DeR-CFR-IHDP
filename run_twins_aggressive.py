"""
TWINS Dataset Experiment Runner for DeR-CFR - Ultra Aggressive版

目标：让模型预测出负的ATE效应，bias < 0.01

策略升级（基于validation PEHE自动筛选的理解）：
1. PRIMARY参数调优（决定最佳点质量）:
   - p_coef_y: 50→58 (增强outcome loss，提高预测准确性)
   - p_coef_gamma: 1e-4→5e-5 (进一步削弱balance，保留效应信号)
2. 正则化微调: 0.1→0.08 (允许更强学习)
3. 其他参数保持成功配置

Previous Results:
  - Aggressive (p_coef_y=50): bias=0.017 ✓
  - Final (p_coef_y=35): bias=0.171 ✗ (太弱导致过拟合)
  - Target: bias < 0.01
"""
import subprocess
import sys
import os

# TWINS dataset configuration - AGGRESSIVE
TWINS_CONFIG = {
    # Data configuration
    'datadir': 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/',
    'dataform': 'twins_1-10.train.npz',
    'data_test': 'twins_1-10.test.npz',
    'outdir': 'results/twins_aggressive/',
    
    # Training configuration
    'experiments': 10,         
    'iterations': 300,        # 适中的迭代次数（你说增加次数用处不大）
    'batch_size': 0,           # Full batch
    'lrate': 1e-2,             # ↑↑ 更大的学习率 (从5e-3增加到1e-2)
    'lrate_decay': 0.995,      # 更慢的衰减
    'val_part': 0.3,           
    
    # Network architecture
    'n_in': 5,                 # ↓ 减少层数，避免过拟合
    'n_out': 5,                
    'n_t': 3,                  
    'dim_in': 100,             # ↑ 增加网络容量
    'dim_out': 100,            
    
    # Loss function weights - ULTRA AGGRESSIVE (目标bias<0.01)
    'p_coef_y': 58.0,          # ↑↑↑ 继续增加 (50→58, 争取更准确的预测)
    'p_coef_alpha': 1e-2,      # α: 保持不变
    'p_coef_beta': 1e-3,       # β: 保持不变
    'p_coef_gamma': 5e-5,      # ↓↓ 进一步削弱balance (1e-4→5e-5, 保留更多效应信号)
    'p_coef_mu': 0.08,         # ↓↓ 略微减小正则化 (0.1→0.08, 允许更强学习)
    'p_coef_lambda': 0.08,     # ↓↓ 略微减小L2正则化 (0.1→0.08)
    
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
    'output_delay': 50,        # 更频繁的输出
    'pred_output_delay': 20,
}

# Python interpreter path
PYTHON_EXE = r'C:\tool\Anaconda3\envs\DeR_CFR_tf1\python.exe'

def run_twins_aggressive():
    """Run TWINS experiment with ultra aggressive hyperparameters"""
    
    print("=" * 80)
    print("Running DeR-CFR TWINS Dataset Experiment - ULTRA AGGRESSIVE VERSION")
    print("=" * 80)
    print(f"\nGoal: bias < 0.01 (True ATE ≈ -0.028)")
    print(f"\nPrevious Results:")
    print(f"  Aggressive (p_coef_y=50, γ=1e-4):  bias=0.017 ✓")
    print(f"  Final (p_coef_y=35, γ=8e-5):       bias=0.171 ✗")
    print(f"  Target:                             bias<0.01 ✓✓")
    print("\n" + "=" * 80)
    print("ULTRA AGGRESSIVE Strategy:")
    print("=" * 80)
    print("  1. Outcome loss: 50.0 → 58.0 (16% increase)")
    print("  2. Balance loss: 1e-4 → 5e-5 (50% decrease)")
    print("  3. Regularization: 0.1 → 0.08 (20% decrease)")
    print("\nRationale (基于validation PEHE自动筛选机制):")
    print("  - 增强p_coef_y: 让最佳迭代点的预测更准确")
    print("  - 削弱balance: 保留更多负效应信号")
    print("  - 降低正则: 允许模型学习更强效应")
    print("  - 这些PRIMARY参数决定最佳点的质量!")
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
        print("[SUCCESS] TWINS ULTRA AGGRESSIVE experiment completed!")
        print("=" * 80)
        print("\n📊 Next steps:")
        print("  1. Check if bias < 0.01")
        print("  2. Compare with previous: bias=0.017")
        print("  3. Verify negative ATE predictions")
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
    exit_code = run_twins_aggressive()
    sys.exit(exit_code)
