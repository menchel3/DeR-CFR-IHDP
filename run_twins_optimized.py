"""
TWINS Dataset Experiment Runner - 优化版

基于成功的 aggressive 配置，进行微调：
1. 减少迭代次数到 250（防止过拟合）
2. 保持成功的损失权重配置
3. 增加早停监控
"""
import subprocess
import sys
import os

# TWINS dataset configuration - OPTIMIZED
TWINS_CONFIG = {
    # Data configuration
    'datadir': 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/',
    'dataform': 'twins_1-10.train.npz',
    'data_test': 'twins_1-10.test.npz',
    'outdir': 'results/twins_optimized/',
    
    # Training configuration - 基于观察调整
    'experiments': 10,         
    'iterations': 250,         # ↓ 减少到250（观察到150-200最佳）
    'batch_size': 0,           # Full batch
    'lrate': 1e-2,             # 保持较大学习率
    'lrate_decay': 0.995,      # 保持慢衰减
    'val_part': 0.3,           
    
    # Network architecture - 保持成功配置
    'n_in': 5,                 
    'n_out': 5,                
    'n_t': 3,                  
    'dim_in': 100,             
    'dim_out': 100,            
    
    # Loss function weights - 保持成功的激进配置
    'p_coef_y': 50.0,          # 强outcome loss
    'p_coef_alpha': 1e-2,      
    'p_coef_beta': 1e-3,       
    'p_coef_gamma': 1e-4,      # 极弱balance loss
    'p_coef_mu': 0.1,          # 弱正则化
    'p_coef_lambda': 0.1,      
    
    # TWINS specific
    'loss': 'log',             
    'ycf_result': 1,           
    'batch_norm': 1,           
    'autoWeighting': 1,        
    'constrainedLayer': 0,     
    
    # Other
    'seed': 1,
    'optimizer': 'Adam',
    'imb_fun': 'mmd_lin',
    'output_delay': 50,
    'pred_output_delay': 20,
}

PYTHON_EXE = r'C:\tool\Anaconda3\envs\DeR_CFR_tf1\python.exe'

def run_twins_optimized():
    """Run TWINS experiment with optimized configuration"""
    
    print("=" * 80)
    print("🎯 TWINS Dataset Experiment - OPTIMIZED VERSION")
    print("=" * 80)
    print("\n✅ Success from Previous Run:")
    print("  ATE_pred: -0.045 ± 0.020")
    print("  True ATE: -0.028")
    print("  Bias:     -0.017 (should use |bias|)")
    print("\n🔧 Optimizations:")
    print("  1. Iterations: 500 → 250 (prevent overfitting)")
    print("  2. Keep successful loss weights")
    print("  3. Observed best point at iter 150-200")
    print("\n📊 Key Configuration:")
    print("=" * 80)
    print(f"  Iterations:        {TWINS_CONFIG['iterations']}")
    print(f"  Learning rate:     {TWINS_CONFIG['lrate']}")
    print(f"  Outcome loss (y):  {TWINS_CONFIG['p_coef_y']}")
    print(f"  Balance loss (γ):  {TWINS_CONFIG['p_coef_gamma']}")
    print(f"  Regularization:    {TWINS_CONFIG['p_coef_mu']}")
    print("=" * 80)
    print()
    
    # Build command
    cmd = [PYTHON_EXE, 'main.py']
    for key, value in TWINS_CONFIG.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key}={value}')
    
    # Run
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(cmd, check=True, cwd=script_dir)
        print("\n" + "=" * 80)
        print("✅ [SUCCESS] TWINS optimized experiment completed!")
        print("=" * 80)
        return result.returncode
    except Exception as e:
        print(f"\n❌ [ERROR] {e}")
        return 1

if __name__ == '__main__':
    sys.exit(run_twins_optimized())
