"""
运行IHDP数据集的DeR-CFR实验
使用指定的Python解释器和超参数配置
"""
import subprocess
import sys
import os

# IHDP数据集的超参数配置
IHDP_CONFIG = {
    # 数据配置
    'datadir': 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/data/',
    'dataform': 'ihdp_npci_1-100.train.npz',
    'data_test': 'ihdp_npci_1-100.test.npz',
    'outdir': 'results/example_ihdp/',
    
    # 训练配置
    'experiments': 100,        # IHDP有100个实验
    'iterations': 300,         # 训练迭代次数
    'batch_size': 0,          # 0表示全批次训练(IHDP样本较少)
    'lrate': 1e-3,            # 初始学习率
    'lrate_decay': 0.97,      # 学习率衰减
    'val_part': 0.3,          # 验证集比例
    
    # 网络架构
    'n_in': 7,                # 表征层数
    'n_out': 4,               # 输出层数
    'n_t': 1,                 # 处理层数
    'dim_in': 32,             # 表征维度
    'dim_out': 256,           # 输出维度
    
    # 损失函数权重 (IHDP优化配置)
    'p_coef_y': 1.0,          # 结果回归损失
    'p_coef_alpha': 5,     # α: 调整分解损失
    'p_coef_beta': 10,         # β: 工具变量分解损失
    'p_coef_gamma': 1e-2,     # γ: 平衡损失
    'p_coef_mu': 10,           # μ: 正交正则化
    'p_coef_lambda': 1e-2,    # λ: L2正则化
    
    # IHDP特定配置
    'loss': 'l2',             # 连续结果使用L2损失
    'ycf_result': 1,          # IHDP有反事实真值
    'batch_norm': 0,          # IHDP不使用batch normalization
    'constrainedLayer': 2,    # 约束层数
    
    # 其他配置
    'seed': 1,
    'optimizer': 'Adam',
    'imb_fun': 'mmd_lin',
    'output_delay': 100,
    'pred_output_delay': 30,
}

# Python解释器路径
PYTHON_EXE = r'C:\tool\Anaconda3\envs\DeR_CFR_tf1\python.exe'

def run_ihdp():
    """Run IHDP experiment"""
    
    print("=" * 80)
    print("Running DeR-CFR IHDP Dataset Experiment")
    print("=" * 80)
    print(f"\nPython Interpreter: {PYTHON_EXE}")
    print(f"Dataset: {IHDP_CONFIG['dataform']}")
    print(f"Experiments: {IHDP_CONFIG['experiments']}")
    print(f"Iterations: {IHDP_CONFIG['iterations']}")
    print(f"Batch Size: {IHDP_CONFIG['batch_size']} (0=full batch)")
    print("\nHyperparameter Configuration:")
    print(f"  alpha (p_coef_alpha):  {IHDP_CONFIG['p_coef_alpha']}")
    print(f"  beta  (p_coef_beta):   {IHDP_CONFIG['p_coef_beta']}")
    print(f"  gamma (p_coef_gamma):  {IHDP_CONFIG['p_coef_gamma']}")
    print(f"  mu    (p_coef_mu):     {IHDP_CONFIG['p_coef_mu']}")
    print(f"  lambda(p_coef_lambda): {IHDP_CONFIG['p_coef_lambda']}")
    print("=" * 80)
    print()
    
    # Build command line arguments
    cmd = [PYTHON_EXE, 'main.py']
    
    for key, value in IHDP_CONFIG.items():
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
        print("[SUCCESS] IHDP experiment completed!")
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
    exit_code = run_ihdp()
    sys.exit(exit_code)
