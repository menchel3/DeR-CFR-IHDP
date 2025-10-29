"""
诊断 TWINS 模型为什么学不到负的 ATE

可能原因：
1. 预测总是接近平均值
2. 治疗效应被平滑掉了
3. 个体效应方差太小
"""
import numpy as np
import os

# Load results
result_dir = 'results/example_twins/results_20251028_112955-897741'
result_file = os.path.join(result_dir, 'result.npz')

if os.path.exists(result_file):
    print("="*80)
    print("Loading model predictions...")
    print("="*80)
    
    result = np.load(result_file)
    print("Available keys:", list(result.keys()))
    
    # Load predictions
    # Shape: [n_units, 2, n_rep, n_outputs]
    # predictions[:,0,:,:] = factual predictions
    # predictions[:,1,:,:] = counterfactual predictions
    predictions = result['pred']
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"  - Number of units: {predictions.shape[0]}")
    print(f"  - [factual, counterfactual]: {predictions.shape[1]}")
    print(f"  - Number of repetitions: {predictions.shape[2]}")
    print(f"  - Number of outputs: {predictions.shape[3]}")
    
    # Load original data
    data_path = 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/twins_1-10.train.npz'
    data = np.load(data_path)
    
    # Use first experiment, first repetition, first output
    i_exp = 0
    i_rep = 0
    i_out = 0
    
    t = data['t'][:,i_exp]
    yf = data['yf'][:,i_exp]
    ycf = data['ycf'][:,i_exp]
    
    yf_p = predictions[:,0,i_rep,i_out]  # Factual prediction
    ycf_p = predictions[:,1,i_rep,i_out]  # Counterfactual prediction
    
    print("\n" + "="*80)
    print("Prediction Statistics")
    print("="*80)
    
    print("\nFactual predictions (yf_p):")
    print(f"  Mean: {np.mean(yf_p):.4f}")
    print(f"  Std:  {np.std(yf_p):.4f}")
    print(f"  Min:  {np.min(yf_p):.4f}")
    print(f"  Max:  {np.max(yf_p):.4f}")
    print(f"  Unique values: {len(np.unique(yf_p))}")
    
    print("\nCounterfactual predictions (ycf_p):")
    print(f"  Mean: {np.mean(ycf_p):.4f}")
    print(f"  Std:  {np.std(ycf_p):.4f}")
    print(f"  Min:  {np.min(ycf_p):.4f}")
    print(f"  Max:  {np.max(ycf_p):.4f}")
    print(f"  Unique values: {len(np.unique(ycf_p))}")
    
    print("\n" + "="*80)
    print("Individual Treatment Effect (ITE) Analysis")
    print("="*80)
    
    # Calculate predicted ITE
    ite_pred = np.zeros_like(yf_p)
    ite_pred[t < 1] = ycf_p[t < 1] - yf_p[t < 1]  # Control: Y(1) - Y(0)
    ite_pred[t > 0] = yf_p[t > 0] - ycf_p[t > 0]  # Treated: Y(1) - Y(0)
    
    # Calculate true ITE
    ite_true = np.zeros_like(yf)
    ite_true[t < 1] = ycf[t < 1] - yf[t < 1]
    ite_true[t > 0] = yf[t > 0] - ycf[t > 0]
    
    print("\nPredicted ITE:")
    print(f"  Mean: {np.mean(ite_pred):.6f}  <- This is ATE prediction")
    print(f"  Std:  {np.std(ite_pred):.6f}")
    print(f"  Min:  {np.min(ite_pred):.6f}")
    print(f"  Max:  {np.max(ite_pred):.6f}")
    print(f"  Median: {np.median(ite_pred):.6f}")
    print(f"  % negative: {100*np.mean(ite_pred < 0):.1f}%")
    print(f"  % zero: {100*np.mean(np.abs(ite_pred) < 0.001):.1f}%")
    print(f"  % positive: {100*np.mean(ite_pred > 0):.1f}%")
    
    print("\nTrue ITE:")
    print(f"  Mean: {np.mean(ite_true):.6f}  <- This is true ATE")
    print(f"  Std:  {np.std(ite_true):.6f}")
    print(f"  Min:  {np.min(ite_true):.6f}")
    print(f"  Max:  {np.max(ite_true):.6f}")
    print(f"  Median: {np.median(ite_true):.6f}")
    print(f"  % negative: {100*np.mean(ite_true < 0):.1f}%")
    print(f"  % positive: {100*np.mean(ite_true > 0):.1f}%")
    
    print("\n" + "="*80)
    print("Problem Diagnosis")
    print("="*80)
    
    # Check if predictions are too similar
    diff_factual_cf = np.abs(yf_p - ycf_p)
    print(f"\nDifference between factual and counterfactual predictions:")
    print(f"  Mean |yf_p - ycf_p|: {np.mean(diff_factual_cf):.6f}")
    print(f"  Median |yf_p - ycf_p|: {np.median(diff_factual_cf):.6f}")
    print(f"  Max |yf_p - ycf_p|: {np.max(diff_factual_cf):.6f}")
    
    if np.mean(diff_factual_cf) < 0.01:
        print("\n  ⚠️  WARNING: Factual and counterfactual predictions are too similar!")
        print("      The model is not learning treatment effects.")
    
    # Check if predictions are close to observed mean
    overall_mean = np.mean(yf)
    print(f"\nOverall observed outcome mean: {overall_mean:.4f}")
    print(f"Distance of yf_p mean from overall mean: {abs(np.mean(yf_p) - overall_mean):.4f}")
    print(f"Distance of ycf_p mean from overall mean: {abs(np.mean(ycf_p) - overall_mean):.4f}")
    
    if abs(np.mean(yf_p) - overall_mean) < 0.01 and abs(np.mean(ycf_p) - overall_mean) < 0.01:
        print("\n  ⚠️  WARNING: Both predictions are very close to the overall mean!")
        print("      The model is just predicting the average.")
    
    # Check heterogeneity
    print(f"\n" + "="*80)
    print("Heterogeneity Analysis")
    print("="*80)
    print(f"Predicted ITE variance: {np.var(ite_pred):.6f}")
    print(f"True ITE variance: {np.var(ite_true):.6f}")
    
    if np.var(ite_pred) < 0.001:
        print("\n  ⚠️  WARNING: Predicted ITE has very low variance!")
        print("      The model is predicting similar effects for everyone.")
    
    # Separate by treatment group
    print(f"\n" + "="*80)
    print("By Treatment Group")
    print("="*80)
    
    print("\nControl group (t=0):")
    print(f"  Predicted ATE: {np.mean(ite_pred[t<1]):.6f}")
    print(f"  True ATE: {np.mean(ite_true[t<1]):.6f}")
    print(f"  Mean yf_p: {np.mean(yf_p[t<1]):.4f}")
    print(f"  Mean ycf_p: {np.mean(ycf_p[t<1]):.4f}")
    print(f"  Mean |yf_p - ycf_p|: {np.mean(np.abs(yf_p[t<1] - ycf_p[t<1])):.6f}")
    
    print("\nTreated group (t=1):")
    print(f"  Predicted ATT: {np.mean(ite_pred[t>0]):.6f}")
    print(f"  True ATT: {np.mean(ite_true[t>0]):.6f}")
    print(f"  Mean yf_p: {np.mean(yf_p[t>0]):.4f}")
    print(f"  Mean ycf_p: {np.mean(ycf_p[t>0]):.4f}")
    print(f"  Mean |yf_p - ycf_p|: {np.mean(np.abs(yf_p[t>0] - ycf_p[t>0])):.6f}")
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"True ATE: {np.mean(ite_true):.6f} (negative = treatment reduces mortality)")
    print(f"Predicted ATE: {np.mean(ite_pred):.6f}")
    print(f"Bias: {np.mean(ite_pred) - np.mean(ite_true):.6f}")
    print(f"PEHE: {np.sqrt(np.mean((ite_pred - ite_true)**2)):.6f}")
    
else:
    print(f"Result file not found: {result_file}")
    print("\nAvailable files in results directory:")
    if os.path.exists(result_dir):
        for f in os.listdir(result_dir):
            print(f"  - {f}")
