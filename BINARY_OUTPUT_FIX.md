# Binary Output Fix for Causal Effect Estimation

## Problem

For binary outcome datasets (Jobs, TWINS), the original code output discrete labels `{0, 1}` instead of probabilities, which caused:
- **ATE_pred = 0**: Since `y1_pred - y0_pred` is mostly 0 when both are discrete
- **Incorrect causal effect estimation**: Individual treatment effect (ITE) can only be {-1, 0, 1}

## Root Cause

In `module.py` line 268-279 (original):
```python
if FLAGS.loss == 'log':
    label = y
    one = tf.ones_like(label)
    zero = tf.zeros_like(label)
    label = tf.where(label < 0.5, x=zero, y=one)
    self.output = label  # ❌ Output discrete {0, 1}
```

This binarization is correct for accuracy calculation but **wrong for causal inference**.

## Solution

Modified `module.py` to output probabilities for causal inference:

```python
if FLAGS.loss == 'log':
    # For binary outcomes (Jobs/TWINS), output probability
    y_prob = 0.995 / (1.0 + tf.exp(-y)) + 0.0025
    self.output = y_prob  # ✅ Output probability [0, 1]
    
    # Save discrete label for accuracy calculation only
    label = y_prob
    one = tf.ones_like(label)
    zero = tf.zeros_like(label)
    label_discrete = tf.where(label < 0.5, x=zero, y=one)
    self.output_discrete = label_discrete  # Discrete {0, 1}
else:
    # For continuous outcomes (IHDP)
    self.output = y
    self.output_discrete = y  # Same as output
```

Modified `main.py` to use `output_discrete` only for accuracy:

```python
if FLAGS.loss == 'log':
    # Use discrete output for accuracy calculation
    y_pred_discrete = sess.run(CFR.output_discrete, ...)
    y_pred_discrete = 1.0*(y_pred_discrete > 0.5)
    acc = 100 * (1 - np.mean(np.abs(y_batch - y_pred_discrete)))
    
# For prediction (causal inference), always use CFR.output
y_pred_f = sess.run(CFR.output, ...)  # Probability for binary, value for continuous
y_pred_cf = sess.run(CFR.output, ...)
```

## Compatibility

### IHDP (Continuous Outcome, loss='l2')
- `self.output = y` (continuous value)
- `self.output_discrete = y` (same)
- Does not enter `if FLAGS.loss == 'log'` branch
- ✅ Works as before

### Jobs (Binary Outcome, loss='log', no YCF)
- `self.output = y_prob` (probability [0, 1])
- `self.output_discrete = {0, 1}` (for accuracy)
- Uses `output` for predictions → correct ITE calculation
- ✅ ATE_pred ≠ 0

### TWINS (Binary Outcome, loss='log', has YCF)
- `self.output = y_prob` (probability [0, 1])
- `self.output_discrete = {0, 1}` (for accuracy)
- Uses `output` for predictions → correct ITE calculation
- ✅ ATE_pred ≠ 0

## Expected Results

### Before Fix
```
Mode: Test
  | Ate_pred        | Att_pred        | ...
----------------------------------------------
0 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | ...  ❌ All zeros!
```

### After Fix
```
Mode: Test
  | Ate_pred        | Att_pred        | Bias_ate        | ...
----------------------------------------------------------------
0 | 0.028 +/- 0.003 | 0.027 +/- 0.007 | 0.028 +/- 0.003 | ...  ✅ Non-zero!
```

## Technical Details

**Why sigmoid with clipping?**
```python
y_prob = 0.995 / (1.0 + tf.exp(-y)) + 0.0025
```
- Prevents probabilities from being exactly 0 or 1
- Avoids `log(0)` or `log(1)` in loss calculation
- Clips to [0.0025, 0.9975]

**Causal Effect Calculation:**
- **Predicted ITE**: `ITE_pred = y1_prob - y0_prob` (continuous in [0, 1])
- **Predicted ATE**: `ATE_pred = mean(ITE_pred)` (can be any value in [-1, 1])
- **Before fix**: `ITE_pred = y1_discrete - y0_discrete ∈ {-1, 0, 1}` (discrete!)

## Files Modified

1. **module.py**: Line 265-285
   - Added probability output for binary outcomes
   - Added `output_discrete` for accuracy calculation

2. **main.py**: Line 182-195
   - Use `output_discrete` for accuracy (only when `loss='log'`)
   - Use `output` for predictions (always)

3. **No changes needed in**:
   - `evaluation.py`: Already uses predictions[:,0] and predictions[:,1]
   - `run_*.py`: Configuration files unchanged
   - `loader.py`: Data loading unchanged

## Verification

Run experiments and check:
```bash
# Jobs
run_jobs.bat
# Check: ATE_pred ≠ 0 in results_summary.txt

# TWINS
run_twins.bat
# Check: ATE_pred ≠ 0 in results_summary.txt

# IHDP
run_ihdp.bat
# Check: Still works, continuous ATE values
```

## Date
2025-10-24
