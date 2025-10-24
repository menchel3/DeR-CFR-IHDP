import sys
import os
import json
import numpy as np

import pickle

from cfr_function.logger import Logger as Log
Log.VERBOSE = True

import cfr_function.evaluation as evaluation
from cfr_function.plotting import plot_evaluation_cont, plot_evaluation_bin

def sort_by_config(results, configs, key):
    vals = np.array([cfg[key] for cfg in configs])
    I_vals = np.argsort(vals)

    for k in results['train'].keys():
        results['train'][k] = results['train'][k][I_vals,]
        results['valid'][k] = results['valid'][k][I_vals,]

        if k in results['test']:
            results['test'][k] = results['test'][k][I_vals,]

    configs_sorted = []
    for i in I_vals:
        configs_sorted.append(configs[i])

    return results, configs_sorted

def load_config(config_file):
    with open(config_file, 'r') as f:
        cfg = [l.split('=') for l in f.read().split('\n') if '=' in l]
        cfg = dict([(kv[0], eval(kv[1])) for kv in cfg])
    return cfg

def evaluate(output_dir='results/example_ihdp/', overwrite=True, filters=None, ground_truth=True, mode = 'ATE', bin_or_cont = 1):

    if not os.path.isdir(output_dir):
        raise Exception('Could not find output at path: %s' % output_dir)

    # Auto-select dataset based on output_dir
    if 'jobs' in output_dir.lower():
        data_train = 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/data/jobs_DW_bin.new.10.train.npz'
        data_test = 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/data/jobs_DW_bin.new.10.test.npz'
        binary = True
    elif 'twins' in output_dir.lower():
        data_train = 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/twins_1-10.train.npz'
        data_test = 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/twins_1-10.test.npz'
        binary = False
    else:
        data_train = 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/data/ihdp_npci_1-100.train.npz'
        data_test = 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/data/ihdp_npci_1-100.test.npz'
        binary = False

    # Evaluate results
    eval_path = '%s/evaluation.npz' % output_dir
    if overwrite or (not os.path.isfile(eval_path)):
        eval_results, configs = evaluation.evaluate(output_dir,
                                data_path_train=data_train,
                                data_path_test=data_test,
                                binary=binary,mode=mode, bin_or_cont=bin_or_cont)
        # Save evaluation
        pickle.dump((eval_results, configs), open(eval_path, "wb"))
    else:
        if Log.VERBOSE:
            print ('Loading evaluation results from %s...' % eval_path)
        # Load evaluation
        eval_results, configs = pickle.load(open(eval_path, "rb"))

    # Call correct plotting function based on dataset type
    if binary:
        # Jobs dataset: display policy_risk, bias_att, err_fact
        plot_evaluation_bin(eval_results, configs, output_dir, data_train, data_test, filters)
    else:
        # IHDP/TWINS dataset: display pehe, bias_ate, etc.
        plot_evaluation_cont(eval_results, configs, output_dir, data_train, data_test, filters)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        with open('configs/run.json','r') as f:
            run_dict = json.load(f)
        evaluate('configs/' + run_dict['config'], overwrite = False, filters = None)
        print ('Usage: python evaluate.py <config_file> <overwrite (default 0)> <filters (optional)>')
    else:
        config_file = sys.argv[1]

        overwrite = False
        if len(sys.argv)>2 and sys.argv[2] == '1':
            overwrite = True

        filters = None
        if len(sys.argv)>3:
            filters = eval(sys.argv[3])

        evaluate(config_file, overwrite, filters=filters)
