import tensorflow as tf
import numpy as np
import sys, os
import getopt
import random
import datetime
import traceback
from evaluate import evaluate

from module import Net
from utils import simplex_project, log, save_config, load_data, validation_split

''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
# Hyper-parameter in Table 1
tf.app.flags.DEFINE_integer('constrainedLayer', 2, """Number of layers with constraints applied.""")
tf.app.flags.DEFINE_integer('batch_norm', 1, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'divide', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_integer('n_in', 5, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 4, """Number of output layers. """)
tf.app.flags.DEFINE_integer('n_t', 1, """Number of treatment layers. """)
tf.app.flags.DEFINE_integer('dim_in', 32, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 128, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_float('p_coef_y', 1.0, """ Default 1: Outcome Regression - Loss FUN L_R in Eq.(8). """)
tf.app.flags.DEFINE_float('p_coef_alpha', 1e-2, """ Hyper-parameter alpha: Decompose Adjustments - Loss FUN L_A in Eq. (3). """)
tf.app.flags.DEFINE_float('p_coef_beta', 1, """ Hyper-parameter beta: Decompose Instruments - Loss FUN L_I in Eq. (5). """)
tf.app.flags.DEFINE_float('p_coef_gamma', 1e-2, """ Hyper-parameter gamma: Balancing Confounders - Loss FUN L_C_B in Eq. (4). """)
tf.app.flags.DEFINE_float('p_coef_mu', 5, """  Hyper-parameter mu: Deep Orthogonal Regularizer - Loss FUN L_O in Eq. (7). """)
tf.app.flags.DEFINE_float('p_coef_lambda', 1e-3, """ Hyper-parameter lambda: Regularization - Loss FUN Reg in Eq. (14). """)
# Training Configurations
tf.app.flags.DEFINE_integer('seed', 1, """Random Seed. """)
tf.app.flags.DEFINE_integer('experiments', 10, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 300, """Number of iterations. """)
tf.app.flags.DEFINE_integer('batch_size', 128, """Batch size. """)
tf.app.flags.DEFINE_float('lrate', 1e-3, """Learning rate. """)
tf.app.flags.DEFINE_float('dropout_in', 1.0, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 1.0, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'elu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_string('optimizer', 'Adam', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_string('imb_fun', 'mmd_lin', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.app.flags.DEFINE_string('loss', 'log', """Type of loss function to use: 'log' for binary outcomes, 'l1' or 'l2' for continuous outcomes.""")
tf.app.flags.DEFINE_float('val_part', 0.3, """Validation part. """)
tf.app.flags.DEFINE_integer('ycf_result', 1, """The exits of ycf. """)
# DataLoader and Logging Configurations
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', 30, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_string('outdir', 'results/example_jobs/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', 'C:/Users/0702ty/OneDrive/Desktop/DRLECB/data/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'jobs_DW_bin.new.10.train.npz', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', 'jobs_DW_bin.new.10.test.npz', """Test data filename form. """)
tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
# Optional Configurations
tf.app.flags.DEFINE_integer('use_p_correction', 0, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_integer('rep_weight_decay', 0, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_boolean('split_output', 1, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_integer('autoWeighting', 1, """Whether to Use Sample Weights to Auto-balance Confounders. """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
# Default Configurations
tf.app.flags.DEFINE_float('decay', 0.3, """RMSProp decay. """)
tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
tf.app.flags.DEFINE_float('weight_init', 0.1, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.97, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('wass_iterations', 10, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_float('wass_lambda', 10, """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_bpt', 1, """Backprop through T matrix? """)


NUM_ITERATIONS_PER_DECAY = 100

def train(CFR, sess, train_first, train_second, D, I_valid, D_test, logfile, i_exp, outdir):
    """ Trains a CFR model on supplied data """

    ''' Train/validation split '''
    n = D['x'].shape[0]
    I = range(n); I_train = list(set(I)-set(I_valid))
    n_train = len(I_train)
    Im = np.array(range(0,n_train))

    ''' Compute treatment probability'''
    p_treated = np.mean(D['t'][I_train,:])
#使用真实的t，之前只有jobs使用，其他数据集的t都归零了
    ttt = D['t'][I_train,:]
    yff = D['yf'][I_train,:]
    yff_0 = yff[ttt[:,0]<0.5,:]
    yff_1 = yff[ttt[:,0]>0.5,:]
    yff_0_median = np.median(yff_0)
    yff_1_median = np.median(yff_1)

    ''' Set up loss feed_dicts'''
    dict_factual = {CFR.I: Im, CFR.x: D['x'][I_train,:], CFR.t: D['t'][I_train,:], CFR.y_: D['yf'][I_train,:], \
      CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.p_t: p_treated, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median}

    if FLAGS.val_part > 0:
        dict_valid = {CFR.I: Im, CFR.x: D['x'][I_valid,:], CFR.t: D['t'][I_valid,:], CFR.y_: D['yf'][I_valid,:], \
          CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.p_t: p_treated, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median}

    if D['HAVE_TRUTH']:
        dict_cfactual = {CFR.I: Im, CFR.x: D['x'][I_train,:], CFR.t: 1-D['t'][I_train,:], CFR.y_: D['ycf'][I_train,:], \
          CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.compat.v1.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []

    ''' Compute losses '''
    losses = []
    obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],\
      feed_dict=dict_factual)

    cf_error = np.nan
    if D['HAVE_TRUTH']:
        cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

    valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
    if FLAGS.val_part > 0:
        valid_obj, valid_f_error, valid_imb = sess.run([CFR.val_loss, CFR.pred_loss, CFR.imb_dist],\
          feed_dict=dict_valid)

    losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])

    objnan = False

    reps = []
    reps_test = []

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):

        ''' Fetch sample '''
        if FLAGS.batch_size == 0:
            I = random.sample(range(0, n_train), n_train)
            Im = I
        else:
            I = random.sample(range(0, n_train), FLAGS.batch_size)

        x_batch = D['x'][I_train, :][I, :]
        t_batch = D['t'][I_train, :][I]
        y_batch = D['yf'][I_train, :][I]
        if FLAGS.ycf_result == 1:
            yc_batch = D['ycf'][I_train, :][I]

        ''' Do one step of gradient descent '''
        if not objnan:
            # 第1阶段: 优化表征+输出+处理网络
            sess.run(train_first, feed_dict={CFR.I: Im, CFR.x: x_batch, CFR.t: t_batch, \
                                            CFR.y_: y_batch, CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out, \
                                            CFR.p_t: p_treated, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
            #  第2阶段: 优化权重(分解权重)
            sess.run(train_second, feed_dict={CFR.I: Im, CFR.x: x_batch, CFR.t: t_batch, \
                                            CFR.y_: y_batch, CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out, \
                                            CFR.p_t: p_treated, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})

        ''' Project variable selection weights '''
        if FLAGS.varsel:
            wip = simplex_project(sess.run(CFR.weights_in[0]), 1)
            sess.run(CFR.projection, feed_dict={CFR.w_proj: wip, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss,f_error,imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                feed_dict=dict_factual)

            rep = sess.run(CFR.h_rep_norm, feed_dict={CFR.x: D['x'], CFR.do_in: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
            rep_norm = np.mean(np.sqrt(np.sum(np.square(rep), 1)))

            cf_error = np.nan
            if D['HAVE_TRUTH']:
                cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
            if FLAGS.val_part > 0:
                valid_obj, valid_f_error, valid_imb = sess.run([CFR.val_loss, CFR.pred_loss, CFR.imb_dist], feed_dict=dict_valid)

            losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])
            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tImb: %.2g,\tVal: %.3f,\tValImb: %.2g,\tValObj: %.2f' \
                        % (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj)

            if FLAGS.loss == 'log':
                # Use discrete output for accuracy calculation
                y_pred_discrete = sess.run(CFR.output_discrete, feed_dict={CFR.I: Im, CFR.x: x_batch, \
                    CFR.t: t_batch, CFR.y_: y_batch, CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
                y_pred_discrete = 1.0*(y_pred_discrete > 0.5)
                acc = 100 * (1 - np.mean(np.abs(y_batch - y_pred_discrete)))
                loss_str += ',\tAcc: %.2f%%' % acc
                if FLAGS.ycf_result == 1:
                    # Use discrete output for counterfactual accuracy calculation
                    yc_pred_discrete = sess.run(CFR.output_discrete, feed_dict={CFR.I: Im, CFR.x: x_batch, \
                                                              CFR.t: 1 - t_batch, CFR.y_: yc_batch, CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
                    yc_pred_discrete = 1.0 * (yc_pred_discrete > 0.5)
                    cacc = 100 * (1 - np.mean(np.abs(yc_batch - yc_pred_discrete)))
                    loss_str += ',\tcAcc: %.2f%%' % cacc

            log(logfile, loss_str)

            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True
                exit()

        ''' Compute predictions every M iterations '''
        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i==FLAGS.iterations-1:

            y_pred_f = sess.run(CFR.output, feed_dict={CFR.I: Im, CFR.x: D['x'], \
                CFR.t: D['t'], CFR.y_: D['yf'], CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
            if FLAGS.ycf_result == 1:
                y_pred_cf = sess.run(CFR.output, feed_dict={CFR.I: Im, CFR.x: D['x'], \
                    CFR.t: 1-D['t'], CFR.y_: D['ycf'], CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
            else:
                y_pred_cf = sess.run(CFR.output, feed_dict={CFR.I: Im, CFR.x: D['x'], \
                    CFR.t: 1-D['t'], CFR.y_: D['yf'], CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf),axis=1))

            if D_test is not None:
                y_pred_f_test = sess.run(CFR.output, feed_dict={CFR.I: Im, CFR.x: D_test['x'], \
                    CFR.t: D_test['t'], CFR.y_: D_test['yf'], CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
                if FLAGS.ycf_result == 1:
                    y_pred_cf_test = sess.run(CFR.output, feed_dict={CFR.I: Im, CFR.x: D_test['x'], \
                        CFR.t: 1-D_test['t'], CFR.y_: D_test['ycf'], CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
                else:
                    y_pred_cf_test = sess.run(CFR.output, feed_dict={CFR.I: Im, CFR.x: D_test['x'], \
                        CFR.t: 1-D_test['t'], CFR.y_: D_test['yf'], CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test),axis=1))

            if FLAGS.save_rep and i_exp == 1:
                reps_i = sess.run([CFR.h_rep], feed_dict={CFR.I: Im, CFR.x: D['x'], \
                    CFR.do_in: 1.0, CFR.do_out: 0.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
                reps.append(reps_i)

                if D_test is not None:
                    reps_test_i = sess.run([CFR.h_rep], feed_dict={CFR.I: Im, CFR.x: D_test['x'], \
                        CFR.do_in: 1.0, CFR.do_out: 0.0, CFR.y_0_median: yff_0_median, CFR.y_1_median: yff_1_median})
                    reps_test.append(reps_test_i)


    w_I, w_C, w_A, w_out, w_pred = sess.run([CFR.weights_in_I, CFR.weights_in_C, CFR.weights_in_A,
                                             CFR.weights_out, CFR.weights_pred], feed_dict={CFR.x: D['x']})
    if os.path.exists(outdir + 'w/'):
        pass
    else:
        os.makedirs(outdir + 'w/')
    npzfile_w = outdir + 'w/w_' + str(999)
    log(logfile, npzfile_w)
    np.savez(npzfile_w, w_I=w_I, w_C=w_C, w_A=w_A, w_out=w_out, w_pred=w_pred)

    return losses, preds_train, preds_test, reps, reps_test

def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = outdir+'result'
    npzfile_test = outdir+'result.test'
    repfile = outdir+'reps'
    repfile_test = outdir+'reps.test'
    outform = outdir+'y_pred'
    outform_test = outdir+'y_pred.test'
    lossform = outdir+'loss'
    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform

    has_test = False
    if not FLAGS.data_test == '': # if test set supplied
        has_test = True
        dataform_test = FLAGS.datadir + FLAGS.data_test

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir+'config.txt')

    log(logfile, 'Training with hyperparameters: p_coef_y=%.2g, p_coef_alpha=%.2g, p_coef_beta=%.2g, p_coef_gamma=%.2g, p_coef_mu=%.2g, p_coef_lambda=%.2g. ' % (FLAGS.p_coef_y,FLAGS.p_coef_alpha,FLAGS.p_coef_beta,FLAGS.p_coef_gamma,FLAGS.p_coef_mu,FLAGS.p_coef_lambda))

    ''' Load Data '''
    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform % 1
        if has_test:
            datapath_test = dataform_test % 1

    log(logfile,     'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath)
    D_test = None
    if has_test:
        D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d]' % (D['dim']))

    ''' Start Session '''
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['n'], D['dim'], FLAGS.dim_in, FLAGS.dim_out]
    CFR = Net(dims, FLAGS)

    ''' Set up optimizer '''
    first_step = tf.compat.v1.Variable(0, trainable=False, name='first_step')
    second_step = tf.compat.v1.Variable(0, trainable=False, name='second_step')
   #双学习率衰减lr = 0.001 * 0.97^(step/100),之前的固定0.001学习率
    first_lr = tf.compat.v1.train.exponential_decay(FLAGS.lrate, first_step, \
                                                        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)
    second_lr = tf.compat.v1.train.exponential_decay(FLAGS.lrate, second_step, \
                                                            NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    first_opt = None
    second_opt = None
    if FLAGS.optimizer == 'Adagrad':
        first_opt = tf.train.AdagradOptimizer(first_lr)
        second_opt = tf.train.AdagradOptimizer(second_lr)
    elif FLAGS.optimizer == 'GradientDescent':
        first_opt = tf.train.GradientDescentOptimizer(first_lr)
        second_opt = tf.train.GradientDescentOptimizer(second_lr)
    elif FLAGS.optimizer == 'Adam':
        first_opt = tf.compat.v1.train.AdamOptimizer(first_lr)
        second_opt = tf.compat.v1.train.AdamOptimizer(second_lr)
    else:
        first_opt = tf.compat.v1.train.RMSPropOptimizer(first_lr, FLAGS.decay)
        second_opt = tf.compat.v1.train.RMSPropOptimizer(second_lr, FLAGS.decay)

    ''' Unused gradient clipping '''
    D_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='representation')
    O_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='output')
    W_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='weight')
    T_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='treatment')

    DOT_vars = D_vars + O_vars + T_vars#表征+输出+处理层

    train_first = first_opt.minimize(CFR.tot1_loss, global_step=first_step, var_list=DOT_vars)
    train_second = second_opt.minimize(CFR.tot_loss, global_step=second_step,var_list=W_vars)


    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []
    if FLAGS.varsel:
        all_weights = None
        all_beta = None

    all_preds_test = []

    ''' Run for all repeated experiments '''
    n_experiments = FLAGS.experiments
    for i_exp in range(1, n_experiments+1):
        log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))
        if i_exp==1 or FLAGS.experiments>1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x']  = D['x'][:,:,i_exp-1]
                D_exp['t']  = D['t'][:,i_exp-1:i_exp]
                D_exp['yf'] = D['yf'][:,i_exp-1:i_exp]
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf'][:,i_exp-1:i_exp]
                else:
                    D_exp['ycf'] = None

                if has_test:
                    D_exp_test = {}
                    D_exp_test['x']  = D_test['x'][:,:,i_exp-1]
                    D_exp_test['t']  = D_test['t'][:,i_exp-1:i_exp]
                    D_exp_test['yf'] = D_test['yf'][:,i_exp-1:i_exp]
                    if D_test['HAVE_TRUTH']:
                        D_exp_test['ycf'] = D_test['ycf'][:,i_exp-1:i_exp]
                    else:
                        D_exp_test['ycf'] = None
            else:
                datapath = dataform % i_exp
                D_exp = load_data(datapath)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    D_exp_test = load_data(datapath_test)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']
            if has_test:
                D_exp_test['HAVE_TRUTH'] = D_test['HAVE_TRUTH']

        ''' Split into training and validation sets '''
        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)

        ''' Run training loop '''
        losses, preds_train, preds_test, reps, reps_test = \
            train(CFR, sess, train_first, train_second, D_exp, I_valid, \
                D_exp_test, logfile, i_exp, outdir)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_losses.append(losses)

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train,1,3),0,2)
        if  has_test:
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test,1,3),0,2)
        out_losses = np.swapaxes(np.swapaxes(all_losses,0,2),0,1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)
        if FLAGS.output_csv:
            np.savetxt('%s_%d.csv' % (outform,i_exp), preds_train[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (outform_test,i_exp), preds_test[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (lossform,i_exp), losses, delimiter=',')

        ''' Compute weights if doing variable selection '''
        if FLAGS.varsel:
            if i_exp == 1:
                all_weights = sess.run(CFR.weights_in[0])
                all_beta = sess.run(CFR.weights_pred)
            else:
                all_weights = np.dstack((all_weights, sess.run(CFR.weights_in[0])))
                all_beta = np.dstack((all_beta, sess.run(CFR.weights_pred)))

        ''' Save results and predictions '''
        all_valid.append(I_valid)
        if FLAGS.varsel:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, w=all_weights, beta=all_beta, val=np.array(all_valid))
        else:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        if has_test:
            np.savez(npzfile_test, pred=out_preds_test)

        ''' Save representations '''
        if FLAGS.save_rep and i_exp == 1:
            np.savez(repfile, rep=reps)

            if has_test:
                np.savez(repfile_test, rep=reps_test)

def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir+'results_'+timestamp+'/'
    os.makedirs(outdir)

    try:
        run(outdir)
        # 根据outdir自动选择评估目录
        eval_dir = FLAGS.outdir
        print(f'\n开始评估实验结果: {eval_dir}')
        evaluate(eval_dir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise

if __name__ == '__main__':
    tf.compat.v1.app.run()
