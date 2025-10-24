import tensorflow as tf
import numpy as np

from utils import lindisc, mmd2_rbf, mmd2_lin, mmd2_rbf, wasserstein

def safe_sqrt(x):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, 1e-6, 1e8))

class Net(object):
    def __init__(self, dims, FLAGS):

        self.variables = {}
        self.wd_loss = 0

        ''' Initialize input placeholders '''
        self.x  = tf.compat.v1.placeholder("float", shape=[None, dims[1]], name='x') # Features
        self.t  = tf.compat.v1.placeholder("float", shape=[None, 1], name='t')   # Treatent
        self.y_ = tf.compat.v1.placeholder("float", shape=[None, 1], name='y_')  # Outcome
        self.do_in = tf.compat.v1.placeholder("float", name='dropout_in')
        self.do_out = tf.compat.v1.placeholder("float", name='dropout_out')
        self.p_t = tf.compat.v1.placeholder("float", name='p_treated')
        self.I  = tf.compat.v1.placeholder("int32", shape=[None, ], name='I')   # weight
        self.y_0_median = tf.compat.v1.placeholder("float", name='y_0_median')
        self.y_1_median = tf.compat.v1.placeholder("float", name='y_1_median')

        self.i0 = tf.to_int32(tf.where(self.t < 0.50)[:, 0])
        self.i1 = tf.to_int32(tf.where(self.t > 0.50)[:, 0])

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        elif FLAGS.nonlin.lower() == 'tanh':
            self.nonlin = tf.nn.tanh
        else:
            self.nonlin = tf.nn.relu
        
        self._build_graph(dims, FLAGS)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i)  # @TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.wd_loss += wd * tf.nn.l2_loss(var)
        return var

    def _build_graph(self, dims, FLAGS):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """

        n, dim_input, dim_in, dim_out = dims

        r_coef_y = FLAGS.p_coef_y
        r_coef_alpha = FLAGS.p_coef_alpha
        r_coef_beta = FLAGS.p_coef_beta
        r_coef_gamma = FLAGS.p_coef_gamma
        r_coef_mu = FLAGS.p_coef_mu
        r_coef_lambda = FLAGS.p_coef_lambda

        weights_in = []
        biases_in = []

        if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
            dim_in = dim_input
        if FLAGS.n_out == 0:
            if FLAGS.split_output == False:
                dim_out = dim_in + 1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm:
            self.bn_biases = []
            self.bn_scales = []


        ####################################################################
        ################# Construct Networks ###############################
        ####################################################################

        ''' Construct Representation Layers '''
        with tf.name_scope("representation"):
            h_rep_I, h_rep_norm_I, weights_in_I, biases_in_I = self._build_representation_graph(dim_input, dim_in, dim_out, FLAGS)
            h_rep_C, h_rep_norm_C, weights_in_C, biases_in_C = self._build_representation_graph(dim_input, dim_in, dim_out, FLAGS)
            h_rep_A, h_rep_norm_A, weights_in_A, biases_in_A = self._build_representation_graph(dim_input, dim_in, dim_out, FLAGS)
            self.h_rep = tf.concat((h_rep_I, h_rep_C, h_rep_A), axis=1)
            self.h_rep_norm = tf.concat((h_rep_norm_I, h_rep_norm_C, h_rep_norm_A), axis=1)
        weights_in = weights_in_I + weights_in_C + weights_in_A
        biases_in = biases_in_I + biases_in_C + biases_in_A
        self.weights_in_I = weights_in_I
        self.weights_in_C = weights_in_C
        self.weights_in_A = weights_in_A

        ''' Design Sample Weights to Auto-balance Confounders '''
        with tf.name_scope("weight"):
            if FLAGS.autoWeighting == 1 and FLAGS.batch_size == 0:
                sample_weight = tf.compat.v1.Variable(tf.ones([n, 1]), name='sample_weight')
                sample_weight = tf.gather(tf.abs(sample_weight), self.I)
                sample_weight_0 = tf.gather(sample_weight, self.i0)
                sample_weight_1 = tf.gather(sample_weight, self.i1)
            else:
                sample_weight = 1.0 
                sample_weight_0 = 1.0
                sample_weight_1 = 1.0
        self.sample_weight = sample_weight

        ''' Construct Two Sub-Networks for Predicting T and Y '''
        with tf.name_scope("treatment"):
            _, _, _, g_I2t = self._build_treatment_graph(h_rep_norm_I, dim_in, dim_out, self.do_out, 't', FLAGS)
            g_A2y, _, _, _, _      = self._build_output_graph(h_rep_norm_A, self.t, dim_in, dim_out, self.do_out, FLAGS)

        ''' Construct Output Layers '''
        with tf.name_scope("output"):
            y, weights_out, weights_pred, biases_out, bias_pred = self._build_output_graph(
                tf.concat([h_rep_norm_C, h_rep_norm_A], 1), self.t, 2 * dim_in, dim_out, self.do_out, FLAGS)
        self.weights_out = weights_out
        self.weights_pred = weights_pred


        ####################################################################
        ################# Objective Function ###############################
        ####################################################################
        

        ''' Deep Orthogonal Regularizer: Loss FUN L_O in Eq. (7). '''
        if r_coef_mu > 0:
            num_constrainedLayer = len(weights_in_I) if FLAGS.constrainedLayer == 0 else FLAGS.constrainedLayer
            III, CCC, AAA = weights_in_I[0], weights_in_C[0], weights_in_A[0]
            for i in range(1, num_constrainedLayer):
                III = tf.matmul(III, weights_in_I[i])
                CCC = tf.matmul(CCC, weights_in_C[i])
                AAA = tf.matmul(AAA, weights_in_A[i])
            I0 = tf.reduce_mean(tf.abs(III), axis=1)
            C0 = tf.reduce_mean(tf.abs(CCC), axis=1)
            A0 = tf.reduce_mean(tf.abs(AAA), axis=1)
            IA_loss = tf.reduce_sum(I0 * A0)
            CA_loss = tf.reduce_sum(C0 * A0)
            IC_loss = tf.reduce_sum(I0 * C0)
            L_O = r_coef_mu * (IA_loss + CA_loss + IC_loss)
        else:
            L_O = 0.0
        

        ''' Balancing Confounders: Loss FUN L_C_B in Eq. (4) '''
        if r_coef_gamma > 0:
            h_rep_norm_C_1 = tf.gather(h_rep_norm_C, self.i1)
            h_rep_norm_C_0 = tf.gather(h_rep_norm_C, self.i0)
            if FLAGS.use_p_correction:
                p = self.p_t
            else:
                p = 0.5
            B_rep_0 = sample_weight_0 * h_rep_norm_C_0
            B_rep_1 = sample_weight_1 * h_rep_norm_C_1
            
            mean_B_1 = tf.reduce_mean(B_rep_0,axis=0)
            mean_B_0 = tf.reduce_mean(B_rep_1,axis=0)

            B_dist = tf.reduce_sum(tf.square(2.0*p*mean_B_1 - 2.0*(1.0-p)*mean_B_0))
            L_C_B = safe_sqrt(tf.square(r_coef_gamma) * B_dist)
        else:
            L_C_B = 0.0


        ''' Decompose Instruments: Loss FUN L_I in Eq. (5) '''
        if r_coef_beta > 0:
            g_ILoss = r_coef_beta * tf.reduce_mean(sample_weight * g_I2t)
            I_error, _ = self._calculate_disc_I(h_rep_norm_I, self.t, self.y_, r_coef_beta, FLAGS)
        else:
            g_ILoss = 0.0
            I_error = 0.0
        L_I = I_error + g_ILoss


        ''' Decompose Adjustments: Loss FUN L_A in Eq. (3) '''
        if FLAGS.loss == 'l1':
            g_ALoss = tf.reduce_mean(sample_weight*tf.abs(self.y_-g_A2y))
            c_pred_error = -tf.reduce_mean(tf.abs(self.y_-g_A2y))
        elif FLAGS.loss == 'log':
            # Use DeR-CFR_0 style for adjustment network too
            g_A2y_prob = 0.995 / (1.0 + tf.exp(-g_A2y)) + 0.0025
            
            labels_A = tf.concat((1.0 - self.y_, self.y_), axis=1)
            logits_A = tf.concat((-g_A2y, g_A2y), axis=1)
            
            loss_A_per_sample = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_A, labels=labels_A)
            loss_A_per_sample = tf.reduce_mean(loss_A_per_sample, axis=1, keepdims=True)
            
            g_ALoss = tf.reduce_mean(sample_weight * loss_A_per_sample)
            c_pred_error = tf.reduce_mean(loss_A_per_sample)
        else:
            g_ALoss = tf.reduce_mean(sample_weight * tf.square(self.y_ - g_A2y))
            c_pred_error = tf.sqrt(tf.reduce_mean(tf.square(self.y_ - g_A2y)))
        if r_coef_alpha > 0:
            g_ALoss = r_coef_alpha * g_ALoss
            A_error, imb_dist = self._calculate_disc(h_rep_norm_A, r_coef_alpha, FLAGS)
        else:
            g_ALoss = 0.0
            A_error, imb_dist = 0.0
        L_A = A_error + g_ALoss
        

        ''' Outcome Regression: Loss FUN L_R in Eq. (8) '''
        if FLAGS.loss == 'l1':
            L_R = tf.reduce_mean(sample_weight*tf.abs(self.y_-y))
            pred_error = -tf.reduce_mean(tf.abs(self.y_-y))
        elif FLAGS.loss == 'log':
            # Use DeR-CFR_0 style: sigmoid with logits (more numerically stable)
            y_prob = 0.995 / (1.0 + tf.exp(-y)) + 0.0025
            
            # Construct one-hot labels: [1-y, y]
            labels = tf.concat((1.0 - self.y_, self.y_), axis=1)
            # Construct logits: [logit_0, logit_1] where logit_1 = y (original), logit_0 = -y
            logits = tf.concat((-y, y), axis=1)
            
            # Use sigmoid_cross_entropy (more stable than manual log)
            loss_per_sample = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            loss_per_sample = tf.reduce_mean(loss_per_sample, axis=1, keepdims=True)  # Average over 2 classes
            
            L_R = tf.reduce_mean(sample_weight * loss_per_sample)
            pred_error = tf.reduce_mean(loss_per_sample)
        else:
            L_R = tf.reduce_mean(sample_weight * tf.square(self.y_ - y))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(self.y_ - y)))
        if r_coef_y > 0:
            L_R = r_coef_y * L_R


        if r_coef_lambda > 0: 
            ''' Regularization: Loss FUN R_W in Eq. (10) '''
            if FLAGS.rep_weight_decay:
                for i in range(0, FLAGS.n_in):
                    if not (FLAGS.varsel and i == 0):  # No penalty on W in variable selection
                        self.wd_loss += tf.nn.l2_loss(weights_in[i])
            R_W = r_coef_lambda * self.wd_loss

            ''' Weights Soft Regularizer: Loss FUN R_C_B in Eq. (11) '''
            R_C_B = r_coef_lambda * (tf.square(tf.reduce_mean(sample_weight_0) - 1.0) + tf.square(tf.reduce_mean(sample_weight_1) - 1.0))

            ''' Orthogonal Regularizer: Loss FUN R_O in Eq. (12) '''
            R_O = r_coef_lambda * (tf.square(tf.reduce_sum(I0) - 1.0) + tf.square(tf.reduce_sum(C0) - 1.0) + tf.square(tf.reduce_sum(A0) - 1.0))
            
            ''' Total Regularization: Loss FUN Reg in Eq. (14) '''
            Reg = R_W + R_C_B + R_O
        else:
            R_W, R_C_B, R_O, Reg = 0.0, 0.0, 0.0, 0.0
        

        tot_error  = L_R + L_C_B + Reg
        tot1_error = L_R  + L_I + L_A + L_O + R_W + R_O
        val_error  = pred_error + A_error/10 + I_error/10


        ####################################################################
        ################# Return Outcomes and Losses #######################
        ####################################################################

        if FLAGS.varsel:
            self.w_proj = tf.placeholder("float", shape=[dim_input], name='w_proj')
            self.projection = weights_in[0].assign(self.w_proj)

        if FLAGS.loss == 'log':
            # For binary outcomes (Jobs/TWINS), output probability for causal inference
            # Use sigmoid to convert logits to probabilities
            y_prob = 0.995 / (1.0 + tf.exp(-y)) + 0.0025
            self.output = y_prob  # Output probability [0, 1] for causal effect calculation
            
            # Also save discrete label for accuracy calculation
            label = y_prob
            one = tf.ones_like(label)
            zero = tf.zeros_like(label)
            label_discrete = tf.where(label < 0.5, x=zero, y=one)
            self.output_discrete = label_discrete  # Discrete output {0, 1} for accuracy only
        else:
            # For continuous outcomes (IHDP), output is the raw value
            self.output = y
            self.output_discrete = y  # Same as output for continuous case
        self.tot1_loss = tot1_error
        self.tot_loss = tot_error
        self.val_loss = val_error
        self.imb_loss = A_error
        self.imb_dist = imb_dist
        self.pred_loss = pred_error

        self.loss_1 = r_coef_y * pred_error

        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.biases_in = biases_in
        self.biases_out = biases_out
        self.bias_pred = bias_pred

    def _build_representation_graph(self, dim_input, dim_in, dim_out, FLAGS):
        weights_in = [];
        biases_in = []

        h_in = [self.x]
        for i in range(0, FLAGS.n_in):
            if i == 0:
                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel:
                    weights_in.append(tf.Variable(1.0 / dim_input * tf.ones([dim_input])))
                else:
                    weights_in.append(tf.Variable(
                        tf.random_normal([dim_input, dim_in], stddev=FLAGS.weight_init / np.sqrt(dim_input))))
            else:
                weights_in.append(
                    tf.Variable(tf.random_normal([dim_in, dim_in], stddev=FLAGS.weight_init / np.sqrt(dim_in))))

            ''' If using variable selection, first layer is just rescaling'''
            if FLAGS.varsel and i == 0:
                biases_in.append([])
                h_in.append(tf.multiply(h_in[i], weights_in[i]))
            else:
                biases_in.append(tf.Variable(tf.zeros([1, dim_in])))
                z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

                if FLAGS.batch_norm:
                    batch_mean, batch_var = tf.nn.moments(z, [0])

                    if FLAGS.normalization == 'bn_fixed':
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                    else:
                        self.bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                        self.bn_scales.append(tf.Variable(tf.ones([dim_in])))
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, self.bn_biases[-1], self.bn_scales[-1],1e-3)

                h_in.append(self.nonlin(z))
                h_in[i + 1] = tf.nn.dropout(h_in[i + 1], self.do_in)

        h_rep = h_in[len(h_in) - 1]

        if FLAGS.normalization == 'divide':
            h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))
        else:
            h_rep_norm = 1.0 * h_rep

        return h_rep, h_rep_norm, weights_in, biases_in

    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out] * FLAGS.n_out)

        weights_out = [];
        biases_out = []

        for i in range(0, FLAGS.n_out):
            wo = self._create_variable_with_weight_decay(
                tf.random_normal([dims[i], dims[i + 1]],
                                 stddev=FLAGS.weight_init / np.sqrt(dims[i])),
                'y_w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1, dim_out])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            h_out.append(self.nonlin(z))
            h_out[i + 1] = tf.nn.dropout(h_out[i + 1], do_out)

        weights_pred = self._create_variable(tf.random_normal([dim_out, 1],
                                                              stddev=FLAGS.weight_init / np.sqrt(dim_out)), 'y_w_pred')
        bias_pred = self._create_variable(tf.zeros([1]), 'y_b_pred')

        if FLAGS.varsel or FLAGS.n_out == 0:
            self.wd_loss += tf.nn.l2_loss(
                tf.slice(weights_pred, [0, 0], [dim_out - 1, 1]))  
        else:
            self.wd_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.matmul(h_pred, weights_pred) + bias_pred

        return y, weights_out, weights_pred, biases_out, bias_pred

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''

        if FLAGS.split_output:
            rep0 = tf.gather(rep, self.i0)
            rep1 = tf.gather(rep, self.i1)

            y0, weights_out0, weights_pred0, biases_out0, bias_pred0 = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS)
            y1, weights_out1, weights_pred1, biases_out1, bias_pred1 = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS)

            y = tf.dynamic_stitch([self.i0, self.i1], [y0, y1])
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
            biases_out = biases_out0 + biases_out1
            bias_pred = bias_pred0 + bias_pred1
        else:
            h_input = tf.concat([rep, t], 1)
            y, weights_out, weights_pred, biases_out, bias_pred = self._build_output(h_input, dim_in + 1, dim_out, do_out, FLAGS)

        return y, weights_out, weights_pred, biases_out, bias_pred

    def _calculate_disc(self, h_rep_norm, coef, FLAGS):
        t = self.t

        if FLAGS.use_p_correction:
            p_ipm = self.p_t
        else:
            p_ipm = 0.5

        if FLAGS.imb_fun == 'mmd2_rbf':
            imb_dist = mmd2_rbf(h_rep_norm, t, p_ipm, FLAGS.rbf_sigma)
            imb_error = coef * imb_dist
        elif FLAGS.imb_fun == 'mmd2_lin':
            imb_dist = mmd2_lin(h_rep_norm, t, p_ipm)
            imb_error = coef * mmd2_lin(h_rep_norm, t, p_ipm)
        elif FLAGS.imb_fun == 'mmd_rbf':
            imb_dist = tf.abs(mmd2_rbf(h_rep_norm, t, p_ipm, FLAGS.rbf_sigma))
            imb_error = safe_sqrt(tf.square(coef) * imb_dist)
        elif FLAGS.imb_fun == 'mmd_lin':
            imb_dist = mmd2_lin(h_rep_norm, t, p_ipm)
            imb_error = safe_sqrt(tf.square(coef) * imb_dist)
        elif FLAGS.imb_fun == 'wass':
            imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                            sq=False, backpropT=FLAGS.wass_bpt)
            imb_error = coef * imb_dist
            self.imb_mat = imb_mat  # FOR DEBUG
        elif FLAGS.imb_fun == 'wass2':
            imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                            sq=True, backpropT=FLAGS.wass_bpt)
            imb_error = coef * imb_dist
            self.imb_mat = imb_mat  # FOR DEBUG
        else:
            imb_dist = lindisc(h_rep_norm, t, p_ipm)
            imb_error = coef * imb_dist

        return imb_error, imb_dist

    def _build_treatment_graph(self, h_rep_norm, dim_in, dim_out, do_out, mode, FLAGS):
        t = self.t

        h_t = [h_rep_norm]

        num = 1
        if mode == 'w':
            num = 1
        elif mode == 't':
            num = FLAGS.n_t
        

        dims = [dim_in] + ([dim_out]*num)

        weights_t = []; biases_t = []

        for i in range(0, num):
            wo = self._create_variable_with_weight_decay(
                    tf.random_normal([dims[i], dims[i+1]],
                        stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                    'w_out_%d' % i, 1.0)
            weights_t.append(wo)

            biases_t.append(tf.Variable(tf.zeros([1,dim_out]), name = 'b_out_%d' % i))
            z = tf.matmul(h_t[i], weights_t[i]) + biases_t[i]
            # No batch norm on output because p_cf != p_f

            h_t.append(self.nonlin(z))
            h_t[i+1] = tf.nn.dropout(h_t[i+1], do_out)

        weights_pred = self._create_variable(tf.random_normal([dim_out,1],
            stddev=FLAGS.weight_init/np.sqrt(dim_out)), 'w_pred_t')
        bias_pred = self._create_variable(tf.zeros([1]), 'b_pred_t')

        self.wd_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_t[-1]
        y = tf.matmul(h_pred, weights_pred)+bias_pred
        
        sigma = 0.995/(1.0+tf.exp(-y)) + 0.0025
        sigma = tf.clip_by_value(sigma, clip_value_min=0.0025, clip_value_max=0.9975)

        pi_0 = tf.multiply(t, sigma) + tf.multiply(1.0-t, 1.0-sigma)
        
        if mode == 'w':
            cost = -tf.reduce_mean( tf.multiply(t, tf.log(sigma)) + tf.multiply(1.0-t, tf.log(1.0-sigma)) )
        elif mode == 't':
            cost = -( tf.multiply(t, tf.log(sigma)) + tf.multiply(1.0-t, tf.log(1.0-sigma)) )
        
        return weights_pred, weights_t, pi_0, cost
    
    def _calculate_disc_I(self, rep, t, y_, coef, FLAGS):
        if FLAGS.use_p_correction:
            p = self.p_t
        else:
            p = 0.5

        if FLAGS.loss == 'log':
            i_1_1,_ = tf.unique(tf.sort(tf.concat([tf.where(t>0)[:,0], tf.where(y_>0)[:,0]],axis=0)))
            i_1_0,_ = tf.unique(tf.sort(tf.concat([tf.where(t>0)[:,0], tf.where(y_<1)[:,0]],axis=0)))
            i_0_1,_ = tf.unique(tf.sort(tf.concat([tf.where(t<1)[:,0], tf.where(y_>0)[:,0]],axis=0)))
            i_0_0,_ = tf.unique(tf.sort(tf.concat([tf.where(t<1)[:,0], tf.where(y_<1)[:,0]],axis=0)))
        else:
            i_1_1,_ = tf.unique(tf.sort(tf.concat([tf.where(t>0)[:,0], tf.where(y_>self.y_1_median)[:,0]],axis=0)))
            i_1_0,_ = tf.unique(tf.sort(tf.concat([tf.where(t>0)[:,0], tf.where(y_<self.y_1_median)[:,0]],axis=0)))
            i_0_1,_ = tf.unique(tf.sort(tf.concat([tf.where(t<1)[:,0], tf.where(y_>self.y_0_median)[:,0]],axis=0)))
            i_0_0,_ = tf.unique(tf.sort(tf.concat([tf.where(t<1)[:,0], tf.where(y_<self.y_0_median)[:,0]],axis=0)))

        sample_weight = self.sample_weight

        if FLAGS.autoWeighting == 1:
            w_1_1 = tf.gather(sample_weight,i_1_1)
            w_1_0 = tf.gather(sample_weight,i_1_0)
            w_0_1 = tf.gather(sample_weight,i_0_1)
            w_0_0 = tf.gather(sample_weight,i_0_0)
        else:
            w_1_1 = 1
            w_1_0 = 1
            w_0_1 = 1
            w_0_0 = 1

        A_1_1 = tf.gather(rep,i_1_1)
        A_1_0 = tf.gather(rep,i_1_0)
        A_0_1 = tf.gather(rep,i_0_1)
        A_0_0 = tf.gather(rep,i_0_0)

        mean_1_1 = tf.reduce_mean(w_1_1 * A_1_1,axis=0)
        mean_1_0 = tf.reduce_mean(w_1_0 * A_1_0,axis=0)
        mean_0_1 = tf.reduce_mean(w_0_1 * A_0_1,axis=0)
        mean_0_0 = tf.reduce_mean(w_0_0 * A_0_0,axis=0)

        mmd_1 = tf.reduce_sum(tf.square(2.0*p*mean_1_1 - 2.0*(1.0-p)*mean_1_0))
        mmd_0 = tf.reduce_sum(tf.square(2.0*p*mean_0_1 - 2.0*(1.0-p)*mean_0_0))

        imb_dist = mmd_1 + mmd_0
        imb_error = safe_sqrt(tf.square(coef) * imb_dist)

        return imb_dist, imb_error
