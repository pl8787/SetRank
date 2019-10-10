from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import io_ops

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
            
    return outputs
    

def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N = tf.shape(inputs)[0]
    T = inputs.get_shape().as_list()[1]
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.float32(np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)]))

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs


def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        cloze=False,
                        scope="multihead_attention", 
                        reuse=None,
                        activation=tf.nn.relu):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    print_op_list = []
    def desc_tensor(t, p=print_op_list):
        if t.dtype == tf.bool:
            p.append(
                tf.print('#'*20,
                    '\n\tName:', t.name,
                    '\n\tShape:', tf.shape(t),
                '\n', '#'*20, '\n')
            )
        else:
            p.append(
                tf.print('#'*20,
                    '\n\tName:', t.name,
                    '\n\tShape:', tf.shape(t),
                    '\n\tMean:', tf.reduce_mean(tf.abs(t)),
                    '\n\tMin/Max:', tf.reduce_min(t), tf.reduce_max(t),
                '\n', '#'*20, '\n')
            )
        return

    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        #desc_tensor(queries)
        #desc_tensor(keys)
        print_op_list.append(tf.print("queries", queries, summarize=-1))
        print_op_list.append(tf.print("keys", keys, summarize=-1))

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=activation) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=activation) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=activation) # (N, T_k, C)
        
        #desc_tensor(Q)
        #desc_tensor(K)
        print_op_list.append(tf.print("Q", Q, summarize=-1))
        print_op_list.append(tf.print("K", K, summarize=-1))

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        #desc_tensor(outputs)
        print_op_list.append(tf.print("Q*K", outputs, summarize=-1))
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        #desc_tensor(outputs)
        print_op_list.append(tf.print("scale", outputs, summarize=-1))
        
        ## Key Masking
        #key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        #key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        #key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        #
        #paddings = tf.ones_like(outputs)*(-2**32+1)
        #outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
        #desc_tensor(outputs)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Cloze = Current blinding
        if cloze:
            range_q_vals = tf.expand_dims(tf.range(0, tf.shape(outputs)[1]), 1)
            range_k_vals = tf.expand_dims(tf.range(0, tf.shape(outputs)[2]), 0)
            mesh_diff_vals = tf.cast(range_q_vals - range_k_vals, tf.float32)
            masks = tf.tile(tf.expand_dims(mesh_diff_vals, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, -1), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
        #desc_tensor(outputs)
        print_op_list.append(tf.print("softmax", outputs, summarize=-1))
         
        ## Query Masking
        #query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        #query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        #query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        #outputs *= query_masks # broadcasting. (N, T_q, C)
        #desc_tensor(outputs)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        #desc_tensor(outputs)
        print_op_list.append(tf.print("dropout", outputs, summarize=-1))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        #desc_tensor(outputs)
        print_op_list.append(tf.print("Q*K*V", outputs, summarize=-1))
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
        #desc_tensor(outputs)
              
        # Residual connection
        outputs += queries
        #desc_tensor(outputs)
        print_op_list.append(tf.print("residual", outputs, summarize=-1))
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
        #desc_tensor(outputs)
        print_op_list.append(tf.print("normalize", outputs, summarize=-1))
 
    return outputs, print_op_list


def multihead_attention_mask(queries, 
                        keys,
                        query_masks,
                        key_masks, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        cloze=False,
                        scope="multihead_attention", 
                        reuse=None,
                        activation=tf.nn.relu):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=activation) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=activation) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=activation) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Cloze = Current blinding
        if cloze:
            range_q_vals = tf.expand_dims(tf.range(0, tf.shape(outputs)[1]), 1)
            range_k_vals = tf.expand_dims(tf.range(0, tf.shape(outputs)[2]), 0)
            mesh_diff_vals = tf.cast(range_q_vals - range_k_vals, tf.float32)
            masks = tf.tile(tf.expand_dims(mesh_diff_vals, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, -1), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs


def induced_multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        num_induced=10,
                        dropout_rate=0,
                        is_training=True,
                        scope="induced_multihead_attention", 
                        reuse=None):
    '''Applies induced multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      num_induced: An int. Dimension of induce variable I.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''

    print_op_list = []
    causality = False
    cloze = False
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        #I_var = tf.Variable(
        #        tf.random_normal(shape=[num_induced, num_units],
        #        mean=0,
        #        stddev=0.1),
        #    name='induce_param')
        I_var = tf.Variable(
                np.random.normal(size=[num_induced, num_units],
                loc=0,
                scale=0.1),
            dtype=tf.float32,
            name='induce_param')
        #I_norm = I_var * tf.rsqrt(tf.reduce_sum(I_var ** 2, axis=1, keepdims=True))
        #I = tf.tile(tf.expand_dims(I_norm, axis=0), [tf.shape(queries)[0], 1, 1])
        I = tf.tile(tf.expand_dims(I_var, axis=0), [tf.shape(queries)[0], 1, 1])
        print_op_list.append(tf.print("I", I, summarize=-1))
        H, print_ops = multihead_attention(I, keys, num_units, num_heads, dropout_rate, 
            is_training, causality, cloze, scope+'/L1', reuse)
        print_op_list += print_ops
        print_op_list.append(tf.print("H", H, summarize=-1))
        O, _ = multihead_attention(queries, H, num_units, num_heads, dropout_rate, 
            is_training, causality, cloze, scope+'/L2', reuse)
        print_op_list.append(tf.print("O", O, summarize=-1))
    return O, print_op_list


def induced_multihead_attention_mask(queries, 
                        keys, 
                        query_masks,
                        key_masks,
                        num_units=None, 
                        num_heads=8, 
                        num_induced=10,
                        dropout_rate=0,
                        is_training=True,
                        scope="induced_multihead_attention", 
                        reuse=None):
    '''Applies induced multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      num_induced: An int. Dimension of induce variable I.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''

    causality = False
    cloze = False
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        #I_var = tf.Variable(
        #        tf.random_normal(shape=[num_induced, num_units],
        #        mean=0,
        #        stddev=0.1),
        #    name='induce_param')
        I_var = tf.Variable(
                np.random.normal(size=[num_induced, num_units],
                loc=0,
                scale=0.1),
            dtype=tf.float32,
            name='induce_param')
        #I_norm = I_var * tf.rsqrt(tf.reduce_sum(I_var ** 2, axis=1, keepdims=True))
        #I = tf.tile(tf.expand_dims(I_norm, axis=0), [tf.shape(queries)[0], 1, 1])
        I = tf.tile(tf.expand_dims(I_var, axis=0), [tf.shape(queries)[0], 1, 1])
        I_masks = tf.ones(shape=[tf.shape(queries)[0], num_induced], dtype=tf.float32)
        H = multihead_attention_mask(I, keys, I_masks, key_masks, num_units, num_heads, dropout_rate, 
            is_training, causality, cloze, scope+'/L1', reuse)
        O = multihead_attention_mask(queries, H, query_masks, I_masks, num_units, num_heads, dropout_rate, 
            is_training, causality, cloze, scope+'/L2', reuse)
    return O


def attention_matrix(queries, 
                        keys, 
                        num_units=None, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="attention_matrix", 
                        reuse=None):
    '''Applies attention matrix.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, T_k)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
        # Multiplication
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) # (N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)
        
        '''
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (N, T_q, T_k)
  
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (N, T_q, T_k)
        outputs = tf.where(tf.equal(query_masks, 0), paddings, outputs) # (N, T_q, T_k)
        '''

    return outputs

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                reuse=None,
                activation=tf.nn.relu):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": activation, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = normalize(outputs)
    
    return outputs

def feedforward_multi(inputs, 
                num_units=[],
                scope="multihead_attention", 
                reuse=False,
                activation=tf.nn.relu):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(len(num_units)-1):
            params = {"inputs": inputs, "filters": num_units[i], "kernel_size": 1,
                      "activation": activation, "use_bias": True}
            inputs = tf.layers.conv1d(**params)
        
            # Normalize
            inputs = normalize(inputs)
    
        params = {"inputs": inputs, "filters": num_units[-1], "kernel_size": 1,
              "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = normalize(outputs)
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)
    
def label_smoothing_mask(inputs, cls_num, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    K = inputs.get_shape().as_list()[-1] # number of channels
    cls_mask = tf.sequence_mask(cls_num, K, dtype=tf.float32)
    cls_num = tf.expand_dims(
                tf.tile(
                    tf.expand_dims(tf.cast(cls_num, tf.float32), axis=1), 
                    [1, K])
                , axis=1)
    cls_mask = tf.expand_dims(cls_mask, axis=1)
    out = ((1-epsilon) * inputs) + (epsilon / cls_num)
    out = out * cls_mask
    return out
    
