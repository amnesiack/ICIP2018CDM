import tensorflow as tf
# network structure

def threestage(input, is_training=True):
    num_feature = 128
    with tf.variable_scope('stage1'):
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(tf.expand_dims(input[:,:,:,1], -1), num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(2, 4 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, epsilon=1e-5, training=is_training))
        with tf.variable_scope('block5'):
            output = tf.layers.conv2d(output, 1, 3, padding='same')
            S1InterG = tf.expand_dims(input[:,:,:,1], -1) + output
    with tf.variable_scope('stage2-1'):
        S2RG = tf.concat([tf.expand_dims(input[:,:,:,0], -1), S1InterG], 3)
        with tf.variable_scope('block6'):
            output = tf.layers.conv2d(S2RG, num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(7, 9 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, epsilon=1e-5, training=is_training))
        with tf.variable_scope('block10'):
            output = tf.layers.conv2d(output, 2, 3, padding='same')
            S2InterRG = S2RG + output
    with tf.variable_scope('stage2-2'):
        S2GB = tf.concat([S1InterG, tf.expand_dims(input[:,:,:,2], -1)], 3)
        with tf.variable_scope('block11'):
            output = tf.layers.conv2d(S2GB, num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(12, 14 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, epsilon=1e-5, training=is_training))
        with tf.variable_scope('block15'):
            output = tf.layers.conv2d(output, 2, 3, padding='same')
            S2InterGB = S2GB + output
    with tf.variable_scope('stage3'):
        S3RGB = tf.concat([tf.expand_dims(S2InterRG[:,:,:,0], -1), S2InterGB], 3)
        with tf.variable_scope('block16'):
            output = tf.layers.conv2d(S3RGB, num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(17, 19 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, epsilon=1e-5, training=is_training))
        with tf.variable_scope('block20'):
            output = tf.layers.conv2d(output, 3, 3, padding='same')
    return S1InterG, S2InterRG, S2InterGB, S3RGB + output