import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from TensorMonitor.tensor_manager import TensorMonitor

sess = tf.Session()

#####################   User Setting    ##################
DEBUG = False
PLOT = True
TEST = False
is_train = True
control = True             # if control = True, ff and fb weight are transposed;
ifLSTM = True
ifGlobalLSTM = False
error_factor = 0.3

#####################   get input (change to img series in future)  ######################
data_dir = 'temp'
mnist = input_data.read_data_sets(data_dir, one_hot=False)
train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])
train_labels = mnist.train.labels
test_labels = mnist.test.labels

####################    model setting   #################
batch_size = 20
test_batch_size = 20
learning_rate = 0.001
generations = 500
interval = 10
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
channel_num = 1
input_shape = (batch_size, image_width, image_height, channel_num)
target_size = np.max(train_labels) + 1
y_target = tf.placeholder(tf.int32, shape=(batch_size))

level_num = 3
convLSTM_cycle = 3
cycle_num = 4

conv_features = [20, 40, 60]

conv1_weight = tf.Variable(tf.truncated_normal([3, 3, channel_num, conv_features[0]], stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros([conv_features[0]], dtype=tf.float32))
conv_lstm1 = tf.Variable(tf.truncated_normal([3, 3, 3*channel_num, channel_num], stddev=0.1, dtype=tf.float32))
conv_lstm_bias1 = tf.Variable(tf.zeros([channel_num], dtype=tf.float32))
conv_abar_weight1 = tf.Variable(tf.truncated_normal([3, 3, channel_num, channel_num], stddev=0.1, dtype=tf.float32))
conv_abar_bias1 = tf.Variable(tf.zeros([channel_num], dtype=tf.float32))
if control:
    fb1_weight = conv1_weight
else:
    fb1_weight = tf.Variable(tf.truncated_normal([3, 3, channel_num, conv_features[0]], stddev=0.1, dtype=tf.float32))
fb1_bias = tf.Variable(tf.zeros([channel_num], dtype=tf.float32))

conv2_weight = tf.Variable(tf.truncated_normal([3, 3, conv_features[0], conv_features[1]], stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv_features[1]], dtype=tf.float32))
conv_lstm2 = tf.Variable(tf.truncated_normal([3, 3, 3*conv_features[0], conv_features[0]], stddev=0.1, dtype=tf.float32))
conv_lstm_bias2 = tf.Variable(tf.zeros([conv_features[0]], dtype=tf.float32))
conv_abar_weight2 = tf.Variable(tf.truncated_normal([3, 3, conv_features[0], conv_features[0]], stddev=0.1, dtype=tf.float32))
conv_abar_bias2 = tf.Variable(tf.zeros([conv_features[0]], dtype=tf.float32))
if control:
    fb2_weight = conv2_weight
else:
    fb2_weight = tf.Variable(tf.truncated_normal([3, 3, conv_features[0], conv_features[1]], stddev=0.1, dtype=tf.float32))
fb2_bias = tf.Variable(tf.zeros([conv_features[0]], dtype=tf.float32))

conv3_weight = tf.Variable(tf.truncated_normal([3, 3, conv_features[1], conv_features[2]], stddev=0.1, dtype=tf.float32))
conv3_bias = tf.Variable(tf.zeros([conv_features[2]], dtype=tf.float32))
conv_lstm3 = tf.Variable(tf.truncated_normal([3, 3, 2*conv_features[1], conv_features[1]], stddev=0.1, dtype=tf.float32))
conv_lstm_bias3 = tf.Variable(tf.zeros([conv_features[1]], dtype=tf.float32))
conv_abar_weight3 = tf.Variable(tf.truncated_normal([3, 3, conv_features[1], conv_features[1]], stddev=0.1, dtype=tf.float32))
conv_abar_bias3 = tf.Variable(tf.zeros([conv_features[1]], dtype=tf.float32))

full1_weight = tf.Variable(tf.truncated_normal([960, 10], stddev=0.1, dtype=tf.float32))
full1_bias = tf.Variable(tf.truncated_normal([10], stddev=0.1, dtype=tf.float32))


conv_weight = [conv1_weight, conv2_weight, conv3_weight]
conv_bias = [conv1_bias, conv2_bias, conv3_bias]
conv_lstm = [conv_lstm1, conv_lstm2, conv_lstm3]
conv_lstm_bias = [conv_lstm_bias1, conv_lstm_bias2, conv_lstm_bias3]
conv_abar_weight = [conv_abar_weight1, conv_abar_weight2, conv_abar_weight3]
conv_abar_bias = [conv_abar_bias1, conv_abar_bias2, conv_abar_bias3]
fb_weight = [fb1_weight, fb2_weight]
fb_bias = [fb1_bias, fb2_bias]



#######################################################################################################
image_input = tf.placeholder(tf.float32, shape=input_shape)
input2e = tf.placeholder(tf.float32, shape=[batch_size, 7, 7, 40], name="plh_2e")
input2r = tf.placeholder(tf.float32, shape=[batch_size, 7, 7, 40], name="plh_2r")
input1e = tf.placeholder(tf.float32, shape=[batch_size, 14, 14, 20], name="plh_1e")
input1r = tf.placeholder(tf.float32, shape=[batch_size, 14, 14, 20], name="plh_1r")
input0e = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1], name="plh_0e")
input0r = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1], name="plh_0r")
if ifLSTM and ifGlobalLSTM:
    lstm_state2 = tf.placeholder(tf.float32, shape=[batch_size, 7, 7, 40], name="plh_lstm2")
    lstm_state1 = tf.placeholder(tf.float32, shape=[batch_size, 14, 14, 20], name="plh_lstm1")
    lstm_state0 = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1], name="plh_lstm0")
    lstm_state = [lstm_state0, lstm_state1, lstm_state2]
new_lstm_state = []
input_e = [input0e, input1e, input2e]
input_r = [input0r, input1r, input2r]
new_represent = []  # each time session run, will it be cleared to [] ?
new_error = []
img = tf.layers.batch_normalization(image_input, training=is_train, name='normalized_image_input')
a = [img]

#####################      build one recurrent instance    ###########################
for l in reversed(range(level_num)):
    if ifLSTM:
        tmp = 0
        output_channels = 0

        if l==level_num-1:
            tmp = tf.concat([input_r[l], input_e[l]], 3)
            output_channels = conv_features[l-1]
            #new_represent.append([])
        else:
            up = tf.nn.conv2d_transpose(input_r[l+1], fb_weight[l], input_e[l].get_shape().as_list(), strides=[1, 2, 2, 1], padding='SAME')
            upsample = tf.layers.batch_normalization(tf.nn.relu(tf.nn.bias_add(up, fb_bias[l])), training=is_train)
            tmp = tf.concat([input_r[l], input_e[l], upsample], 3)
            if l==0:
                output_channels = channel_num
            else:
                output_channels = conv_features[l-1]
        if ifGlobalLSTM is False:
            prevState = []
        else:
            prevState = [lstm_state[l]]

        for t in range(convLSTM_cycle):
            with tf.variable_scope("convLSTM"+"_level"+str(l)+'_recur_'+str(t)):
                cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=tmp.get_shape().as_list()[1:], output_channels=output_channels, kernel_shape=[3, 3])
                if len(prevState)==0:
                    prevState.append(cell.zero_state(batch_size=batch_size, dtype=tf.float32))
                output, final_state = cell.call(inputs=tmp, state=prevState[-1])
                prevState.append(final_state)
                if t == convLSTM_cycle-1:
                    final_lstm_output = output

        new_represent.insert(0, tf.layers.batch_normalization(final_lstm_output, training=is_train))
        new_lstm_state.insert(0, prevState[-1])

    elif l == level_num-1:
        new_represent = []  # each time session run, will it be cleared to [] ?
        new_error = []
        tmp = tf.concat([input_r[l], input_e[l]], 3)
        tmp1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tmp, conv_lstm[l], strides=[1, 1, 1, 1], padding='SAME'), conv_lstm_bias[l]))
        new_represent.insert(0, tf.layers.batch_normalization(tmp1, training=is_train))
    else:
        up = tf.nn.conv2d_transpose(input_r[l+1], fb_weight[l], input_e[l].get_shape().as_list(), strides=[1, 2, 2, 1], padding='SAME')
        upsample = tf.layers.batch_normalization(tf.nn.relu(tf.nn.bias_add(up, fb_bias[l])), training=is_train)
        tmp = tf.concat([input_r[l], input_e[l], upsample], 3)
        tmp1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tmp, conv_lstm[l], strides=[1, 1, 1, 1], padding='SAME'), conv_lstm_bias[l]))
        new_represent.insert(0, tf.layers.batch_normalization(tmp1, training=is_train))

for l in range(level_num):
    tmp = tf.nn.conv2d(new_represent[l], conv_abar_weight[l], strides=[1, 1, 1, 1], padding='SAME')
    tmp1 = tf.nn.relu(tf.nn.bias_add(tmp, conv_abar_bias[l]), name='level'+str(l)+'_pred')
    tmp2 = tf.layers.batch_normalization(tmp1, training=is_train)

    err = tf.subtract(tmp2, a[-1], name='level'+str(l)+'_error')
    error = tf.layers.batch_normalization(err, name='level'+str(l)+'_error_norm')
    new_error.append(error)

    if l < (level_num-1):
        tmp3 = tf.nn.conv2d(new_error[-1], conv_weight[l], strides=[1, 1, 1, 1], padding='SAME')
        tmp4 = tf.nn.relu(tf.nn.bias_add(tmp3, conv_bias[l]))
        tmp5 = tf.nn.max_pool(tmp4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='level'+str(l+1)+'_err_in')
        a.append(tf.layers.batch_normalization(tmp5, training=is_train))

#######  get loss   ######
tmp1 = tf.nn.conv2d(new_error[-1], conv_weight[2], strides=[1, 1, 1, 1], padding='SAME')
tmp2 = tf.nn.relu(tf.nn.bias_add(tmp1, conv_bias[2]))
tmp3 = tf.nn.max_pool(tmp2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
final_conv_shape = tmp3.get_shape().as_list()
final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
flat_output = tf.reshape(tmp3, [final_conv_shape[0], final_shape])
full1_weight = tf.Variable(tf.truncated_normal([final_shape, target_size], stddev=0.1, dtype=tf.float32))
full1_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))
output = tf.add(tf.matmul(flat_output, full1_weight), full1_bias)
#####
los2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y_target), name="label_loss")
all_lev_error = 0
for level_error in new_error:
    err_square = tf.pow(level_error, 2)
    z = tf.zeros_like(level_error, dtype=tf.float32)
    sum = [err_square, z]
    all_lev_error = all_lev_error + tf.reduce_mean(sum)

loss = error_factor*all_lev_error + los2

optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = optimizer.minimize(loss)
train_pred = tf.nn.softmax(output)

def get_accuracy(current, target):
    prediction = np.argmax(current, axis=1)
    num_correct = np.sum(np.equal(prediction, target))
    return 100. * num_correct/prediction.shape[0]

###################################################################################
e0_0 = np.zeros(shape=input_shape)
e0_1 = np.zeros(shape=[batch_size, 14, 14, conv_features[0]])
e0_2 = np.zeros(shape=[batch_size, 7, 7, conv_features[1]])
r0_0 = np.zeros(shape=input_shape)
r0_1 = np.zeros(shape=[batch_size, 14, 14, conv_features[0]])
r0_2 = np.zeros(shape=[batch_size, 7, 7, conv_features[1]])

init = tf.global_variables_initializer()
sess.run(init)
train_loss = []
temp_train_loss = []
train_pred_plot = []

for i in range(generations):
    rand_index = np.random.choice(len(train_xdata), size=batch_size)
    rand_x = train_xdata[rand_index]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = train_labels[rand_index]
    # session run recurrent
    for t in range(cycle_num):
        if ifLSTM and ifGlobalLSTM:
            if t==0:
                l0 = np.zeros(shape=input_shape)
                l1 = np.zeros(shape=[batch_size, 14, 14, conv_features[0]])
                l2 = np.zeros(shape=[batch_size, 7, 7, conv_features[1]])
                train_dict = {image_input:rand_x, input0e:e0_0, input1e:e0_1, input2e:e0_2,
                            input0r:r0_0, input1r:r0_1, input2r:r0_2, y_target:rand_y,
                            lstm_state0:l0, lstm_state1:l1, lstm_state2:l2}
            else:
                train_dict = {image_input:rand_x, input0e:new_e[0], input1e:new_e[1], input2e:new_e[2],
                            input0r:new_r[0], input1r:new_r[1], input2r:new_r[2], y_target:rand_y,
                            lstm_state0:new_l[0], lstm_state1:new_l[1], lstm_state2:new_l[2]}
        elif t==0:
            train_dict = {image_input:rand_x, input0e:e0_0, input1e:e0_1, input2e:e0_2, input0r:r0_0, input1r:r0_1, input2r:r0_2, y_target:rand_y}
        else:
            train_dict = {image_input:rand_x, input0e:new_e[0], input1e:new_e[1], input2e:new_e[2], input0r:new_r[0], input1r:new_r[1], input2r:new_r[2], y_target:rand_y}
        new_e = []
        new_r = []
        new_e.append(sess.run(new_error[0], feed_dict=train_dict))
        new_e.append(sess.run(new_error[1], feed_dict=train_dict))
        new_e.append(sess.run(new_error[2], feed_dict=train_dict))
        new_r.append(sess.run(new_represent[0], feed_dict=train_dict))
        new_r.append(sess.run(new_represent[1], feed_dict=train_dict))
        new_r.append(sess.run(new_represent[2], feed_dict=train_dict))
        if ifLSTM and ifGlobalLSTM:
            new_l = []
            new_l.append(sess.run(new_lstm_state[0], feed_dict=train_dict))
            new_l.append(sess.run(new_lstm_state[1], feed_dict=train_dict))
            new_l.append(sess.run(new_lstm_state[2], feed_dict=train_dict))

        sess.run(train_step, feed_dict=train_dict)
        temp_train_loss = sess.run(loss, feed_dict=train_dict)

    if (i+1) % interval == 0:
        if DEBUG:
            TensorMonitor.Beat(sess, input=train_dict)
        acc_and_loss = [(i+1), temp_train_loss]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss: {:.2f} .'.format(*acc_and_loss))
        train_loss.append(temp_train_loss)

        train_prediction = sess.run(train_pred, feed_dict=train_dict)
        train_acc = get_accuracy(train_prediction, rand_y)
        train_pred_plot.append(train_acc)

###########################################################################################################
if PLOT:
    if TEST is False:
        eval_indices = range(0, generations, interval)
        plt.plot(eval_indices, train_pred_plot, 'r--')
        plt.title('Train accuracy per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.show()

    else:
        eval_indices = range(0, generations, interval)
        try:
            plt.plot(eval_indices, test_accuracy, 'r--')
        except ValueError:
            pass
        plt.title('Test Accuracy per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.show()
