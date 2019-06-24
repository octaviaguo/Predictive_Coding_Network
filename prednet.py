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
ifGlobalLSTM = False       # if True, the state of LSTM will be across the global "cycle"
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
generations = 5000
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
#convLSTM_cycle = 1
#cycle_num = 12
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
#update_ratea1 = tf.Variable([0.5], dtype=tf.float32)
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
#update_ratea2 = tf.Variable([0.5], dtype=tf.float32)
conv3_weight = tf.Variable(tf.truncated_normal([3, 3, conv_features[1], conv_features[2]], stddev=0.1, dtype=tf.float32))
conv3_bias = tf.Variable(tf.zeros([conv_features[2]], dtype=tf.float32))
conv_lstm3 = tf.Variable(tf.truncated_normal([3, 3, 2*conv_features[1], conv_features[1]], stddev=0.1, dtype=tf.float32))
conv_lstm_bias3 = tf.Variable(tf.zeros([conv_features[1]], dtype=tf.float32))
conv_abar_weight3 = tf.Variable(tf.truncated_normal([3, 3, conv_features[1], conv_features[1]], stddev=0.1, dtype=tf.float32))
conv_abar_bias3 = tf.Variable(tf.zeros([conv_features[1]], dtype=tf.float32))
#update_ratea3 = tf.Variable([0.5], dtype=tf.float32)


conv_weight = [conv1_weight, conv2_weight, conv3_weight]
conv_bias = [conv1_bias, conv2_bias, conv3_bias]
conv_lstm = [conv_lstm1, conv_lstm2, conv_lstm3]
conv_lstm_bias = [conv_lstm_bias1, conv_lstm_bias2, conv_lstm_bias3]
conv_abar_weight = [conv_abar_weight1, conv_abar_weight2, conv_abar_weight3]
conv_abar_bias = [conv_abar_bias1, conv_abar_bias2, conv_abar_bias3]
fb_weight = [fb1_weight, fb2_weight]
fb_bias = [fb1_bias, fb2_bias]

#######################################################################################################

class prednet:
    def __init__(self, input):
        e0_0 = tf.zeros(shape=input_shape, dtype=tf.float32)
        e0_1 = tf.zeros(shape=[batch_size, 14, 14, conv_features[0]], dtype=tf.float32)
        e0_2 = tf.zeros(shape=[batch_size, 7, 7, conv_features[1]], dtype=tf.float32)
        r0_0 = tf.zeros(shape=input_shape, dtype=tf.float32)
        r0_1 = tf.zeros(shape=[batch_size, 14, 14, conv_features[0]], dtype=tf.float32)
        r0_2 = tf.zeros(shape=[batch_size, 7, 7, conv_features[1]], dtype=tf.float32)

        self.input = input
        self.error = [ [e0_0, e0_1, e0_2] ]             # [time_0_all_level], [time_1_all_level]
        self.represent = [ [r0_0, r0_1, r0_2] ]
        self.a_bar = []
        self.a = []

        prevState = [[] for i in range(level_num)]
        for time in range(cycle_num):
            for l in reversed(range(level_num)):
                if ifLSTM:
                    tmp = 0
                    output_channels = 0

                    if l==level_num-1:
                        tmp = tf.concat([self.error[time][l], self.represent[time][l]], 3)
                        output_channels = conv_features[l-1]
                        self.represent.append([])
                    else:
                        up = tf.nn.conv2d_transpose(self.represent[-1][0], fb_weight[l], self.error[time][l].get_shape().as_list(), strides=[1, 2, 2, 1], padding='SAME')
                        upsample = tf.layers.batch_normalization(tf.nn.relu(tf.nn.bias_add(up, fb_bias[l])), training=is_train)
                        tmp = tf.concat([self.error[time][l], self.represent[time][l], upsample], 3)
                        if l==0:
                            output_channels = channel_num
                        else:
                            output_channels = conv_features[l-1]

                    if ifGlobalLSTM is False:
                        prevState[l] = []
                    final_lstm_output = 0

                    for t in range(convLSTM_cycle):
                        with tf.variable_scope("convLSTM"+"_level"+str(l)+"_"+str(time)+'_recur_'+str(t)):
                            cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=tmp.get_shape().as_list()[1:], output_channels=output_channels, kernel_shape=[3, 3])
                            #if t==0:
                            if len(prevState[l])==0:
                                prevState[l].append(cell.zero_state(batch_size=batch_size, dtype=tf.float32))
                            output, final_state = cell.call(inputs=tmp, state=prevState[l][-1])
                            prevState[l].append(final_state)

                            if t == convLSTM_cycle-1:
                                final_lstm_output = output

                        #output, final_state = tf.nn.dynamic_rnn(cell, inputtt, dtype=tf.float32, time_major=False, initial_state=initial_state)

                    self.represent[-1].insert(0, tf.layers.batch_normalization(final_lstm_output, training=is_train))

                elif l == level_num-1:
                    tmp = tf.concat([self.error[time][l], self.represent[time][l]], 3)
                    tmp2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tmp, conv_lstm[l], strides=[1, 1, 1, 1], padding='SAME'), conv_lstm_bias[l]))
                    self.represent.append([tf.layers.batch_normalization(tmp2, training=is_train)])

                else:
                    # upsample from higher level
                    up = tf.nn.conv2d_transpose(self.represent[-1][0], fb_weight[l], self.error[time][l].get_shape().as_list(), strides=[1, 2, 2, 1], padding='SAME')
                    upsample = tf.layers.batch_normalization(tf.nn.relu(tf.nn.bias_add(up, fb_bias[l])), training=is_train)
                    tmp = tf.concat([self.error[time][l], self.represent[time][l], upsample], 3)
                    tmp1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tmp, conv_lstm[l], strides=[1, 1, 1, 1], padding='SAME'), conv_lstm_bias[l]))
                    self.represent[-1].insert(0, tf.layers.batch_normalization(tmp1, training=is_train))

            for l in range(level_num):
                if l == 0:
                    tmp = tf.nn.conv2d(self.represent[-1][l], conv_abar_weight[l], strides=[1, 1, 1, 1], padding='SAME')
                    tmp1 = tf.nn.relu(tf.nn.bias_add(tmp, conv_abar_bias[l]), name='level0_pred_'+str(time))
                    tmp2 = tf.layers.batch_normalization(tmp1, training=is_train)
                    self.a_bar.append([tmp2])
                    if time==0:
                        tmp3 = tf.layers.batch_normalization(self.input, training=is_train, name='normalized_image_input')
                    else:
                        tmp3 = self.a[0][0]
                    self.a.append([tmp3])
                    self.error.append([])
                else:
                    tmp = tf.nn.conv2d(self.represent[-1][l], conv_abar_weight[l], strides=[1, 1, 1, 1], padding='SAME')
                    tmp1 = tf.nn.relu(tf.nn.bias_add(tmp, conv_abar_bias[l]), name='level'+str(l)+'_pred_'+str(time))
                    tmp2 = tf.layers.batch_normalization(tmp1, training=is_train)
                    self.a_bar[-1].append(tmp2)

                err = tf.subtract(self.a_bar[-1][-1],self.a[-1][-1], name='level'+str(l)+'_error_'+str(time))
                #self.error[-1].append(tf.nn.relu(err, name='level'+str(l)+'_error_relu'+str(time)))
                self.error[-1].append(tf.layers.batch_normalization(err, name='level'+str(l)+'_error_norm'+str(time)))

                if l < (level_num-1):
                    tmp1 = tf.nn.conv2d(self.error[-1][-1], conv_weight[l], strides=[1, 1, 1, 1], padding='SAME')
                    tmp2 = tf.nn.relu(tf.nn.bias_add(tmp1, conv_bias[l]))
                    tmp3 = tf.nn.max_pool(tmp2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='level'+str(l+1)+'_input_'+str(time+1))
                    self.a[-1].append(tf.layers.batch_normalization(tmp3, training=is_train))


    def get_output(self):
        tmp1 = tf.nn.conv2d(self.error[-1][-1], conv_weight[2], strides=[1, 1, 1, 1], padding='SAME')
        tmp2 = tf.nn.relu(tf.nn.bias_add(tmp1, conv_bias[2]))
        tmp3 = tf.nn.max_pool(tmp2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        final_conv_shape = tmp3.get_shape().as_list()
        final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
        flat_output = tf.reshape(tmp3, [final_conv_shape[0], final_shape])

        self.full1_weight = tf.Variable(tf.truncated_normal([final_shape, target_size], stddev=0.1, dtype=tf.float32))
        self.full1_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))
        output = tf.add(tf.matmul(flat_output, self.full1_weight), self.full1_bias)

        return output


    def get_level_error(self):
        error = 0
        for time in range(1, cycle_num):
            for level_error in self.error[time]:
                err_square = tf.pow(level_error, 2)
                z = tf.zeros_like(level_error, dtype=tf.float32)
                #sum = [level_error, z]
                sum = [err_square, z]
                error = error + tf.reduce_mean(sum)

        return error


###################################

def get_accuracy(current, target):
    prediction = np.argmax(current, axis=1)
    num_correct = np.sum(np.equal(prediction, target))
    return 100. * num_correct/prediction.shape[0]

#########################################################################

image_input = tf.placeholder(tf.float32, shape=input_shape)
pcn_network = prednet(image_input)
los1 = pcn_network.get_level_error()

model_output = pcn_network.get_output()
los2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target), name="final_label_loss")

loss = error_factor*los1 + los2

optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = optimizer.minimize(loss)

train_pred = tf.nn.softmax(model_output)

##########################################################################
init = tf.global_variables_initializer()
sess.run(init)
train_loss = []
temp_train_acc = []

for i in range(generations):
    rand_index = np.random.choice(len(train_xdata), size=batch_size)
    rand_x = train_xdata[rand_index]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = train_labels[rand_index]
    train_dict = {pcn_network.input: rand_x, y_target: rand_y}
    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss = sess.run(loss, feed_dict=train_dict)
    #temp_train_los = sess.run(loss, feed_dict=train_dict)

    if (i+1) % interval == 0:
        if DEBUG:
            TensorMonitor.Beat(sess, input=train_dict)
        acc_and_loss = [(i+1), temp_train_loss]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss: {:.2f} .'.format(*acc_and_loss))
        train_loss.append(temp_train_loss)
        train_prediction = sess.run(train_pred, feed_dict=train_dict)
        train_acc = get_accuracy(train_prediction, rand_y)
        temp_train_acc.append(train_acc)
        #if temp_train_loss < 0.01:
        #    break

#################       draw pic        ######################

if PLOT:
    if TEST is False:
        eval_indices = range(0, generations, interval)
        plt.plot(eval_indices, temp_train_acc, 'r--')

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
        plt.title('Test Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.show()
