import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import sys
# root path of tensor_manager
sys.path.append("../../")
from TensorMonitor.tensor_manager import TensorMonitor

sess = tf.Session()

#####################   User Setting    ##################
DEBUG = True
PLOT = False
TEST = False
is_train = True
control = True             # if control = True, ff and fb weight are transposed;

#####################   get input   ######################
data_dir = 'temp'
mnist = input_data.read_data_sets(data_dir, one_hot=False)
train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])
train_labels = mnist.train.labels
test_labels = mnist.test.labels

####################    model setting   #################
batch_size = 20
test_batch_size = 20
learning_rate = 0.004
generations = 5000
interval = 10
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
channel_num = 1
input_shape = (batch_size, image_width, image_height, channel_num)
target_size = np.max(train_labels) + 1
y_target = tf.placeholder(tf.int32, shape=(batch_size))
fb_factor = 1.0


conv_features = [20, 40, 60]
cycle_num = 8
level_num = 4              # if change level_num (including input image), also change conv_features and below variables setting;

conv1_weight = tf.Variable(tf.truncated_normal([3, 3, channel_num, conv_features[0]], stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros([conv_features[0]], dtype=tf.float32))
bp1_weight = tf.Variable(tf.truncated_normal([1, 1, channel_num, conv_features[0]], stddev=0.1, dtype=tf.float32))
bp1_bias = tf.Variable(tf.zeros([conv_features[0]], dtype=tf.float32))
if control:
    fb1_weight = conv1_weight
else:
    fb1_weight = tf.Variable(tf.truncated_normal([3, 3, channel_num, conv_features[0]], stddev=0.1, dtype=tf.float32))
fb1_bias = tf.Variable(tf.zeros([channel_num], dtype=tf.float32))
update_ratea1 = tf.Variable([0.5], dtype=tf.float32)

conv2_weight = tf.Variable(tf.truncated_normal([3, 3, conv_features[0], conv_features[1]], stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv_features[1]], dtype=tf.float32))
bp2_weight = tf.Variable(tf.truncated_normal([1, 1, conv_features[0], conv_features[1]], stddev=0.1, dtype=tf.float32))
bp2_bias = tf.Variable(tf.zeros([conv_features[1]], dtype=tf.float32))
if control:
    fb2_weight = conv2_weight
else:
    fb2_weight = tf.Variable(tf.truncated_normal([3, 3, conv_features[0], conv_features[1]], stddev=0.1, dtype=tf.float32))
fb2_bias = tf.Variable(tf.zeros([conv_features[0]], dtype=tf.float32))
update_ratea2 = tf.Variable([0.5], dtype=tf.float32)

conv3_weight = tf.Variable(tf.truncated_normal([3, 3, conv_features[1], conv_features[2]], stddev=0.1, dtype=tf.float32))
conv3_bias = tf.Variable(tf.zeros([conv_features[2]], dtype=tf.float32))
bp3_weight = tf.Variable(tf.truncated_normal([1, 1, conv_features[1], conv_features[2]], stddev=0.1, dtype=tf.float32))
bp3_bias = tf.Variable(tf.zeros([conv_features[2]], dtype=tf.float32))
if control:
    fb3_weight = conv3_weight
else:
    fb3_weight = tf.Variable(tf.truncated_normal([3, 3, conv_features[1], conv_features[2]], stddev=0.1, dtype=tf.float32))
fb3_bias = tf.Variable(tf.zeros([conv_features[1]], dtype=tf.float32))
update_ratea3 = tf.Variable([0.5], dtype=tf.float32)

conv_weight = [0, conv1_weight, conv2_weight, conv3_weight]
conv_bias = [0, conv1_bias, conv2_bias, conv3_bias]
fb_weight = [0, fb1_weight, fb2_weight, fb3_weight]
fb_bias = [0, fb1_bias, fb2_bias, fb3_bias]
update_a = [0, update_ratea1, update_ratea2, update_ratea3]
bp_weight = [0, bp1_weight, bp2_weight, bp3_weight]
bp_bias = [0, bp1_bias, bp2_bias, bp3_bias]

##########################################################

class localPCN:

    def __init__(self, input):
        self.input = input
        self.pool = [self.input]                         # level 0 is the input image
        p = []
        e = []
        for i in range(1, level_num):
            self.pool.append([])
            p.append([])
            e.append([])

        self.bn = [[]]                                   # bn[i] is the input of level i
        self.conv = [[]]                                 # conv[i] is the output of level i
        self.relu = [[]]
        self.full1_weight = 0
        self.full1_bias = 0


        for i in range(1, level_num):
            if i==1:
                self.bn.append(tf.layers.batch_normalization(self.pool[0], training=is_train))
            else:
                self.bn.append(tf.layers.batch_normalization(self.pool[i-1][-1], training=is_train))
            self.conv.append([])
            self.relu.append([])

            self.conv[i].append(tf.nn.conv2d(self.bn[i], conv_weight[i], strides=[1, 1, 1, 1], padding='SAME'))
            self.relu[i].append(tf.nn.relu(tf.nn.bias_add(self.conv[i][-1], conv_bias[i])))
            self.pool[i].append(tf.nn.max_pool(self.relu[i][-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))

            for t in range(1, cycle_num+1):
                if i==1:
                    f1 = tf.layers.batch_normalization(self.pool[i][-1], training=is_train)
                    f1 = tf.nn.conv2d_transpose(f1, fb_weight[i], input_shape, strides=[1, 2, 2, 1], padding='SAME')
                else:
                    f1 = tf.layers.batch_normalization(self.pool[i][-1], training=is_train)
                    f1 = tf.nn.conv2d_transpose(f1, fb_weight[i], self.pool[i-1][-1].get_shape().as_list(), strides=[1, 2, 2, 1], padding='SAME')
                p[i-1] = tf.nn.bias_add(f1, fb_bias[i])
                e[i-1] = tf.nn.relu(self.bn[i]-p[i-1])
                e[i-1] = tf.layers.batch_normalization(e[i-1], training=is_train)
                update_a[i] = tf.nn.relu(update_a[i])
                self.conv[i].append(tf.nn.conv2d(e[i-1], conv_weight[i], strides=[1, 1, 1, 1], padding='SAME'))
                self.relu[i].append(tf.nn.relu(tf.nn.bias_add(self.conv[i][-1], conv_bias[i])))
                ff1 = tf.nn.max_pool(self.relu[i][-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                self.pool[i].append(tf.add(self.pool[i][-1], update_a[i]*ff1*fb_factor))

            bypass = tf.nn.conv2d(self.bn[i], bp_weight[i], strides=[1, 2, 2, 1], padding='SAME')
            byp = tf.nn.relu(tf.nn.bias_add(bypass, bp_bias[i]))
            self.pool[i].append(tf.add(self.pool[i][-1], byp))


        if DEBUG:
            TensorMonitor.AddUserList(
                last_predInput = p[0],
                last_error = e[0],
                batch_norm_input = self.bn[1]
                )


    def get_output(self):
        # fully connected network
        self.pool[-1][-1] = tf.layers.batch_normalization(self.pool[-1][-1], training=is_train)
        final_conv_shape = self.pool[-1][-1].get_shape().as_list()
        final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
        flat_output = tf.reshape(self.pool[-1][-1], [final_conv_shape[0], final_shape])

        self.full1_weight = tf.Variable(tf.truncated_normal([final_shape, target_size], stddev=0.1, dtype=tf.float32))
        self.full1_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))
        output = tf.add(tf.matmul(flat_output, self.full1_weight), self.full1_bias)
        return output


#######################     build network     #############################

image_input = tf.placeholder(tf.float32, shape=input_shape)
pcn_network = localPCN(image_input)
model_output = pcn_network.get_output()
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))
optimization = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = optimization.minimize(loss)

###################        train            #####################

init = tf.global_variables_initializer()
sess.run(init)
train_loss = []
test_accuracy = []
jump_train = False

for i in range(generations):
    if jump_train is False:
        rand_index = np.random.choice(len(train_xdata), size=batch_size)
        rand_x = train_xdata[rand_index]
        rand_x = np.expand_dims(rand_x, 3)
        rand_y = train_labels[rand_index]
        train_dict = {pcn_network.input: rand_x, y_target: rand_y}
        sess.run(train_step, feed_dict=train_dict)
        temp_train_loss = sess.run(loss, feed_dict=train_dict)

    if (i+1) % interval == 0:
        if DEBUG and jump_train is False:
            TensorMonitor.Beat(sess, input=train_dict)

        if jump_train is False:
            acc_and_loss = [(i+1), temp_train_loss]
            acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
            print('Generation # {}. Train Loss: {:.2f} .'.format(*acc_and_loss))
            train_loss.append(temp_train_loss)
            if temp_train_loss < 0.01:
                jump_train = True


#################       draw pic        ######################

if PLOT:
    if TEST is False:
        eval_indices = range(0, generations, interval)
        plt.plot(eval_indices, train_loss, 'k-')

        plt.title('Softmax Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Softmax Loss')
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
