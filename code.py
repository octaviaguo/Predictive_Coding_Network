import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

sess = tf.Session()

import sys
# root path of tensor_manager
sys.path.append("../../")
from TensorMonitor.tensor_manager import TensorMonitor

DEBUG = False
PLOT = True
is_train = True
TEST = True
#####################   get input   ######################

data_dir = 'temp'
mnist = input_data.read_data_sets(data_dir, one_hot=False)
# Convert images into 28x28 (they are downloaded as 1x784)
train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])
train_labels = mnist.train.labels
test_labels = mnist.test.labels

####################    model setting   #################

batch_size = 20
test_batch_size = batch_size
learning_rate = 0.004
generations = 300
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
channel_num = 1
input_shape = (batch_size, image_width, image_height, channel_num)
conv1_features = 20
conv2_features = 40
conv3_features = 60
#fully_connected_size1 = 100

target_size = np.max(train_labels) + 1
y_target = tf.placeholder(tf.int32, shape=(batch_size))
if TEST:
    test_target = tf.placeholder(tf.int32, shape=(test_batch_size))
interval = 10
control = True             # if control = True, ff and fb weight are transposed;

#######################################################
#   training tuning factor
fb_weight = 1.0
PcnStep_num = 4
#######################################################

class PcnStep:
    ######   static variables in class   ##########
    global channel_num, conv1_features, conv2_features, conv3_features, fb_weight
    conv1_weight = tf.Variable(tf.truncated_normal([4, 4, channel_num, conv1_features], stddev=0.1, dtype=tf.float32))
    conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))
    if control:
        fb1_weight = conv1_weight
    else:
        fb1_weight = tf.Variable(tf.truncated_normal([4, 4, channel_num, conv1_features], stddev=0.1, dtype=tf.float32))
    fb1_bias = tf.Variable(tf.zeros([channel_num], dtype=tf.float32))
    update_ratea1 = tf.Variable([0.5], dtype=tf.float32)

    conv2_weight = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features], stddev=0.1, dtype=tf.float32))
    conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))
    if control:
        fb2_weight = conv2_weight
    else:
        fb2_weight = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features], stddev=0.1, dtype=tf.float32))
    fb2_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))
    update_rateb2 = tf.Variable([0.5], dtype=tf.float32)
    update_ratea2 = tf.Variable([0.5], dtype=tf.float32)

    conv3_weight = tf.Variable(tf.truncated_normal([4, 4, conv2_features, conv3_features], stddev=0.1, dtype=tf.float32))
    conv3_bias = tf.Variable(tf.zeros([conv3_features], dtype=tf.float32))
    if control:
        fb3_weight = conv3_weight
    else:
        fb3_weight = tf.Variable(tf.truncated_normal([4, 4, conv2_features, conv3_features], stddev=0.1, dtype=tf.float32))
    fb3_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))
    update_rateb3 = tf.Variable([0.5], dtype=tf.float32)
    update_ratea3 = tf.Variable([0.5], dtype=tf.float32)

    def __init__(self, time0, input, preTimeInput = [], last = False):
        global input_shape
        self.input = input
        self.inputt = tf.layers.batch_normalization(self.input, training=is_train)
        self.ifTime0 = time0
        self.last = last
        self.first = time0
        # layer state
        self.pool1 = []
        self.pool2 = []
        self.pool3 = []

        if time0:
            self.convLayer()
        else:                                                           # flag if time is 0
            self.pool1 = [preTimeInput[0]]
            self.pool2 = [preTimeInput[1]]
            self.pool3 = [preTimeInput[2]]

    def convLayer(self):
        # layer 1
        conv1 = tf.nn.conv2d(self.inputt, self.conv1_weight, strides=[1, 1, 1, 1], padding='SAME', name='init_conv1')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.conv1_bias), name='init_relu1')
        max_pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        max_pool1 = tf.layers.batch_normalization(max_pool1, training=is_train, name='init_maxpool1')
        self.pool1.append(max_pool1)
        # layer 2
        conv2 = tf.nn.conv2d(max_pool1, self.conv2_weight, strides=[1, 1, 1, 1], padding='SAME', name='init_conv2')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_bias), name='init_relu2')
        max_pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        max_pool2 = tf.layers.batch_normalization(max_pool2, training=is_train, name='init_maxpool2')
        self.pool2.append(max_pool2)
        # layer 3
        conv3 = tf.nn.conv2d(max_pool2, self.conv3_weight, strides=[1, 1, 1, 1], padding='SAME', name='init_conv3')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, self.conv3_bias), name='init_relu3')
        max_pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        max_pool3 = tf.layers.batch_normalization(max_pool3, training=is_train, name='init_maxpool3')
        self.pool3.append(max_pool3)

    def addFeedback(self):
        #####################   Feedback    #####################
        # from layer 3 to 2
        f3 = tf.nn.conv2d_transpose(self.pool3[-1], self.fb3_weight, self.pool2[-1].get_shape().as_list(), strides=[1, 2, 2, 1], padding='SAME', name='fb3_2')
        p2 = tf.nn.relu(tf.nn.bias_add(f3, self.fb3_bias), name='predict3_2') #pred of 2
        self.update_rateb3 = tf.nn.relu(self.update_rateb3, name='update_rateb3')
        self.pool2.append(tf.nn.relu(tf.add( (1-self.update_rateb3)*self.pool2[-1], p2*self.update_rateb3)))
        # from layer 2 to 1
        self.pool2.append(tf.layers.batch_normalization(self.pool2[-1], training=is_train))
        f2 = tf.nn.conv2d_transpose(self.pool2[-1], self.fb2_weight, self.pool1[-1].get_shape().as_list(), strides=[1, 2, 2, 1], padding='SAME', name='fb2_1')
        p1 = tf.nn.relu(tf.nn.bias_add(f2, self.fb2_bias), name='predict2_1') #pred of 1
        self.update_rateb2 = tf.nn.relu(self.update_rateb2, name='update_rateb2')
        self.pool1.append(tf.nn.relu(tf.add( (1-self.update_rateb2)*self.pool1[-1], p1*self.update_rateb2)))
        # from layer 1 to input
        self.pool1.append(tf.layers.batch_normalization(self.pool1[-1], training=is_train))
        f1 = tf.nn.conv2d_transpose(self.pool1[-1], self.fb1_weight, input_shape, strides=[1, 2, 2, 1], padding='SAME', name='fb1_0') #pred of input
        predInput = tf.nn.relu(tf.nn.bias_add(f1, self.fb1_bias), name='predict_input')

        #################### Feedforward to update ###########################
        # update layer 1
        e0 = self.inputt-predInput
        c1 = tf.nn.conv2d(e0, self.conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
        r1 = tf.nn.relu(tf.nn.bias_add(c1, self.conv1_bias))
        ff1 = tf.nn.max_pool(r1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.update_ratea1 = tf.nn.relu(self.update_ratea1, name='update_ratea1')
        nextStep1 = tf.nn.relu(tf.add(self.pool1[-1], self.update_ratea1*ff1*fb_weight))
        nextStep1 = tf.layers.batch_normalization(nextStep1, training=is_train)
        self.pool1.append(nextStep1)
        # update layer 2
        e1 = nextStep1-p1
        c2 = tf.nn.conv2d(e1, self.conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
        r2 = tf.nn.relu(tf.nn.bias_add(c2, self.conv2_bias))
        ff2 = tf.nn.max_pool(r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.update_ratea2 = tf.nn.relu(self.update_ratea2, name='update_ratea2')
        nextStep2 = tf.nn.relu(tf.add(self.pool2[-1], self.update_ratea2*ff2*fb_weight))
        nextStep2 = tf.layers.batch_normalization(nextStep2, training=is_train)
        self.pool2.append(nextStep2)
        # update layer 3
        e2 = nextStep2-p2
        c3 = tf.nn.conv2d(e2, self.conv3_weight, strides=[1, 1, 1, 1], padding='SAME')
        r3 = tf.nn.relu(tf.nn.bias_add(c3, self.conv3_bias))
        ff3 = tf.nn.max_pool(r3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.update_ratea3 = tf.nn.relu(self.update_ratea3, name='update_ratea3')
        nextStep3 = tf.nn.relu(tf.add(self.pool3[-1], self.update_ratea3*ff3*fb_weight))
        nextStep3 = tf.layers.batch_normalization(nextStep3, training=is_train)
        self.pool3.append(nextStep3)

        if DEBUG and self.last is True:
            TensorMonitor.AddUserList(
                last_predInput=predInput,
                last_e0 = e0,
                last_e1 = e1,
                last_e2 = e2,
                last_p1 = p1,
                last_p2 = p2,
                image = self.inputt,
                b2 = self.update_rateb2,
                b3 = self.update_rateb3,
                a1 = self.update_ratea1,
                a2 = self.update_ratea2,
                a3 = self.update_ratea3
                )


        if DEBUG and self.first is True:
            TensorMonitor.AddUserList(
                first_predInput=predInput,
                first_e0 = e0,
                first_e1 = e1,
                first_e2 = e2
                )

        return [nextStep1, nextStep2, nextStep3]

    def get_output(self):
        if self.last:
            # fully connected layer
            final_conv_shape = self.pool3[-1].get_shape().as_list()
            final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
            flat_output = tf.reshape(self.pool3[-1], [final_conv_shape[0], final_shape])

            global target_size
            self.full1_weight = tf.Variable(tf.truncated_normal([final_shape, target_size], stddev=0.1, dtype=tf.float32))
            self.full1_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))
            output = tf.add(tf.matmul(flat_output, self.full1_weight), self.full1_bias)
            return output
        else:
            pass



##################     get accuracy of test data    #####################
def get_accuracy(current, target):
    prediction = np.argmax(current, axis=1)
    num_correct = np.sum(np.equal(prediction, target))
    return 100. * num_correct/prediction.shape[0]
######################      build network    ####################

image_input = tf.placeholder(tf.float32, shape=input_shape)
#image_input = tf.layers.batch_normalization(image_input, training=is_train)
if PcnStep_num > 1:
    count = 0
    transfer = []
    while count < PcnStep_num:
        if count==0:
            t0 = PcnStep(True, image_input)
            transfer = t0.addFeedback()
        elif count == (PcnStep_num-1):
            t_last = PcnStep(False, image_input, transfer, True)
            transfer = t_last.addFeedback()
            model_output = t_last.get_output()
        else:
            ti = PcnStep(False, image_input, transfer)
            transfer = ti.addFeedback()
        count = count +1

elif PcnStep_num==1:
    t = PcnStep(time0=True, input=image_input, last=True)
    model_output = t.get_output()

if TEST:
    test_prediction = tf.nn.softmax(model_output)


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
        train_dict = {t0.input: rand_x, y_target: rand_y}
        sess.run(train_step, feed_dict=train_dict)
        temp_train_loss = sess.run(loss, feed_dict=train_dict)

    if (i+1) % interval == 0:
        if DEBUG and jump_train is False:
            TensorMonitor.Beat(sess, input=train_dict)

        if TEST:
            test_index = np.random.choice(len(test_xdata), size=test_batch_size)
            test_x = test_xdata[test_index]
            test_x = np.expand_dims(test_x, 3)
            test_y = test_labels[test_index]
            test_dict = {t0.input: test_x, test_target: test_y}
            test_predict = sess.run(test_prediction, feed_dict=test_dict)
            temp_test_acc = get_accuracy(test_predict, test_y)
            #print("test acc", temp_test_acc)
            test_accuracy.append(temp_test_acc)

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
        try:
            plt.plot(eval_indices, train_loss, 'k-')
        except ValueError:
            pass
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
