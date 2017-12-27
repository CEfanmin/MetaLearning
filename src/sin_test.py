import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sinData import DataGenerator
import random


def sin_test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    # load ground truth   
    x = np.arange(-5.0, 5.0, 0.01)
    y = np.sin(x)
    
    num_classes = 1 
    np.random.seed(1)    
    random.seed(1)
    pre_y10, pre_y1, pre_y0 = [], [], []
    pre_update = []
    inputs = np.array(x).reshape(1, len(x), 1)
    label = np.array(y).reshape(1,len(y),1)

    feed_dict = {model.inputa:inputs, model.inputb:inputs, model.labela:label, model.labelb:label}
    result, result0, result1, result10 = sess.run([model.outputas, model.outputbs[0], model.outputbs[1], model.outputbs[-1]], feed_dict)
    print("ooutputbs len is: ", len(model.outputbs))
    pre_y10.extend(np.array(result10).reshape(len(x),))
    pre_y1.extend(np.array(result1).reshape(len(x),1))
    pre_y0.extend(np.array(result0).reshape(len(x),1))
    pre_update.extend(np.array(result).reshape(len(x),1))
    # plot the pre-y curve 
    plt.figure()
    plt.title("test the model with MAML")
    plt.plot(x, y, 'r-', label="ground truth")
    plt.plot(x, pre_y10, 'g--',label="10-grad steps")
    plt.plot(x, pre_y1, 'b--', label="2-grad step")
    plt.plot(x, pre_y0, 'b-', label="1-grad step")
    plt.plot(x, pre_update, 'y--',label="pre-update")
    plt.legend(loc="upper right")
    plt.show()

    ## pretrained and pre-update
    # result = sess.run(model.outputas, feed_dict)
    # pre_update.extend(np.array(result).reshape(len(x),1))
    # plt.figure()
    # plt.title("test the model with no MAML")
    # plt.plot(x, y, 'r-', label="ground truth")
    # plt.plot(x, pre_update, 'y--',label="pre-update" )
    # plt.legend(loc="upper right")
    # plt.show()

