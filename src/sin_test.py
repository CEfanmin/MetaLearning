import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sinData import DataGenerator
import random


def sin_test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    # load ground truth   
    x = np.arange(-5.0, 5.0, 0.001)
    y = np.sin(x)

    num_classes = 1 
    np.random.seed(1)    
    random.seed(1)
    pre_y = []
    inputs = np.array(x).reshape(1, len(x), 1)
    label = np.array(y).reshape(1,len(y),1)
    feed_dict = {model.inputa:inputs, model.inputb:inputs, model.labela:label, model.labelb:label}
    result = sess.run(model.outputbs[-1], feed_dict)
    pre_y.extend(np.array(result).reshape(len(x),))

    # plot the pre-y curve 
    plt.figure()
    plt.title("test the model")
    plt.plot(x, y, 'r-', label="ground truth")
    plt.plot(x, pre_y, 'g--',label="pre_y" )
    plt.legend(loc="upper right")
    plt.show()



