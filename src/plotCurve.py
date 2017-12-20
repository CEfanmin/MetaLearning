__author__ = 'fanmin'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plotData(filename, sensorlist):
	ahrs_data = pd.read_table(filename)
	for sensor in sensorlist:
		sensor = ahrs_data[sensor].iloc[0:2000]
		epochs = range(0, len(sensor))
		# plt.figure()
		plt.plot(epochs, sensor, 'g')
		plt.title(sensor)
		plt.legend()
		plt.show()

plotData('./data/ahrs', ['roll_degree ', 'pitch_degree '])

