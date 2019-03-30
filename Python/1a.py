"""
Mustafa Onur Bakır
150130059

Homework 1
Part1-a
Part1-b

This file include the solution of the Part1-a (Lineer Regression with Gradiant Descent) and
solution of the Part1-b (Cross Validation)

This program tested on Python 3.7.0

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


#this function calculate the regression
def gradient_descent(head_size, brain_size, iterations, learning_rate):
	m_val = 0
	c_val = 0
	N = len(head_size)

	for i in range(int(iterations)):
		#last prediction
		brain_size_predicted = head_size * m_val + c_val

		#take derivative like formula
		m_derivative = -1 * (2 / N) * sum(head_size * (brain_size - brain_size_predicted))
		c_derivative = -1 * (2 / N) * sum(brain_size - brain_size_predicted)

		#update the value of m and c
		m_val -= learning_rate * m_derivative
		c_val -= learning_rate * c_derivative

	return m_val, c_val


#This function plot the datas
def plot_regression_line(head_size, brain_size, lines):

	#points from data, b=blue, s=size
	plt.scatter(head_size, brain_size, color = "b", s=10)

	#plot the all lines from k-cross fold
	for i in range(len(lines)):
		#plot the prediction line
		brain_size_prediction = lines[0]*head_size + lines[1]
		plt.plot(head_size, brain_size_prediction)

	#Labels
	plt.xlabel('Head Size')
	plt.ylabel('Brain Size')

	plt.show()


#this function read datas from file
def read_file(file_name):
	#file open and take lines
	f = open(file_name, "r")
	lines = f.readlines()

	#empty numpy arrays
	head_size = np.empty(shape=[0,1])
	brain_size = np.empty(shape=[0,1])

	#second line to last line
	for counter in range(1, len(lines)):
		#split data
		temp_head_size, temp_brain_size = lines[counter].split("\t\t\t")

		#in the file there is blank lines. For avoiding from an error
		if len(temp_brain_size) != 0 or len(temp_head_size) != 0:
			head_size = np.append(head_size, int(temp_head_size))
			brain_size = np.append(brain_size, int(temp_brain_size))

	datas = pd.DataFrame()
	datas["head_size"] = head_size
	datas["brain_size"] = brain_size
	return datas


if __name__ == "__main__":
	file_name = "regression_data.txt"
	K = 5 #for k-fold cross val

	#read the data from file
	datas = read_file(file_name)

	#calculate regresion via gradient method
	m, c = gradient_descent(datas["head_size"], datas["brain_size"], 1000, 0.000000001)

	print("m value(slope) \t\t: ", m)
	print("c value(constanst)\t: ", c)

	#plot the values and lines
	plot_regression_line(datas["head_size"], datas["brain_size"], (m,c))