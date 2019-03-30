"""
Mustafa Onur Bakır
150130059

Homework 1
Part2-a

This program take datas from "classification_train.txt" and plot them.

This program tested on Python 3.7.0

"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 


#This function plot the datas
def plot_regression_line(head_size, brain_size):

	#points from data, b=blue, s=size
	plt.scatter(head_size, brain_size, color = "b", s=10)

	#Labels
	plt.xlabel('feature1')
	plt.ylabel('feature2')

	plt.show()


#this function read datas from file
def read_file(file_name):
	#file open and take lines
	f = open(file_name, "r")
	lines = f.readlines()

	#empty numpy arrays
	feat1 = np.empty(shape=[0,1])
	feat2 = np.empty(shape=[0,1])
	label = np.empty(shape=[0,1])

	#second line to last line
	for counter in range(1, len(lines)):
		#split data
		temp_feat1, temp_feat2, temp_label = lines[counter].split("\t")

		#in the file there is blank lines. For avoiding from an error
		if len(temp_feat1) != 0 or len(temp_feat2) != 0 or len(temp_label):
			feat1 = np.append(feat1, float(temp_feat1))
			feat2 = np.append(feat2, float(temp_feat2))
			label = np.append(label, float(temp_label))

	datas = pd.DataFrame()
	datas["feat1"] = feat1
	datas["feat2"] = feat2
	datas["label"] = label
	#print (datas)
	return datas


if __name__ == "__main__":
	file_name = "classification_train.txt"
	if not os.path.isfile(file_name):
		print( "File not found!")
		exit()

	#read the data from file
	datas = read_file(file_name)

	#determine the classes
	classes = np.unique(datas["label"])
	print("classes: ", classes)

	#count the classes
	how_many = [0 for i in range(len(classes))]
	
	for j in range(len(classes)):
		for i in datas["label"]:
			if i == classes[j]:
				how_many[j] += 1
	print("how_many: ", how_many)

	#print the possibility
	for i in range(len(classes)):
		print("Class: ", classes[i], " -> Probability: ", how_many[i]/sum(how_many))

	#plot the values and lines
	plot_regression_line(datas["feat1"], datas["feat2"])
