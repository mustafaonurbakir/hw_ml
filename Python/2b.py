"""
Mustafa Onur Bakır
150130059

Homework 1
Part 2-b

This program tested on Python 3.7.0

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def split_classes(datas, classes):
	for i in classes:
		feat1 = [datas["feat1"][x] for x in range(len(datas["feat1"])) if datas["label"][x] == i]
		feat2 = [datas["feat2"][x] for x in range(len(datas["feat2"])) if datas["label"][x] == i]
		label = [datas["label"][x] for x in range(len(datas["label"])) if datas["label"][x] == i]

		yield np.array(feat1), np.array(feat2), np.array(label)


#This function plot the datas
def plot_regression_line(head_size, brain_size):

	#points from data, b=blue, s=size
	plt.scatter(head_size, brain_size, color = "b", s=10)

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

	#read the data from file
	datas = read_file(file_name)

	#determine the classes
	classes = np.unique(datas["label"])
	print("classes: ", classes)

	#split_classes function split all possible classes and return seperate numpy array for all
	for feat1_val, feat2_val, label_val in split_classes(datas, classes):
		print("\nMean vector for label: ", label_val[0], "\n", np.sum(feat1_val) / len(feat1_val), np.sum(feat2_val) / len(feat1_val))
		print("\nCovariance vector for label: ", label_val[0], "\n", np.cov([feat1_val, feat2_val]))

	#plot the values and lines
	plot_regression_line(datas["feat1"], datas["feat2"])
