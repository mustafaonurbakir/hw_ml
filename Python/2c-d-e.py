"""
Mustafa Onur Bakir
150130059

Homework 1
Part 2-c-d-e

This program tested on Python 3.7.0

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#this function split datas according to label
def split_classes(datas, classes):
	for i in classes:
		feat1 = [datas["feat1"][x] for x in range(len(datas["feat1"])) if datas["label"][x] == i]
		feat2 = [datas["feat2"][x] for x in range(len(datas["feat2"])) if datas["label"][x] == i]
		label = [datas["label"][x] for x in range(len(datas["label"])) if datas["label"][x] == i]

		yield np.array(feat1), np.array(feat2), np.array(label)


#this function read datas from file
def read_file(file_name_train, file_name_test):

	#second line to last line
	def convert_lines(lines):
		#empty numpy arrays
		feat1 = np.empty(shape=[0,1])
		feat2 = np.empty(shape=[0,1])
		label = np.empty(shape=[0,1])
		
		for counter in range(1, len(lines)):
			#split data
			temp_feat1, temp_feat2, temp_label = lines[counter].split("\t")

			#in the file there is blank lines. For avoiding from an error
			if len(temp_feat1) != 0 or len(temp_feat2) != 0 or len(temp_label):
				feat1 = np.append(feat1, float(temp_feat1))
				feat2 = np.append(feat2, float(temp_feat2))
				label = np.append(label, float(temp_label))
		
		return feat1, feat2, label

	#file open and take lines
	f = open(file_name_train, "r")
	lines = f.readlines()
	f.close()
	
	feat1, feat2, label = convert_lines(lines)
	datas_train = pd.DataFrame()
	datas_train["feat1"] = feat1
	datas_train["feat2"] = feat2
	datas_train["label"] = label
	
	#test file
	f = open(file_name_test, "r")
	lines = f.readlines()
	f.close()
	
	feat1, feat2, label = convert_lines(lines)
	datas_test = pd.DataFrame()
	datas_test["feat1"] = feat1
	datas_test["feat2"] = feat2
	datas_test["label"] = label
	
	return datas_train, datas_test
	


	


if __name__ == "__main__":
	file_name = "classification_train.txt"
	file_name_test = "classification_test.txt"
	
	exists = os.path.isfile(file_name)
	exists2 = os.path.isfile(file_name_test)
	if not exists or not exists2:
		print( "File not found!")
		exit()
	
	#read the data from file
	datas_train, datas_test = read_file(file_name, file_name_test)

	#determine the classes
	classes = np.unique(datas_train["label"])
	print("classes: ", classes)

	#split_classes function split all possible classes and return seperate numpy array for all
	for feat1_val, feat2_val, label_val in split_classes(datas_train, classes):
		print("\nMean vector for label: ", label_val[0], "\n", np.sum(feat1_val) / len(feat1_val), np.sum(feat2_val) / len(feat1_val))
		print("\nCovariance vector for label: ", label_val[0], "\n", np.cov([feat1_val, feat2_val]))
	
	#model = LDA(n_components=2)
	model = LDA()
	X_lda = model.fit(np.array(list(zip(datas_train["feat1"],datas_train["feat2"]))), datas_train["label"])
	print(X_lda)
	
	prediction = model.predict(np.array(list(zip(datas_test["feat1"],datas_test["feat2"]))))
	
	true_pred=0
	false_pred=0
	for i in range(len(datas_test["feat1"])):
		print("feat1: ", datas_test["feat1"][i], "\tfeat2: ", datas_test["feat2"][i], "\tlabel: ", datas_test["label"][i], "\tprediction: ", prediction[i])
		if (datas_test["label"][i] == prediction[i]): true_pred += 1 
		else: false_pred += 1
	
	print("\n#True_prediction\t: ", true_pred)
	print("#False_prediction\t: ", false_pred)
	print("Accuracy\t\t: ", true_pred / len(datas_test["feat1"]))
		
	
	
	
	
	
