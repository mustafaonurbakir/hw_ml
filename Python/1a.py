import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 500
    n = len(x)
    learning_rate = 0.000000001

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        #cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = (2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,1, i))
    return m_curr, b_curr


def plot_regression_line(x, y, b): 
	# plotting the actual points as scatter plot 
	plt.scatter(x, y, color = "m", marker = "o", s = 30) 
	print("in plot: ", int(b[0]), ", ", b[1])
	# predicted response vector 
	y_pred = b[0]*x + b[1]
  
	# plotting the regression line 
	plt.plot(x, y_pred, color = "g") 
  
	# putting labels 
	plt.xlabel('x') 
	plt.ylabel('y') 
  
	# function to show plot 
	plt.show() 

#this function read datas from file
def read_datas(file_name):
	f = open(file_name, "r")
	lines = f.readlines()

	head_size = np.empty(shape=[0,1])
	brain_size = np.empty(shape=[0,1])

	for counter in range(1, len(lines), 1):
		temp_head_size, temp_brain_size = lines[counter].split("\t\t\t")
		
		#in the file there is blank lines. For avoding from an error
		if len(temp_brain_size) != 0 or len(temp_head_size) != 0:
			temp_head_size = int(temp_head_size)
			temp_brain_size = int(temp_brain_size.split("\n")[0])

			head_size = np.append(head_size, temp_head_size)
			brain_size = np.append(brain_size, temp_brain_size)
	
	datas = pd.DataFrame()
	datas["head_size"] = head_size
	datas["brain_size"] = brain_size
	return datas

if __name__ == "__main__": 
	file_name = "regression_data.txt"
	
	datas = read_datas(file_name)

	m=0
	b=0

	m, b = gradient_descent(datas["head_size"], datas["brain_size"])

	#print("\nEstimated coefficients5:\n\t b_0 = {}  \n\t b_1 = {}".format(b4[0], b4[1]))

	plot_regression_line(datas["head_size"], datas["brain_size"], (m,b))