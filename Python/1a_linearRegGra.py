import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

def estimate_coef3(x,y):
	m=0
	c=0
	
	L=0.0001
	epochs = 1000
	
	n = float(len(x))
	
	#for i in range(len(x)):
	for i in range(0,10):
		
		Y_pred2 = [(e * m + c) for e in x]
		print("uzunluk: ",len(Y_pred2))
		print("y_pred2: ", Y_pred2)
		Y_pred = np.array(Y_pred2)
		print("y_pred: ", Y_pred)
		
		print("deneme: ", (1/n) * np.sum(x * (y-Y_pred)))
		
		#D_m = "{:.2f}".format((-2/n) * np.sum(x * (y - Y_pred)))  # Derivative wrt m
		#D_c = "{:.2f}".format((-2/n) * np.sum(y - Y_pred))  # Derivative wrt c
		D_m = int((-2/n) * np.sum(x * (y - Y_pred)))  # Derivative wrt m
		D_c = int((-2/n) * np.sum(y - Y_pred))  # Derivative wrt c
		print ("d_m: ", D_m, " D_c: ", D_c)
		
		m = m - (L * D_m)  # Update m
		c = c - (L * D_c)  # Update c
		#m = "{:.4f}".format(m - L * D_m)  # Update m
		#c = "{:.4f}".format(c - L * D_c)  # Update c		
		print("m: ", m, " c: ", c)
		
	return( m, c)

def estimate_coef2(x, y): 

	# Building the model
	m = 0
	c = 0

	m_deriv = 0
	c_deriv = 0

	L = 0.0001  # The learning Rate
	epochs = 100  # The number of iterations to perform gradient descent

	n = np.size(x) # Number of elements in X
	
	for a in range(epochs):
		
		m_deriv = 0
		c_deriv = 0
		# Performing Gradient Descent 
		for i in range(n): 
			#Y_pred = map( lambda e:e * m, x)   # The current predicted value of Y
			#print(Y_pred)
			#Y_pred = map( lambda e:e + c, x)
			#print(Y_pred)
			
			""" Y_pred = [e * m + c for e in x]
			print(Y_pred)

			D_m = (-2/n) * np.sum(x * (y - Y_pred))  # Derivative wrt m
			D_c = (-2/n) * np.sum(y - Y_pred)  # Derivative wrt c
			
			m = m - L * D_m  # Update m
			c = c - L * D_c  # Update c """

			m_deriv = m_deriv + (-2 * x[i] * (y[i] - (m * x[i] + c)))
			c_deriv = c_deriv + (-2 * (y[i] - (m * x[i] + c)))
			#print ("i: ", i, "m_deriv", m_deriv)
			#print ("i: ", i, "c_deriv", c_deriv)
			
		m = m - ((m_deriv / float(n)) * L)
		c = c - ((c_deriv / float(n)) * L)

	print (m, c)

	return(m, c) 

def linear_regression(X, y, m_current=0, b_current=0, epochs=1000, learning_rate=0.0001):
	N = float(len(y))
	for i in range(epochs):
		y_current = (m_current * X) + b_current
		cost = sum([data**2 for data in (y-y_current)]) / N
		
		m_gradient = -(2/N) * sum(X * (y - y_current))
		b_gradient = -(2/N) * sum(y - y_current)

		m_current = m_current - (learning_rate * m_gradient)
		b_current = b_current - (learning_rate * b_gradient)
	return m_current, b_current, cost

def estimate_coef(x, y): 
	# number of observations/points 
	n = np.size(x) 
  
	# mean of x and y vector 
	m_x, m_y = np.mean(x), np.mean(y) 
  
	# calculating cross-deviation and deviation about x 
	SS_xy = np.sum(y*x) - n*m_y*m_x 
	SS_xx = np.sum(x*x) - n*m_x*m_x 
  
	# calculating regression coefficients 
	b_1 = SS_xy / SS_xx 
	b_0 = m_y - b_1*m_x  

	print (b_0, b_1)

	return(b_0, b_1) 


def plot_regression_line(x, y, b): 
	# plotting the actual points as scatter plot 
	plt.scatter(x, y, color = "m", marker = "o", s = 30) 
  
	# predicted response vector 
	y_pred = b[0] + b[1]*x 
  
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
	
	# estimating coefficients 
	#b = estimate_coef(datas["head_size"], datas["brain_size"]) 
	#b2 = estimate_coef2(datas["head_size"], datas["brain_size"]) 
	#b3 = linear_regression(datas["head_size"], datas["brain_size"], 0, 0, 1000, 0.0001)
	b3 = estimate_coef3(datas["head_size"], datas["brain_size"])

	#print("\nEstimated coefficients:\n\t b_0 = {}  \n\t b_1 = {}".format(b[0], b[1])) 
	#print("\nEstimated coefficients2:\n\t b_0 = {}  \n\t b_1 = {}".format(b2[0], b2[1])) 
	print("\nEstimated coefficients3:\n\t b_0 = {}  \n\t b_1 = {}".format(b3[0], b3[1])) 
	
	# plotting regression line 
	#plot_regression_line(datas["head_size"], datas["brain_size"], b) 
	#plot_regression_line(datas["head_size"], datas["brain_size"], b2) 
	plot_regression_line(datas["head_size"], datas["brain_size"], b3) 
