import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
  
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
    
    print(datas)

    # estimating coefficients 
    b = estimate_coef(datas["head_size"], datas["brain_size"]) 
    print("Estimated coefficients:\nb_0 = {} \\ nb_1 = {}".format(b[0], b[1])) 
  
    # plotting regression line 
    plot_regression_line(datas["head_size"], datas["brain_size"], b) 
