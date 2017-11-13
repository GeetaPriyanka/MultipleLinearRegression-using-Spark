
# Name- Geeta Priyanka Janapareddy
# Email id - gjanapar@uncc.edu



import sys
import numpy as np

from pyspark import SparkContext

#To print output with precision
np.set_printoptions(precision=13)

def keyA(line):
    "This function is to find X * (X_transpose)"
    line[0]=1.0
    temp_x = np.array(line).astype('float')
    X = np.asmatrix(temp_x).T
    #print X
    X_Xtranspose = np.dot(X,X.T)
    return X_Xtranspose

def keyB(line):
    "This function is to find X * Y"
    Y = float(line[0])
    line[0] = 1.0
    temp_x = np.array(line).astype('float')
    X = np.asmatrix(temp_x).T
    #print "Y: ",Y
    #print "X: ",X
    X_Y = np.multiply(X,Y)
    return X_Y

if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])
  yxlines = yxinputFile.map(lambda line: line.split(','))
  
  #Calculating (X * (X_Transpose)) and add them using reduceBYKey function  
  A = np.asmatrix(yxlines.map(lambda line: ("KeyA",keyA(line))).reduceByKey(lambda x1,x2: np.add(x1,x2)).map(lambda line: line[1]).collect()[0])
  #print A
  
  #Calculating (X * Y) and add them using reduceBYKey function
  B = np.asmatrix(yxlines.map(lambda line: ("KeyB",keyB(line))).reduceByKey(lambda x1,x2: np.add(x1,x2)).map(lambda line: line[1]).collect()[0])
  #print B
  
  #Shape give the dimension of matrix
  #print A.shape
  #print B.shape
  
  #Multiplying A_inverse with B to get the coefficients
  beta = np.dot(np.linalg.inv(A),B)
  #Converting the matrix to list 
  beta_list = np.array(beta).tolist()
  print beta_list
  

  # printing the linear regression coefficients 
  print "beta: "
  for coeff in beta_list:
      print coeff[0]

  sc.stop()
