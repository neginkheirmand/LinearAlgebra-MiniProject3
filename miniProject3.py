import pandas as pd
import numpy as np
df = None

def readData(nameFile):
    global df
    try:
        df = pd.read_csv(nameFile)
        print(df.info())
        print(df)
    except FileNotFoundError:
        print("Can't find specified file") 


def linearRegression(data):
    #since its linear regression the A matrix will be something like:
    #       | 1   x1|
    # A=    | 1   x2|
    #       | 1   x3|
    # and the b matrix would be:
    #       |   y1  |
    #  b=   |   y2  |
    #       |   y3  |
    A = np.ones((len(data), 2))
    b = np.ones((len(data)))
    # data contains pairs of x and y : data =[ ... [xn, yn], [x(n+1), y(n+1)]   ... ]
    for i in range(0, len(data)):
        x = data[i][0]
        y = data[i][1]
        A[i][1] = x
        b[i] = y
    print("Linear Regression")
    print(A)
    print(b)
    newX = Regression(A, b)



def squareRegression(data):
    #since its linear regression the A matrix will be something like:
    #       | 1   x1    x1^2|
    # A=    | 1   x2    x2^2|
    #       | 1   x3    x3^2|
    # and the b matrix would be:
    #       |   y1  |
    #  b=   |   y2  |
    #       |   y3  |
    A = np.ones((len(data), 3))
    b = np.ones((len(data)))
    # data contains pairs of x and y : data =[ ... [xn, yn], [x(n+1), y(n+1)]   ... ]
    for i in range(0, len(data)):
        x = data[i][0]
        y = data[i][1]
        A[i][1] = x
        A[i][2] = x*x
        b[i] = y
    print("Linear Regression")
    print(A)
    print(b)
    newX = Regression(A, b)



def Regression(A, b):
    #this function solves the least square
    # for the equation Ax=b
    # by using the AᵗA xˆ= Aᵗ b
    #and finding the xˆ
    transposeA = A.transpose()
    # we will call the product of AᵗA the newA
    newA = np.dot(transposeA, A)
    #we will call the product of Aᵗb the newb
    newb = np.dot(transposeA, b)
    #now find the newx for which the equation newA * newx = newb
    #since we are sure the newA is not singular and invertible 
    newx = np.linalg.solve(newA, newb)
    

if __name__ == "__main__":
    # readData("GOOGL.csv")
    # A = np.array([ [1,1,0,0], [1,1,0,0], [1,0,1,0], [1,0,1,0], [1,0,0,1], [1,0,0,1]])
    # b = np.array([ -3, -1, 0, 2, 5, 1])
    # Regression(A, b)
    linearRegression([[1,2], [2,3], [3, 4]])