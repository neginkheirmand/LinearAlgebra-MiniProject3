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


def linearRegression(A, b):
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
    for i in range(0, len(newx)):
        print(newx[i])

if __name__ == "__main__":
    # readData("GOOGL.csv")
    A = np.array([ [1,2], [1,5], [1,7], [1,8] ])
    b = np.array([ 1, 2, 3, 3])
    linearRegression(A, b)