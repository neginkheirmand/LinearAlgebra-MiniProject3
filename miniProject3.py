import pandas as pd
import numpy as np
import math  #just to deal with round off errors
import matplotlib.pyplot as plt

openData = None
openDataPrime = None
allData = None
xLinear = None
xPoly = None
sumAbsoluteErrorLinear = 0
sumAbsoluteErrorPoly = 0
def readData(nameFile):
    global openData
    global openDataPrime
    global allData
    try:
        df = pd.read_csv(nameFile)
        openData = df.head(len(df)-10)['Open'].to_list()
        allData = df['Open'].to_list()
        openDataPrime = df.tail(10)['Open'].to_list()
        return True
    except FileNotFoundError:
        print("Can't find specified file") 
        return False

def getb(data):
    # the b matrix would be:
    #       |   y1  |
    #  b=   |   y2  |
    #       |   y3  |
    b = np.ones( ((len(data), 1)) )
    # data contains y values : data =[ ... [yn], [y(n+1)]   ... ]
    for i in range(0, len(data)):
        y = data[i]
        b[i] = y
    return b

def getA_linearRegression(data):
    #since its linear regression the A matrix will be something like:
    #       | 1   x1|
    # A=    | 1   x2|
    #       | 1   x3|
    A = np.ones((len(data), 2))
    # data contains y values (indexes are x's): data =[ ... [yn], [y(n+1)]   ... ]
    for i in range(0, len(data)):
        x = i
        A[i][1] = x
    return A

def getA_SquareRegression(data):
    #since its linear regression the A matrix will be something like:
    #       | 1   x1    x1^2|
    # A=    | 1   x2    x2^2|
    #       | 1   x3    x3^2|
    A = np.ones((len(data), 3))
    # data contains y values (indexes are x's): data =[ ... [yn], [y(n+1)]   ... ]
    for i in range(0, len(data)):
        x = i
        A[i][1] = x
        A[i][2] = x*x
    return A

def linearRegression(data):
    #get A and b
    A = getA_linearRegression(data)
    b = getb(data)
    print('Linear Regression:')
    print("Trying to solve Ax = b, where A:")
    print(A)
    print("and b:")
    print(b)
    #now solve the equation
    newX = Regression(A, b)
    #and print the solution
    print("as result x would be:")
    print(newX)
    print('\033[0m')
    global xLinear 
    xLinear = newX
    return newX


def squareRegression(data):
    #get A and b
    A = getA_SquareRegression(data)
    b = getb(data)
    print("polynomial Regression(n=2)")
    print(A)
    print(b)
    #now solve the equation
    newX = Regression(A, b)
    #and print the solution
    print(newX)
    global xPoly 
    xPoly = newX
    return newX

def leastSquares_error_Linear(date, data , newX):
    calculatedValue = newX[0]+newX[1]*date
    actualValue = data
    errorMatrix = calculatedValue - actualValue
    #dealing with round off errors
    ceil = math.ceil(errorMatrix)
    floor = math.floor(errorMatrix)
    if ceil - errorMatrix < 0.0000001:
        errorMatrix = ceil
    elif errorMatrix - floor < 0.0000001:
        errorMatrix = floor
    #printing the error
    print('\033[33mcalculated value: \033[0m', calculatedValue)
    print('\033[33mactual value: \033[0m', actualValue)
    print('\033[33merror: \033[0m', errorMatrix)
    return errorMatrix 


def leastSquares_error_Square(date, data , newX):
    calculatedValue = newX[0] + newX[1]*date + (newX[2]*date*date)
    actualValue = data
    errorMatrix = calculatedValue - actualValue
    #dealing with round off errors
    ceil = math.ceil(errorMatrix)
    floor = math.floor(errorMatrix)
    if ceil - errorMatrix < 0.0000001:
        errorMatrix = ceil
    elif errorMatrix - floor < 0.0000001:
        errorMatrix = floor
    #printing the error 
    print('\033[33mcalculated value: \033[0m', calculatedValue)
    print('\033[33mactual value: \033[0m', actualValue)
    print('\033[33merror: \033[0m', errorMatrix)
    return errorMatrix 

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
    #dealing with roundoff errors
    for i in range(0, len(newx)):
        ceil = math.ceil(newx[i])
        floor = math.floor(newx[i])
        if ceil - newx[i] < 0.0000001:
            newx[i] = ceil
        elif newx[i] - floor < 0.0000001:
            newx = floor
    return newx

def process():
    global openData
    global openDataPrime
    print('\033[36m Process:\033[0m')
    linearX = linearRegression(openData)
    global sumAbsoluteErrorLinear
    #Linear regression
    print('\033[31m and the least error squares with linear regression:')
    for i in range(0, len(openDataPrime)):
        #the date would be len(openData)-10 + i
        sumAbsoluteErrorLinear += math.abs(leastSquares_error_Linear(len(openData) - 10 + i, openDataPrime[i], linearX))
        print()
    print('\033[36m Process:\033[0m')
    polynomialX = squareRegression(openData)
    global sumAbsoluteErrorPoly
    #Polinomial regression
    print('\033[31m and the least error squares with square regression:')
    for i in range(0, len(openDataPrime)):
        #the date would be len(openData)-10 + i
        sumAbsoluteErrorPoly +=math.abs(leastSquares_error_Square(len(openData) - 10 + i, openDataPrime[i], polynomialX))
        print()

def draw():
    global xLinear
    global  xPoly
    global allData
    plt.plot([1, 2, 3, 4])
    plt.ylabel('Google shares')
    plt.xlabel('#day')
    x = np.arange(0, len(allData)) 
    plt.title("Analysis of Google shares using Regression")
    plt.scatter( range(0, len(allData)), allData, color='red', label='provided data')
    yLinear = xLinear[0] + xLinear[1] * x 
    plt.plot(yLinear, color = 'blue', label='Linear Regression Analysis')
    yPoly = xPoly[0]+xPoly[1]*x+xPoly[2]*x*x 
    plt.plot(yPoly, color = 'cyan', label='Polynomial Regression Analysis (n=2) ')
    plt.legend()
    plt.show()
    
def comparator():
    print("Taking in account only tyhe last 10 values of the doc we can say:")
    global sumAbsoluteErrorPoly
    global sumAbsoluteErrorLinear
    if sumAbsoluteErrorLinear>sumAbsoluteErrorPoly:
        print("Polinomial regression(n=2) is more accurate")
    else:
        print("Linear regression(n=2) is more accurate")

if __name__ == "__main__":
    if readData("GOOGL.csv"):
        process()
        comparator()
        draw()