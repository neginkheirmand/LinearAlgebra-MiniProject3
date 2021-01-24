import pandas as pd
df = None

def readData(nameFile):
    global df
    try:
        df = pd.read_csv(nameFile)
        print(df.info())
        print(df)
    except FileNotFoundError:
        print("Can't find specified file") 

readData("GOOGL.cs")