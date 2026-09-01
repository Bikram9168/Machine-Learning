import numpy as np
import pandas as pd

data = {
    "Name":["Bikram" , "Aryan" , "Haladhar" , "Anubhav" , "Mohak"],
    "Roll_No":[1, 2, 3, 4, 5],
    "Marks":[80, 85, 65, 90 ,84],
    "Attendance":[88,92,68,72,81]
}

df = pd.DataFrame(data)
x=df[df["Marks"]>80]
print(x)