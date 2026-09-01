import numpy as np
import pandas as pd

data = {
    "Name":["Bikram" , "Aryan" , "Haladhar" , "Anubhav" , "Mohak"],
    "Roll_No":[1, 2, 3, 4, 5],
    "Marks":[80, 85, 65, 90 ,84],
    "Attendance":[88,92,68,72,81]
}

df = pd.DataFrame(data)


def assign_grade(Marks):
    if Marks >= 90:
        return "A"
    elif Marks >= 80 and Marks <= 89:
        return "B"
    elif Marks >= 70:
        return "C"
    elif Marks >= 60:
        return "D"
    else:
        return "Fail"

df["Grade"]=df["Marks"].apply(assign_grade)
print(df)