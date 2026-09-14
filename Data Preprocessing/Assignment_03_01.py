import numpy as np
import pandas as pd

data = {
    "Age": np.random.randint(22, 60, size=10).astype(float), 
    "Salary": np.random.randint(40000, 120000, size=10).astype(float), 
    "Department": np.random.choice(["HR", "IT", "Marketing", "Sales"], size=10), 
    "Years_of_Experience": np.random.randint(1, 35, size=10).astype(float), 
}
df = pd.DataFrame(data) 

df.loc[2, 'Age'] = np.nan                  
df.loc[5, 'Salary'] = np.nan               
df.loc[4, 'Department'] = np.nan           
df.loc[7, 'Years_of_Experience'] = np.nan  
