import pandas as pd

# Load the data
df = pd.read_excel(r"G:\Schneider Electric\NLP\SampleData.xlsx",sheet_name="Sheet3")
scripts = df['AnimationValue']
labels = df['Connections']
print(df.columns)