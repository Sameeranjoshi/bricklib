import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("SDD_outputs.csv")

# Write the DataFrame to the Excel file directly
df.to_excel("SDD_outputs.xlsx", index=False)

print("Excel file created successfully.")
