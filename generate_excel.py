import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("SDD_outputs.csv")

# Create a new Excel writer object
excel_writer = pd.ExcelWriter("SDD_outputs.xlsx", engine="xlsxwriter")

# Write the DataFrame to the Excel file
df.to_excel(excel_writer, index=False)

# Save the Excel file
excel_writer.save()

print("Excel file created successfully.")

