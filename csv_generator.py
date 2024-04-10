# Define the input and output filenames
input_filename = "output.txt"
output_filename = "SDD_outputs.csv"

# Initialize a list to hold the SDD values
sdd_values = []

# Open the input file
with open(input_filename, "r") as input_file:
    lines = input_file.readlines()

# Iterate through the lines of the input file
for line in lines:
    # Find lines containing SDD outputs
    if "Out of total" in line:
        # Extract SDD values
        parts = line.split("SDD")[1:]  # Split the line based on "SDD" and discard the first empty part
        sdd_values_line = [part.split("=")[-1].strip() for part in parts]
        sdd_values.append(sdd_values_line)

# create 2 lists input and output.
# Separate even and odd indexed elements into two lists
input_sdd_values = []
output_sdd_values = []

for index, value in enumerate(sdd_values):
    if index % 2 == 0:
        input_sdd_values.append(value)
    else:
        output_sdd_values.append(value)

# Write the SDD values to the output file
with open(output_filename, "w") as output_file:
    output_file.write("Bricks,SDD0,SDD1,SDD2,SDD3,SDD4,SDD5,SDD6,SDD7,SDD8,SDD9,SDD10,SDD11,SDD12,SDD13,SDD14,SDD15\n")
    for values in input_sdd_values:
        output_file.write("216," + "".join(values) + "\n")
    for values1 in output_sdd_values:
        output_file.write("216," + "".join(values1) + "\n")   

print(f"CSV {output_filename} created successfully.")

