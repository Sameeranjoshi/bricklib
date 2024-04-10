import subprocess
import sys

# Get the number of runs from the command line
num_runs = int(sys.argv[1])  # First command-line argument after script name

# Run the executable the specified number of times and capture the output into output.txt
with open("output.txt", "w") as output_file:
    for i in range(num_runs):
        subprocess.run(["/uufs/chpc.utah.edu/common/home/u1418973/other/verify_gravel/bricklib/examples/external/build/example"], stdout=output_file, text=True)

# Run the script to generate the CSV file
subprocess.run(["python3", "csv_excel_generator.py"])
