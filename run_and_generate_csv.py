import subprocess

# Run the executable 100 times and capture the output into output.txt
with open("output.txt", "w") as output_file:
    for i in range(50):
        subprocess.run(["/uufs/chpc.utah.edu/common/home/u1418973/other/verify_gravel/bricklib/examples/external/build/example"], stdout=output_file, text=True)

# Run the script to generate the CSV file
subprocess.run(["python3", "csv_generator.py"])
