#include "grid_functions.h"
#include <fstream>
#include <iostream>


void fill_data_in_grid_default_way(unsigned *&grid_ptr, long sizeofgrid){
  for (unsigned pos = 0; pos < sizeofgrid; ++pos)
    grid_ptr[pos] = pos;
}

void fill_data_in_grid_from_inputfile(unsigned *&grid_ptr, long sizeofgrid, const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Unable to open file " << filename  << "for reading data from." << std::endl;
        return;
    }
    // Read data from the file into the grid
    // check the size of the grid
    for (unsigned pos = 0; pos < sizeofgrid; ++pos) {
      if (!(infile >> grid_ptr[pos])) {
          std::cerr << "Error reading data from file." << std::endl;
         exit(1);
      }
    }
    std::cout << "\nRead grid_contents.txt";
    infile.close();
}

void dump_data_from_grid_into_outputfile(unsigned *&grid_ptr, long sizeofgrid, const std::string &filename){
   std::ofstream outfile(filename);
  // Check if the file opened successfully
  if (!outfile.is_open()) {
      std::cerr << "Unable to open file for writing the data into." << std::endl;
      return;
  }
  // actual data.
  for (unsigned pos = 0; pos < sizeofgrid; ++pos){
    outfile << grid_ptr[pos];
    outfile << std::endl;
  }
  // close the file handle.
  outfile.close();
}

void print_data_in_grid_default_way(unsigned *&grid_ptr, long sizeofgrid, const std::string &msg){
  std::cout << msg;
  std::cout << std::endl;
  for (unsigned pos = 0; pos < 10; ++pos)
    std::cout << grid_ptr[pos] << "\n";
}