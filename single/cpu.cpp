#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include "brick.h"
#include "stencils/stencils.h"
#include <iostream>
#include <fstream>

bElem *coeff;

void write_coeff_into_file(){
    std::cout << "\n Dumping coefficients into Coefficients.txt";
    // Open a file for writing
    std::ofstream outfile("coefficients.txt");

    // Check if file opened successfully
    if (!outfile.is_open()) {
        std::cerr << "Unable to open file for writing." << std::endl;
        exit(1);
    }

    // Write coefficients to the file
    for (int i = 0; i < 129; ++i)
        outfile << coeff[i] << std::endl;

    // Close the file
    outfile.close();
}
int main() {
  // What this code does is fills the coefficient values 
  // this is going to be anyways constant so.... just fill and use later.
  coeff = (bElem *) malloc(129 * sizeof(bElem));
  std::random_device r;
  std::mt19937_64 mt(r());
  std::uniform_real_distribution<bElem> u(0, 1);

  for (int i = 0; i < 129; ++i)
    coeff[i] = u(mt);
  
  write_coeff_into_file();

  copy();
  // d3pt27();
  // // implemented for cuda, hip and DPC.
  // d3pt7();
  // d3cond();

  // // not implemented yet
  // //error
  // d3pt13();
  // //error_t
  // d3pt19();
  // //error
  // d3pt25();
  // //error
  // d3pt125();
  // //error
  // d3iso();

void d3cns();
  std::cout << "result match" << std::endl;
  return 0;
}
