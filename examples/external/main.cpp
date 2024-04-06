#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include "brick.h"
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include "brickverify.h"
#include <string.h>
#include <utility>
#include <cstdlib>
#include <fstream>

// Setting for X86 with at least AVX2 support
#include <immintrin.h>
#define VSVEC "AVX2"
#define VFOLD 2,2

// Domain size
#define N 4
#define TILE 1

#define GZ TILE
#define PADDING 8

// Stride for arrays is GHOSTZONE + PADDING on each side
#define STRIDE (N + 2 * (GZ + PADDING))
#define STRIDEG (N + 2 * GZ)

#define NB (N / TILE)
#define GB (GZ / TILE)

// Stride for bricks is GHOSTZONE on each side
#define STRIDEB ((N + 2 * GZ) / TILE)

#define BDIM TILE,TILE,TILE
#define TOT_TIME 5

#define _PARFOR _Pragma("omp parallel for collapse(2)")
#define _TILEFOR _Pragma("omp parallel for collapse(2)") \
for (long tk = PADDING; tk < PADDING + STRIDEG; tk += TILE) \
for (long tj = PADDING; tj < PADDING + STRIDEG; tj += TILE) \
for (long ti = PADDING; ti < PADDING + STRIDEG; ti += TILE) \
for (long k = tk; k < tk + TILE; ++k) \
for (long j = tj; j < tj + TILE; ++j) \
_Pragma("omp simd") \
for (long i = ti; i < ti + TILE; ++i)

// -- Utilities
// struct Result {
//     Brick<Dim<BDIM>, Dim<VFOLD>> bOut;
//     unsigned *grid_ptr;
//     Brick<Dim<BDIM>, Dim<VFOLD>> bIn;

// };
struct Result {
    Brick<Dim<BDIM>, Dim<VFOLD>> *bOut;
    unsigned *grid_ptr;
    Brick<Dim<BDIM>, Dim<VFOLD>> *bIn;
};

struct global_args {
  int write_coeff_into_file;
  int read_coeff_from_file;
  int write_grid_with_ghostzone_into_file;
  int read_grid_with_ghostzone_from_file;
};

// global declaration allocates space automatically.
global_args arg_handler1;
global_args arg_handler2;

void debug_global_args(global_args *handler){
  std::cout << handler->write_coeff_into_file << std::endl;
  std::cout << handler->write_grid_with_ghostzone_into_file << std::endl;
  std::cout << handler->read_coeff_from_file << std::endl;
  std::cout << handler->read_grid_with_ghostzone_from_file << std::endl;
}

void write_coeff_into_file(bElem *coeff, std::string &filename) {
    // Open a file for writing in binary mode
    std::ofstream outfile(filename, std::ios::binary);

    // Check if file opened successfully
    if (!outfile.is_open()) {
        std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
        exit(1);
    }

    // Write coefficients to the file
    outfile.write(reinterpret_cast<const char*>(coeff), sizeof(bElem) * 129);

    // Close the file
    outfile.close();
    std::cout << "\nWritten " << filename;
}

void read_coeff_from_file(bElem *coeff, std::string &filename) {
    // Open the file for reading in binary mode
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Unable to open file " << filename << " for reading." << std::endl;
        exit(1);
    }
    // Read coefficients from the file
    infile.read(reinterpret_cast<char*>(coeff), sizeof(bElem) * 129);
    infile.close();
    std::cout << "\nRead " << filename;
}

void handle_coefficient_data(bElem *coeff, global_args *handler, std::string &filename){
  
  std::random_device r;
  std::mt19937_64 mt(r());
  std::uniform_real_distribution<bElem> u(0, 1);

  // reading
  if (handler->read_coeff_from_file){
    read_coeff_from_file(coeff, filename);
  } else{
    // Randomly initialize coefficient, Towan's way of initialization.
    for (int i = 0; i < 129; ++i)
      coeff[i] = u(mt);
  }
  // writing
  if (handler->write_coeff_into_file){
    write_coeff_into_file(coeff, filename);
  }

}

// // Function to write the grid contents to a file
// void write_grid_with_ghostzone_into_file(unsigned (*grid)[STRIDEB][STRIDEB]) {
//     // Open a file for writing
//     std::ofstream outfile("grid_contents.txt");
//     // Check if the file opened successfully
//     if (!outfile.is_open()) {
//         std::cerr << "Unable to open file for writing." << std::endl;
//         return;
//     }
//     // Write the contents of the grid to the file
//     // Write the contents of the grid to the file
//     for (int i = 0; i < STRIDEB; ++i) {
//         for (int j = 0; j < STRIDEB; ++j) {
//             for (int k = 0; k < STRIDEB; ++k) {
//                 auto b = grid[i][j][k];
//                 for (long k1 = 0; k1 < TILE; ++k1) {
//                     for (long j1 = 0; j1 < TILE; ++j1) {
//                         for (long i1 = 0; i1 < TILE; ++i1) {            
//                             outfile << grid[b][k1][j1][i1] << " ";
//                         }
//                     }
//                     outfile << std::endl; // Move to the next line after printing each row
//                 }
//             }
//         }
//         outfile << std::endl; // Separate each STRIDEBxSTRIDEB slice with an empty line
//     }
//     outfile << std::endl; // Separate each STRIDEBxSTRIDEB slice with an empty line
//     // Close the file
//     outfile.close();
//     std::cout << "\n Written grid_contents.txt \n";
// }

int handle_argument_parsing(int argc, char** argv, global_args *handler) {
    auto generic_error_msg = [&]() {
        std::cerr << "Usage: " << argv[0] << " --dump-coeff=<true|false> --dump-grid=<true|false> --read-coeff=<true|false> --read-grid=<true|false>" << std::endl;
        std::cerr << "Usage: " << argv[0] << " --help" << std::endl;
    };

    // Check if argument list has help.

    for (int i=0; i< argc; i++){
      std::string temparg = argv[i];
      if (temparg.find("--help") == 0){
        generic_error_msg();
        exit(0);
      }
    }

    // Check the number of command-line arguments
    if (argc != 5 || argc == 1) {
        generic_error_msg();
        exit(1);
    }

    // Parse the command-line argument
    // --dump-coeff=<true|false>
    std::string arg1 = argv[1];
    if (arg1.find("--dump-coeff=") == 0) {
        std::string value = arg1.substr(13); // Skip "--dump-coeff="
        if (value == "true") {
            handler->write_coeff_into_file = 1;
        } else if (value == "false") {
            handler->write_coeff_into_file = 0;
        } else {
            generic_error_msg();
            return 1;
        }
    } else {
        generic_error_msg();
        return 1;
    }

    std::string arg2 = argv[2];
    if (arg2.find("--dump-grid=") == 0) {
        std::string value = arg2.substr(12); // Skip "--dump-grid="
        if (value == "true") {
            handler->write_grid_with_ghostzone_into_file = 1;
        } else if (value == "false") {
            handler->write_grid_with_ghostzone_into_file = 0;
        } else {
            generic_error_msg();
            return 1;
        }
    } else {
        generic_error_msg();
        return 1;
    }

    std::string arg3 = argv[3];
    if (arg3.find("--read-coeff=") == 0) {
        std::string value = arg3.substr(13); // Skip "--read-coeff="
        if (value == "true") {
            handler->read_coeff_from_file = 1;
        } else if (value == "false") {
            handler->read_coeff_from_file = 0;
        } else {
            generic_error_msg();
            return 1;
        }
    } else {
        generic_error_msg();
        return 1;
    }

    std::string arg4 = argv[4];
    if (arg4.find("--read-grid=") == 0) {
        std::string value = arg4.substr(12); // Skip "--read-grid="
        if (value == "true") {
            handler->read_grid_with_ghostzone_from_file = 1;
        } else if (value == "false") {
            handler->read_grid_with_ghostzone_from_file = 0;
        } else {
            generic_error_msg();
            return 1;
        }
    } else {
        generic_error_msg();
        return 1;
    }
    return 0;
}

// -- end utilities

template<typename T>
double time_func(T func) {
  int it = 1;
  func(); // Warm up
  double st = omp_get_wtime();
  double ed = st;
  while (ed < st + TOT_TIME) {
    for (int i = 0; i < it; ++i)
      func();
    it <<= 1;
    ed = omp_get_wtime();
  }
  return (ed - st) / (it - 1);
}

using std::max;

bElem *coeff1;
bElem *coeff2;

void d3pt7() {
     std::string filename_brick_original = "brick_original.bin";
    std::string filename_brick_CDC = "brick_CDC.bin"; // should be CDC brick at some point.
    std::string filename_coeff_original = "coeff_original.bin";
    std::string filename_coeff_CDC = "coeff_CDC.bin";
  // arg_handler is a global struct containing various flags.
  global_args arg_handler1 = {0};
  global_args arg_handler2 = {0};
  // allocate space for coefficients
  coeff1 = (bElem *) malloc(129 * sizeof(bElem));
  coeff2 = (bElem *) malloc(129 * sizeof(bElem));
  // handle_argument_parsing(argc, argv, &arg_handler);
  arg_handler1.write_coeff_into_file = 1;
  arg_handler1.read_coeff_from_file = 0;
  arg_handler1.write_grid_with_ghostzone_into_file = 1;
  arg_handler1.read_grid_with_ghostzone_from_file = 0;  
  // handle coefficients
  
  handle_coefficient_data(coeff1, &arg_handler1, filename_coeff_original);
  // Run once and collect data.  

// #######################################


// ---------------------------------------

  // Bricks
  unsigned *grid_ptr1;
  auto bInfo1 = init_grid<3>(grid_ptr1, {STRIDEB, STRIDEB, STRIDEB}, arg_handler1.read_grid_with_ghostzone_from_file, arg_handler1.write_grid_with_ghostzone_into_file);
  auto grid1 = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr1;
  // Array
  bElem *in_ptr1 = randomArray({STRIDE, STRIDE, STRIDE});
  // Bricks
  auto bSize1 = cal_size<BDIM>::value;
  auto bStorage1 = BrickStorage::allocate(bInfo1.nbricks, bSize1 * 2);
  Brick<Dim<BDIM>, Dim<VFOLD>> bIn1(&bInfo1, bStorage1, 0);
  Brick<Dim<BDIM>, Dim<VFOLD>> bOut1(&bInfo1, bStorage1, bSize1);

  // Remember array is <STRIDE> which is 288, brick is of STRIDEG=272, but notice here they only copy 272 size elements from array.
  // which means a portion of array is copied.
  copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr1, grid_ptr1, bIn1);
  // Write brick data into file.
  if (arg_handler1.write_grid_with_ghostzone_into_file){
 
    write_brick_into_file_verify<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr1, grid_ptr1, bIn1, filename_brick_original);
  } else{
    std::cout << "\n Skipped writing brick1 data into file";
  }

  auto brick_func1 = [&grid1, &bIn1, &bOut1]() -> void {
    _PARFOR
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj)
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
          unsigned b = grid1[tk][tj][ti];
          for (long k = 0; k < TILE; ++k)
            for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                bOut1[b][k][j][i] = coeff1[5] * bIn1[b][k + 1][j][i] + coeff1[6] * bIn1[b][k - 1][j][i] +
                                   coeff1[3] * bIn1[b][k][j + 1][i] + coeff1[4] * bIn1[b][k][j - 1][i] +
                                   coeff1[1] * bIn1[b][k][j][i + 1] + coeff1[2] * bIn1[b][k][j][i - 1] +
                                   coeff1[0] * bIn1[b][k][j][i];
              }
        }
  };

  std::cout << "\n Running - d3pt7 Original" << std::endl;
  brick_func1();
// #######################################
//CRUSHER POINT
  // input files.
  std::cout << "\n \t Running CRUSHER ";
  std::string python_output = "python_output.txt";
  std::string command = "python3 ../python-wrapper/runner.py " + filename_coeff_original + " " + filename_brick_original + " > " + python_output + " 2>&1";
  // only copy outputs no crushing
  // std::string command = "cp " + filename_coeff_original + " " + filename_coeff_CDC + " && cp " + filename_brick_original + " " + filename_brick_CDC;
  std::cout << "\n " << command << "\n";
  // system("python ../python-wrapper/runner.py");
  system(command.c_str());
  
// #######################################
  // handle_argument_parsing(argc, argv, &arg_handler);
  arg_handler2.write_coeff_into_file = 0;
  arg_handler2.read_coeff_from_file = 1;
  arg_handler2.write_grid_with_ghostzone_into_file = 0;
  arg_handler2.read_grid_with_ghostzone_from_file = 1;  
  // handle coefficients
  
  handle_coefficient_data(coeff2, &arg_handler2, filename_coeff_CDC);
  // Run once and collect data.
// ---------------------------------------

  // Bricks
  unsigned *grid_ptr2;
  auto bInfo2 = init_grid<3>(grid_ptr2, {STRIDEB, STRIDEB, STRIDEB}, arg_handler2.read_grid_with_ghostzone_from_file, arg_handler2.write_grid_with_ghostzone_into_file);
  auto grid2 = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr2;
  // Array
  bElem *in_ptr2 = randomArray({STRIDE, STRIDE, STRIDE});
  // Bricks
  auto bSize2 = cal_size<BDIM>::value;
  auto bStorage2 = BrickStorage::allocate(bInfo2.nbricks, bSize2 * 2);
  Brick<Dim<BDIM>, Dim<VFOLD>> bIn2(&bInfo2, bStorage2, 0);
  Brick<Dim<BDIM>, Dim<VFOLD>> bOut2(&bInfo2, bStorage2, bSize2);

  // We don't need to copy into second brick, it should be read from file.
  // copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr2 , grid_ptr2, bIn2);
 // Write brick data into file.
  if (arg_handler2.read_grid_with_ghostzone_from_file){
    
    read_brick_from_file_verify<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr2, grid_ptr2, bIn2, filename_brick_CDC);
  } else{
    std::cout << "\n Skipped reading brick2 data from file";
  }
    // std::string filename = "brick_data_read_written_check.txt";
    // write_brick_into_file_verify<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr2, grid_ptr2, bIn2, filename);

  auto brick_func2 = [&grid2, &bIn2, &bOut2]() -> void {
    _PARFOR
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj)
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
          unsigned b = grid2[tk][tj][ti];
          for (long k = 0; k < TILE; ++k)
            for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                bOut2[b][k][j][i] = coeff2[5] * bIn2[b][k + 1][j][i] + coeff2[6] * bIn2[b][k - 1][j][i] +
                                   coeff2[3] * bIn2[b][k][j + 1][i] + coeff2[4] * bIn2[b][k][j - 1][i] +
                                   coeff2[1] * bIn2[b][k][j][i + 1] + coeff2[2] * bIn2[b][k][j][i - 1] +
                                   coeff2[0] * bIn2[b][k][j][i];
              }
        }
  };
  // auto brick_func_trans2 = [&grid2, &bIn2, &bOut2]() -> void {
  //     _PARFOR
  //     for (long tk = GB; tk < STRIDEB - GB; ++tk)
  //       for (long tj = GB; tj < STRIDEB - GB; ++tj)
  //         for (long ti = GB; ti < STRIDEB - GB; ++ti) {
  //           unsigned b = grid2[tk][tj][ti];
  //           brick("7pt.py", VSVEC, (BDIM), (VFOLD), b);
  //         }
  //   };

  std::cout << "\n Running - d3pt7 CDC version" << std::endl;
  brick_func2();
  


// #######################################
  if (!verifyBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, grid_ptr1, bIn1, grid_ptr2, bIn2))
    std::cout << "\n 1). Floating point verification mismatched (bIn1, bIn2)";
  else
    std::cout << "\n 1). Floating point verification matched (bIn1, bIn2)";  

  // B1 - must be original
  // B2 - must be CDC
  if (!verifyBrick_numerical<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, grid_ptr1, bIn1, grid_ptr2, bIn2))
    std::cout << "\n 2). Numerical verification mismatched (bIn1, bIn2)\n";
  else
    std::cout << "\n 2). Numerical vcerification match (bIn1, bIn2)\n";  

  std::cout << "\n\t Running VERIFICATION \n";
  if (!verifyBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, grid_ptr1, bOut1, grid_ptr2, bOut2))
    std::cout << "\n 1). Floating point verification mismatched (bOut1, bOut2)";
  else
    std::cout << "\n 1). Floating point verification matched (bout1, bOut2)";  

  // B1 - must be original
  // B2 - must be CDC
  if (!verifyBrick_numerical<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, grid_ptr1, bOut1, grid_ptr2, bOut2))
    std::cout << "\n 2). Numerical verification mismatched (bOut1, bOut2)\n";
  else
    std::cout << "\n 2). Numerical vcerification match (bout1, bOut2)\n";  
  
  // DEBUG
  print_both_Bricks_verify<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, grid_ptr1, bIn1, grid_ptr2, bIn2);
  std::cout << "\n\n";
  std::cout << "\n Above is input below is output";
  std::cout << "\n\n";
  print_both_Bricks_verify<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, grid_ptr1, bOut1, grid_ptr2, bOut2);

}

int main(int argc, char **argv) {
  d3pt7();

  return 0;
}


// TODO:
// 1. printBrick -> copytoBrick same function create one.
// 2. writeBrick -> copytoBrick same as this function.