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

// Setting for X86 with at least AVX2 support
#include <immintrin.h>
#define VSVEC "AVX2"
#define VFOLD 2,2

// Domain size
#define N 256
#define TILE 8

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
    Brick<Dim<BDIM>, Dim<VFOLD>> &bIn;
};

struct global_args {
  int write_coeff_into_file;
  int read_coeff_from_file;
  int write_grid_with_ghostzone_into_file;
  int read_grid_with_ghostzone_from_file;
};

// global declaration allocates space automatically.
global_args arg_handler;

void debug_global_args(global_args *handler){
  std::cout << handler->write_coeff_into_file << std::endl;
  std::cout << handler->write_grid_with_ghostzone_into_file << std::endl;
  std::cout << handler->read_coeff_from_file << std::endl;
  std::cout << handler->read_grid_with_ghostzone_from_file << std::endl;
}

void write_coeff_into_file(bElem *coeff){
    
    // Open a file for writing
    std::string filename="coefficients.txt";
    std::ofstream outfile(filename);

    // Check if file opened successfully
    if (!outfile.is_open()) {
        std::cerr << "Unable to open file" << filename << "for writing." << std::endl;
        exit(1);
    }

    // Write coefficients to the file
    for (int i = 0; i < 129; ++i)
        outfile << coeff[i] << std::endl;
    
    // Close the file
    outfile.close();
    std::cout << "\n Written " << filename;
}

void read_coeff_from_file(bElem *coeff) {
    std::vector<bElem> coefficients;

    // Open the file for reading
    std::string filename = "coefficients.txt";
    std::ifstream infile(filename);

    // Check if the file opened successfully
    if (!infile.is_open()) {
        std::cerr << "Unable to open file " << filename << " for reading." << std::endl;
        exit(1);
    }

    // Read coefficients from the file
    int i=0;
    while (infile >> coeff[i]) {
        i++;
    }

    // Close the file
    infile.close();

    std::cout << "\n Read " << filename;

}

void handle_coefficient_data(bElem *coeff, global_args *handler){
  
  std::random_device r;
  std::mt19937_64 mt(r());
  std::uniform_real_distribution<bElem> u(0, 1);

  // reading
  if (handler->read_coeff_from_file){
    read_coeff_from_file(coeff);
  } else{
    // Randomly initialize coefficient, Towan's way of initialization.
    for (int i = 0; i < 129; ++i)
      coeff[i] = u(mt);
  }
  // writing
  if (handler->write_coeff_into_file){
    write_coeff_into_file(coeff);
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

bElem *coeff;

void d3pt7(global_args *handler, Result *result) {
   unsigned *grid_ptr;

  auto bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB}, handler->read_grid_with_ghostzone_from_file, handler->write_grid_with_ghostzone_into_file);
  auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr;

  bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
  bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
  bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;

  auto bSize = cal_size<BDIM>::value;
  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
  Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

  // copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);

// auto arr_func = [&arr_in, &arr_out]() -> void {
//     _TILEFOR arr_out[k][j][i] = coeff[5] * arr_in[k + 1][j][i] + coeff[6] * arr_in[k - 1][j][i] +
//                                 coeff[3] * arr_in[k][j + 1][i] + coeff[4] * arr_in[k][j - 1][i] +
//                                 coeff[1] * arr_in[k][j][i + 1] + coeff[2] * arr_in[k][j][i - 1] +
//                                 coeff[0] * arr_in[k][j][i];
//   };

// #define bIn(i, j, k) arr_in[k][j][i]
// #define bOut(i, j, k) arr_out[k][j][i]
//   auto arr_tile_func = [&arr_in, &arr_out]() -> void {
//     #pragma omp parallel for
//     for (long tk = GZ; tk < STRIDE - GZ; tk += TILE)
//     for (long tj = GZ; tj < STRIDE - GZ; tj += TILE)
//     for (long ti = GZ; ti < STRIDE - GZ; ti += TILE)
//       tile("7pt.py", "FLEX", (BDIM), ("tk", "tj", "ti"), (1,1,4));
//   };
// #undef bIn
// #undef bOut

  auto brick_func = [&grid, &bIn, &bOut]() -> void {
    _PARFOR
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj)
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          for (long k = 0; k < TILE; ++k)
            for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                bOut[b][k][j][i] = coeff[5] * bIn[b][k + 1][j][i] + coeff[6] * bIn[b][k - 1][j][i] +
                                   coeff[3] * bIn[b][k][j + 1][i] + coeff[4] * bIn[b][k][j - 1][i] +
                                   coeff[1] * bIn[b][k][j][i + 1] + coeff[2] * bIn[b][k][j][i - 1] +
                                   coeff[0] * bIn[b][k][j][i];
              }
        }
  };

  // auto brick_func_trans = [&grid, &bIn, &bOut]() -> void {
  //   _PARFOR
  //   for (long tk = GB; tk < STRIDEB - GB; ++tk)
  //     for (long tj = GB; tj < STRIDEB - GB; ++tj)
  //       for (long ti = GB; ti < STRIDEB - GB; ++ti) {
  //         unsigned b = grid[tk][tj][ti];
  //         brick("7pt.py", VSVEC, (BDIM), (VFOLD), b);
  //       }
  // };

  std::cout << "\n Running - d3pt7" << std::endl;
  // arr_func();
  brick_func();


    std::ofstream outfile("dump_orig.txt");
    
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj){
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
                unsigned b = grid[tk][tj][ti];
                // Print inside a block/brick
                for (long k = 0; k < TILE; ++k) {
                    for (long j = 0; j < TILE; ++j) {
                        for (long i = 0; i < TILE; ++i) {
                             outfile<< bIn[b][k][j][i] << " ";
                        }
                    }
                }
                outfile << std::endl;
              }
              
          }

  // if (!compareBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
  //   throw std::runtime_error("\nCompare result mismatch(out_ptr, bOut)(array, brick)!\n");
  // else
  //   std::cout << "\n Compare Results match(out_ptr, bOut)(array, brick)\n";

//  if (!verifyBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, grid_ptr, bIn, grid_ptr, bOut))
//     std::cout << "\n Verification mismatch inside";
//   else
//     std::cout << "\nVerification results match inside(bout1, bOut2)";
  // free(in_ptr);
  // free(out_ptr);
  // if these are freed can lead to UB.
  // free(grid_ptr);
  // free(bInfo.adj);

    // Assign values to the members of the result struct
    result->bOut =  &bOut; // Assuming bOut is defined in the function
    result->grid_ptr = (unsigned *)grid_ptr;
    result->bIn = bIn; // Assuming in_ptr is defined in the function

}

int main(int argc, char **argv) {
  // arg_handler is a global struct containing various flags.
  global_args arg_handler = {0};
  // allocate space for coefficients
  coeff = (bElem *) malloc(129 * sizeof(bElem));
  

  // handle_argument_parsing(argc, argv, &arg_handler);
  arg_handler.write_coeff_into_file = 1;
  arg_handler.read_coeff_from_file = 0;
  arg_handler.write_grid_with_ghostzone_into_file = 1;
  arg_handler.read_grid_with_ghostzone_from_file = 0;  
  // handle coefficients
  handle_coefficient_data(coeff, &arg_handler);
  // Run once and collect data.
  // unsigned *grid_ptr1;
  Result *brick1_output = (Result*)malloc(sizeof(Result));
  d3pt7(&arg_handler, brick1_output); //, grid_ptr1);  // dumps and runs first
  

    std::ofstream outfile("dump.txt");
    auto grid = (unsigned (*)[STRIDEB][STRIDEB]) brick1_output->grid_ptr;
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj){
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
                unsigned b = grid[tk][tj][ti];
                // Print inside a block/brick
                for (long k = 0; k < TILE; ++k) {
                    for (long j = 0; j < TILE; ++j) {
                        for (long i = 0; i < TILE; ++i) {
                             outfile<< brick1_output->bIn[b][k][j][i] << " ";
                        }
                    }
                }
                outfile << std::endl;
              }
              
          }

   
  // std::cout << "\n CDC Brick";
  // // change the parameters
  // arg_handler.write_coeff_into_file = 0;
  // arg_handler.read_coeff_from_file = 1;
  // arg_handler.write_grid_with_ghostzone_into_file = 0;
  // arg_handler.read_grid_with_ghostzone_from_file = 1;
  // handle_coefficient_data(coeff, &arg_handler);
  // Result *brick2_output = (Result*)malloc(sizeof(Result));
  // d3pt7(&arg_handler, brick2_output); // , grid_ptr2);
  

  // if (!verifyBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, brick1_output->grid_ptr, brick1_output->bIn, brick2_output->grid_ptr, brick2_output->bOut)) 
  //   throw std::runtime_error("\nVerification result mismatch outside!");
  // else
  //   std::cout << "\nVerification results match(bout1, bOut2) outside";



  return 0;
}

