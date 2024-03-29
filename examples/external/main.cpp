#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include "brick.h"
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include <string.h>

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

// Utilities 

struct global_args {
  int write_coeff_into_file;
};
// global declaration allocates space automatically.
global_args arg_handler;

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
    
    std::cout << "\n Written coefficients.txt \n";
    // Close the file
    outfile.close();
}

void handle_coefficient_data(bElem *coeff, global_args *handler){

  
  std::random_device r;
  std::mt19937_64 mt(r());
  std::uniform_real_distribution<bElem> u(0, 1);

  // Randomly initialize coefficient, Towan's way of initialization.
  for (int i = 0; i < 129; ++i)
    coeff[i] = u(mt);

  if (handler->write_coeff_into_file){
    write_coeff_into_file(coeff);
  }

}

int handle_argument_parsing(int argc, char** argv, global_args *handler) {
    auto generic_error_msg = [&]() {
        std::cerr << "Usage: " << argv[0] << " <flags>" << std::endl;
        std::cerr << "flags = --dump-coeff=<true|false>, ..." << std::endl;
    };

    // Check the number of command-line arguments
    if (argc != 2) {
        generic_error_msg();
        return 1;
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

void d3pt7() {
  unsigned *grid_ptr;

  auto bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
  auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr;

  bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
  bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
  bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;

  auto bSize = cal_size<BDIM>::value;
  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
  Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

  copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);

  auto arr_func = [&arr_in, &arr_out]() -> void {
    _TILEFOR arr_out[k][j][i] = coeff[5] * arr_in[k + 1][j][i] + coeff[6] * arr_in[k - 1][j][i] +
                                coeff[3] * arr_in[k][j + 1][i] + coeff[4] * arr_in[k][j - 1][i] +
                                coeff[1] * arr_in[k][j][i + 1] + coeff[2] * arr_in[k][j][i - 1] +
                                coeff[0] * arr_in[k][j][i];
  };

#define bIn(i, j, k) arr_in[k][j][i]
#define bOut(i, j, k) arr_out[k][j][i]
  auto arr_tile_func = [&arr_in, &arr_out]() -> void {
    #pragma omp parallel for
    for (long tk = GZ; tk < STRIDE - GZ; tk += TILE)
    for (long tj = GZ; tj < STRIDE - GZ; tj += TILE)
    for (long ti = GZ; ti < STRIDE - GZ; ti += TILE)
      tile("7pt.py", "FLEX", (BDIM), ("tk", "tj", "ti"), (1,1,4));
  };
#undef bIn
#undef bOut

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

  auto brick_func_trans = [&grid, &bIn, &bOut]() -> void {
    _PARFOR
    for (long tk = GB; tk < STRIDEB - GB; ++tk)
      for (long tj = GB; tj < STRIDEB - GB; ++tj)
        for (long ti = GB; ti < STRIDEB - GB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          brick("7pt.py", VSVEC, (BDIM), (VFOLD), b);
        }
  };

  std::cout << "d3pt7" << std::endl;
  std::cout << "Arr: " << time_func(arr_func) << std::endl;
  std::cout << "Bri: " << time_func(brick_func) << std::endl;
  if (!compareBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");
  std::cout << "Arr Scatter: " << time_func(arr_tile_func) << std::endl;
  std::cout << "Trans: " << time_func(brick_func_trans) << std::endl;
  if (!compareBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bOut))
    throw std::runtime_error("result mismatch!");

  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
  free(bInfo.adj);
}

int main(int argc, char **argv) {
  // arg_handler is a global struct containing various flags.
  global_args arg_handler = {0};
  handle_argument_parsing(argc, argv, &arg_handler);
  // allocate space for coefficients
  coeff = (bElem *) malloc(129 * sizeof(bElem));
  handle_coefficient_data(coeff, &arg_handler);

  // brick1 = d3pt7();  // dumps and runs first
  // beick2 = d3pt7();  // reads CDC data and runs second time 
  // // compare here.
  // results  = comparenumerically(brick1, brick2);
  // if (results == true){
  //     Verified
  // }
  // else {
  //   not correct
  // }

  return 0;
}

