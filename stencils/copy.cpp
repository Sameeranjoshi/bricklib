//
// Created by Tuowen Zhao on 12/TILE/18.
//

#include "stencils.h"
#include <iostream>
#include <fstream>
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include "cpuvfold.h"


// Function to write the grid contents to a file
void writeGridToFile(unsigned (*grid)[STRIDEB][STRIDEB]) {
    // Open a file for writing
    std::ofstream outfile("grid_contents.txt");

    // Check if the file opened successfully
    if (!outfile.is_open()) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return;
    }

    // Write the contents of the grid to the file
    for (int i = 0; i < STRIDEB; ++i) {
        for (int j = 0; j < STRIDEB; ++j) {
            outfile << (*grid)[i][j] << " ";
        }
        outfile << std::endl; // Move to the next line after printing each row
    }

    outfile << std::endl; // Separate each STRIDEBxSTRIDEB slice with an empty line

    // Close the file
    outfile.close();

    std::cout << "\nGrid contents have been written to 'grid_contents.txt'\n" << std::endl;
}


void copy() {
  unsigned *grid_ptr;

  // Why is grid_ptr = 3D and grid=2D? THis seems interesting.
  auto bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
  auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_ptr;

  writeGridToFile(grid);
  // This part is all about array.
  // randomArray = allocates space for array and fills with random values.
  bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
  bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
  bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
  bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;

  auto bSize = cal_size<BDIM>::value;

  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize);
  Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);

// no brick used here. only array to array.
  auto arr_func = [&arr_in, &arr_out]() -> void {
    _TILEFOR arr_out[k][j][i] = arr_in[k][j][i];
  };

// reads from array and assigns to brick.
  auto to_func = [&grid, &bIn, &arr_in]() -> void {
    _PARFOR
    for (long tk = 0; tk < STRIDEB; ++tk)
      for (long tj = 0; tj < STRIDEB; ++tj)
        for (long ti = 0; ti < STRIDEB; ++ti) {
          unsigned b = grid[tk][tj][ti];  // This captures the brick index 
          for (long k = 0; k < TILE; ++k)
            for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                bIn[b][k][j][i] = arr_in[PADDING + tk * TILE + k][PADDING + tj * TILE + j][PADDING + ti * TILE + i];
              }
        }
  };

// reads from brick and assigns to array.
  auto from_func = [&grid, &bIn, &arr_out]() -> void {
    _PARFOR
    for (long tk = 0; tk < STRIDEB; ++tk)
      for (long tj = 0; tj < STRIDEB; ++tj)
        for (long ti = 0; ti < STRIDEB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          for (long k=0; k< TILE; k++)
          for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                arr_out[PADDING + tk * TILE + k][PADDING + tj * TILE + j][PADDING + ti * TILE + i] = bIn[b][k][j][i];
              }
        }
  };

  std::cout << "Copy" << std::endl;
  int cnt;
  std::cout << "Arr: " << time_func(arr_func) << std::endl;
  std::cout << "To: " << time_func(to_func) << std::endl;


  auto print_brick_stdout = [&grid, &bIn, &arr_out]() -> void {
    std::cout << "Hello";
    _PARFOR
    for (long tk = 0; tk < STRIDEB; ++tk)
      for (long tj = 0; tj < STRIDEB; ++tj)
        for (long ti = 0; ti < STRIDEB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          // print inside a block/brick.
          for (long k=0; k< TILE; k++)
          for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                std::cout << bIn[b][k][j][i] << " ";
              }
          std::cout << "\n";
        }
  };

  auto print_brick_file = [&grid, &bIn, &arr_out]() -> void {
    // Open a file for writing
    std::ofstream outfile("output_brick.txt");

    // Check if the file opened successfully
    if (!outfile.is_open()) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return;
    }


     _PARFOR
    for (long tk = 0; tk < STRIDEB; ++tk)
      for (long tj = 0; tj < STRIDEB; ++tj)
        for (long ti = 0; ti < STRIDEB; ++ti) {
          unsigned b = grid[tk][tj][ti];
          // print inside a block/brick.
          for (long k=0; k< TILE; k++)
          for (long j = 0; j < TILE; ++j)
              for (long i = 0; i < TILE; ++i) {
                outfile << bIn[b][k][j][i] << " ";
              }
          outfile << "\n";
        }

    outfile.close();

    std::cout << "\nBrick contents have been written to 'output_brick.txt'\n" << std::endl;

  };

  std::cout << "\nPrinting Brick to file\n";
  print_brick_file();
  

  // // SKip comparing bricks for now.
  // if (!compareBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, in_ptr, grid_ptr, bIn))
  //   throw std::runtime_error("result mismatch!");
  // // std::cout << "From: " << time_func(from_func) << std::endl;
  // if (!compareBrick<3>({N, N, N}, {PADDING,PADDING,PADDING}, {GZ, GZ, GZ}, out_ptr, grid_ptr, bIn))
  //   throw std::runtime_error("result mismatch!");

  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
  free(bInfo.adj);
}
