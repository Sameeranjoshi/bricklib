/**
 * @file
 * @brief Brick iterator and setup code
 */

#ifndef BRICK_SETUP_H
#define BRICK_SETUP_H

#include <vector>
#include <typeinfo>
#include <initializer_list>
#include <algorithm>
#include <fstream>
#include <iostream>
#include "brick.h"
#include "grid_functions.h"

struct RunningTag {
};
struct StopTag {
};

template<unsigned select>
struct TagSelect {
  static constexpr RunningTag value = RunningTag();
};

template<>
struct TagSelect<0> {
  static constexpr StopTag value = StopTag();
};

template<unsigned dims, unsigned d>
inline void
init_fill(const std::vector<long> &stride, unsigned *adjlist, unsigned *grid_ptr, unsigned *low, unsigned *high,
          RunningTag t) {
  unsigned str = static_power<3, d - 1>::value;
  init_fill<dims, d - 1>(stride, adjlist, grid_ptr - stride[d - 1], low, high, TagSelect<d - 1>::value);
  init_fill<dims, d - 1>(stride, adjlist + str, grid_ptr, low, high, TagSelect<d - 1>::value);
  init_fill<dims, d - 1>(stride, adjlist + str * 2, grid_ptr + stride[d - 1], low, high, TagSelect<d - 1>::value);
}

template<unsigned dims, unsigned d>
inline void
init_fill(const std::vector<long> &stride, unsigned *adjlist, unsigned *grid_ptr, unsigned *low, unsigned *high,
          StopTag t) {
  if (grid_ptr >= low && grid_ptr < high)
    *adjlist = *grid_ptr;
  else
    *adjlist = 0;
}

template<unsigned dims, unsigned d>
inline void
init_iter(const std::vector<long> &dimlist, const std::vector<long> &stride, BrickInfo<dims> &bInfo, unsigned *grid_ptr,
          unsigned *low, unsigned *high, RunningTag t) {
  if (dims == d) {
#pragma omp parallel for
    for (long s = 0; s < dimlist[dims - d]; ++s)
      init_iter<dims, d - 1>(dimlist, stride, bInfo, grid_ptr + s * stride[dims - d], low, high,
                             TagSelect<d - 1>::value);
  } else {
    for (long s = 0; s < dimlist[dims - d]; ++s)
      init_iter<dims, d - 1>(dimlist, stride, bInfo, grid_ptr + s * stride[dims - d], low, high,
                             TagSelect<d - 1>::value);
  }
}

template<unsigned dims, unsigned d>
inline void
init_iter(const std::vector<long> &dimlist, const std::vector<long> &stride, BrickInfo<dims> &bInfo, unsigned *grid_ptr,
          unsigned *low, unsigned *high, StopTag t) {
  init_fill<dims, dims>(stride, bInfo.adj[*grid_ptr], grid_ptr, low, high, RunningTag());
}

template<unsigned dims>
BrickInfo<dims> init_grid(unsigned *&grid_ptr, const std::vector<long> &dimlist) {
  long size = 1;
  std::vector<long> stride;
  for (const auto a: dimlist) {
    stride.push_back(size);
    size *= a;
  }
  // allocate space and fill the grid_ptr reference.
  grid_ptr = (unsigned *) malloc(size * sizeof(unsigned));
  // if option is specified, use the data from file.
  // else option is not specified use from towan's synthetic data.
  // if option given to print/dump data - dump into file.
  fill_data_in_grid_default_way(grid_ptr, size);
  // print_data_in_grid_default_way(grid_ptr, size, "Printing after Towan");
  dump_data_from_grid_into_outputfile(grid_ptr, size, "output_grid_data_dmp.txt");
  fill_data_in_grid_from_inputfile(grid_ptr, size, "output_grid_data_dmp.txt");
  // print_data_in_grid_default_way(grid_ptr, size, "Printing after my read from file");

  BrickInfo<dims> bInfo(size);
  // some bookkeeping data structures, this deals with adjacency list.
  init_iter<dims, dims>(dimlist, stride, bInfo, grid_ptr, grid_ptr, grid_ptr + size, RunningTag());

  return bInfo;
}

// overloaded version 
template<unsigned dims>
BrickInfo<dims> init_grid(unsigned *&grid_ptr, const std::vector<long> &dimlist, int read_from_file, int write_into_file) {
  long size = 1;
  std::vector<long> stride;
  for (const auto a: dimlist) {
    stride.push_back(size);
    size *= a;
  }
  // allocate space and fill the grid_ptr reference.
  grid_ptr = (unsigned *) malloc(size * sizeof(unsigned));

  // changes
  if (read_from_file){
    fill_data_in_grid_from_inputfile(grid_ptr, size, "grid_contents.txt");
  } else{
    // towan's synthetic data.
    fill_data_in_grid_default_way(grid_ptr, size);
  }
  if (write_into_file){
    dump_data_from_grid_into_outputfile(grid_ptr, size, "grid_contents.txt");
  }

  // print_data_in_grid_default_way(grid_ptr, size, "Printing after Towan");
  // dump_data_from_grid_into_outputfile(grid_ptr, size, "output_grid_data_dmp.txt");
  // print_data_in_grid_default_way(grid_ptr, size, "Printing after my read from file");

  BrickInfo<dims> bInfo(size);
  // some bookkeeping data structures, this deals with adjacency list.
  init_iter<dims, dims>(dimlist, stride, bInfo, grid_ptr, grid_ptr, grid_ptr + size, RunningTag());

  return bInfo;
}

template<unsigned dims, unsigned d, typename F, typename A>
inline void fill(const std::vector<long> &tile, const std::vector<long> &stride, bElem *arr, A a, F f, RunningTag t) {
  for (long s = 0; s < tile[d - 1]; ++s)
    fill<dims, d - 1>(tile, stride, arr + s * stride[d - 1], a[s], f, TagSelect<d - 1>::value);
}

template<unsigned dims, unsigned d, typename F, typename A>
inline void fill(const std::vector<long> &tile, const std::vector<long> &stride, bElem *arr, A &a, F f, StopTag t) {
  f(a, arr);
}

template<unsigned dims, unsigned d, typename T, typename F>
inline void iter(const std::vector<long> &dimlist, const std::vector<long> &tile,
                 const std::vector<long> &strideA, const std::vector<long> &strideB,
                 const std::vector<long> &padding, const std::vector<long> &ghost,
                 T &brick, bElem *arr, unsigned *grid_ptr, F f, RunningTag t) {
  constexpr unsigned dimp = d - 1;
  if (dims == d) {
#pragma omp parallel for
    for (long s = ghost[dimp] / tile[dimp]; s < (dimlist[dimp] + ghost[dimp]) / tile[dimp]; ++s)
      iter<dims, d - 1>(dimlist, tile, strideA, strideB, padding, ghost, brick,
                        arr + (padding[dimp] + s * tile[dimp]) * strideA[dimp],
                        grid_ptr + s * strideB[dimp], f, TagSelect<dimp>::value);
  } else {
    for (long s = ghost[dimp] / tile[dimp]; s < (dimlist[dimp] + ghost[dimp]) / tile[dimp]; ++s)
      iter<dims, d - 1>(dimlist, tile, strideA, strideB, padding, ghost, brick,
                        arr + (padding[dimp] + s * tile[dimp]) * strideA[dimp],
                        grid_ptr + s * strideB[dimp], f, TagSelect<dimp>::value);
  }
}

template<unsigned dims, unsigned d, typename T, typename F>
inline void iter(const std::vector<long> &dimlist, const std::vector<long> &tile,
                 const std::vector<long> &strideA, const std::vector<long> &strideB,
                 const std::vector<long> &padding, const std::vector<long> &ghost,
                 T &brick, bElem *arr, unsigned *grid_ptr, F f, StopTag t) {
  fill<dims, dims>(tile, strideA, arr, brick[*grid_ptr], f, RunningTag());
}

/*
 * Iterate elements side by side in brick and arrays.
 *
 * dimlist: the internal regions, iterated
 * padding: the padding necessary for arrays, skipped
 * ghost: the padding for both, skipped
 * f: F (&bElem, *bElem) -> void
 */
template<unsigned dims, typename F, typename T, unsigned ... BDims>
inline void
iter_grid(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost,
          bElem *arr, unsigned *grid_ptr, Brick<Dim<BDims...>, T> &brick, F f) {
  std::vector<long> strideA;
  std::vector<long> strideB;
  std::vector<long> tile = {BDims...};
  // Arrays are contiguous first
  std::reverse(tile.begin(), tile.end());

  long sizeA = 1;
  long sizeB = 1;
  for (long a = 0; a < dimlist.size(); ++a) {
    strideA.push_back(sizeA);
    strideB.push_back(sizeB);
    sizeA *= (dimlist[a] + 2 * (padding[a] + ghost[a]));
    sizeB *= ((dimlist[a] + 2 * ghost[a]) / tile[a]);
  }

  iter<dims, dims>(dimlist, tile, strideA, strideB, padding, ghost, brick, arr, grid_ptr, f, RunningTag());
}


/**
 * @brief Copy values from an array to bricks
 * @tparam dims number of dimensions
 * @tparam T type for brick
 * @param dimlist dimensions, contiguous first
 * @param padding padding applied to array format (skipped)
 * @param ghost padding applied to array and brick (skipped)
 * @param arr array input
 * @param grid_ptr the grid array contains indices of bricks
 * @param brick the brick data structure
 */
template<unsigned dims, typename T>
inline void
copyToBrick(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost,
            bElem *arr, unsigned *grid_ptr, T &brick) {
  auto f = [](bElem &brick, bElem *arr) -> void {
    brick = *arr;
  };

  iter_grid<dims>(dimlist, padding, ghost, arr, grid_ptr, brick, f);
}

/**
 * @brief Copy values from an array to bricks without ghost or padding
 * @tparam dims
 * @tparam T
 * @param dimlist
 * @param arr
 * @param grid_ptr
 * @param brick
 *
 * For parameters see copyToBrick(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost, bElem *arr, unsigned *grid_ptr, T &brick)
 */
template<unsigned dims, typename T>
inline void copyToBrick(const std::vector<long> &dimlist, bElem *arr, unsigned *grid_ptr, T &brick) {
  std::vector<long> padding(dimlist.size(), 0);
  std::vector<long> ghost(dimlist.size(), 0);

  copyToBrick<dims>(dimlist, padding, ghost, arr, grid_ptr, brick);
}

/**
 * @brief Copy values from bricks to an array
 * @tparam dims number of dimensions
 * @tparam T type for brick
 * @param dimlist dimensions, contiguous first
 * @param padding padding applied to array format (skipped)
 * @param ghost padding applied to array and brick (skipped)
 * @param arr array input
 * @param grid_ptr the grid array contains indices of bricks
 * @param brick the brick data structure
 */
template<unsigned dims, typename T>
inline void copyFromBrick(const std::vector<long> &dimlist, const std::vector<long> &padding,
                          const std::vector<long> &ghost, bElem *arr, unsigned *grid_ptr, T &brick) {
  auto f = [](bElem &brick, bElem *arr) -> void {
    *arr = brick;
  };

  iter_grid<dims>(dimlist, padding, ghost, arr, grid_ptr, brick, f);
}

// ----------
// template<unsigned dims, unsigned d, typename F, typename B1, typename B2>
// inline void fill_verify_bug(const std::vector<long> &tile, const std::vector<long> &strideA, const std::vector<long> &strideB, 
//         B1 &brickelem1, B2 &brickelem2 , F f){ //  StopTag t) {
//           std::cout << "\n First";
//   f(brickelem1, brickelem2);
// }

// template<unsigned dims, unsigned d, typename F, typename B1, typename B2>
// inline void fill_verify_bug(const std::vector<long> &tile, const std::vector<long> &strideA, const std::vector<long> &strideB, 
//         B1 &brickelem1, B2 &brickelem2 , F f, StopTag t) {
//         std::cout << "\n Second stop";
//   // f(brickelem1, brickelem2);
// }
// template<unsigned dims, unsigned d, typename F, typename B1, typename B2>
// inline void fill_verify_bug(const std::vector<long> &tile, const std::vector<long> &strideA, const std::vector<long> &strideB, 
//         B1 &brickelem1, B2 &brickelem2 , F f,  RunningTag t) {
//         std::cout << "\n Second running";
//   // f(brickelem1, brickelem2);
// }

template<unsigned dims, unsigned d, typename F, typename B1, typename B2>
inline void fill_verify(const std::vector<long> &tile, const std::vector<long> &strideA, const std::vector<long> &strideB, 
        B1 brickelem1, B2 brickelem2, F f, RunningTag t) {
  for (long s = 0; s < tile[d - 1]; ++s)
    fill_verify<dims, d - 1>(tile, strideA, strideB, brickelem1[s], brickelem2[s], f, TagSelect<d - 1>::value); 
}

template<unsigned dims, unsigned d, typename F, typename B1, typename B2>
inline void fill_verify(const std::vector<long> &tile, const std::vector<long> &strideA, const std::vector<long> &strideB, 
        B1 &brickelem1, B2 &brickelem2, F f, StopTag t) {
        f(brickelem1, brickelem2);
}

template<unsigned dims, unsigned d /*2nd Dims*/, typename T1, typename T2, typename F>
inline void iter_verify(const std::vector<long> &dimlist, const std::vector<long> &tile,
                 const std::vector<long> &strideA, const std::vector<long> &strideB,
                 const std::vector<long> &padding, const std::vector<long> &ghost,
                 T1 &brick1, unsigned *grid_ptr1, T2 &brick2, unsigned *grid_ptr2, F f, RunningTag t) {
                  
  constexpr unsigned dimp = d - 1;
  if (dims == d) {
#pragma omp parallel for
    for (long s = ghost[dimp] / tile[dimp]; s < (dimlist[dimp] + ghost[dimp]) / tile[dimp]; ++s)
      iter_verify<dims, d - 1>(dimlist, tile, strideA, strideB, padding, ghost, brick1,
                        grid_ptr1 + s * strideA[dimp], brick2,  grid_ptr2 + s * strideB[dimp],  f, TagSelect<dimp>::value);
  } else {
    for (long s = ghost[dimp] / tile[dimp]; s < (dimlist[dimp] + ghost[dimp]) / tile[dimp]; ++s)
      iter_verify<dims, d - 1>(dimlist, tile, strideA, strideB, padding, ghost, brick1,  // Any Brick should be OK here I guess as shapes same, not tested yet?
                        grid_ptr1 + s * strideA[dimp], brick2, grid_ptr2 + s * strideB[dimp], f, TagSelect<dimp>::value);
  }
}

template<unsigned dims, unsigned d, typename T1, typename T2, typename F>
inline void iter_verify(const std::vector<long> &dimlist, const std::vector<long> &tile,
                 const std::vector<long> &strideA, const std::vector<long> &strideB,
                 const std::vector<long> &padding, const std::vector<long> &ghost,
                 T1 &brick1, unsigned *grid_ptr1, T2 &brick2, unsigned *grid_ptr2, F f, StopTag t) {   
  fill_verify<dims, dims>(tile, strideA, strideB, brick1[*grid_ptr1], brick2[*grid_ptr2], f, RunningTag());
}

/*
 * Iterate elements side by side in brick1 and brick2.
 *
 * dimlist: the internal regions, iterated
 * padding: the padding necessary for arrays, skipped
 * ghost: the padding for both, skipped
 * f: F (&bElem, &bElem) -> void
 */
template<unsigned dims, typename F, typename T1, typename T2, unsigned ... BDims>
inline void
iter_grid_verify(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost,
          unsigned *grid_ptr1, Brick<Dim<BDims...>, T1> &brick1, unsigned *grid_ptr2, Brick<Dim<BDims...>, T2> &brick2, F f) {
  std::vector<long> strideA;
  std::vector<long> strideB;
  std::vector<long> tile = {BDims...};
  // Arrays are contiguous first
  std::reverse(tile.begin(), tile.end());

  long sizeA = 1;
  long sizeB = 1;
  for (long a = 0; a < dimlist.size(); ++a) {
    strideA.push_back(sizeA);
    strideB.push_back(sizeB);
    sizeA *= ((dimlist[a] + 2 * ghost[a]) / tile[a]);
    sizeB *= ((dimlist[a] + 2 * ghost[a]) / tile[a]);
  }
  
  iter_verify<dims, dims>(dimlist, tile, strideA, strideB, padding, ghost, brick1, grid_ptr1, brick2, grid_ptr2, f, RunningTag());
}

template<unsigned dims, typename T>
inline void
write_brick_into_file_verify(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost,
            bElem *arr, unsigned *grid_ptr, T &brick, std::string &filename) {
  std::ofstream outfile(filename);
  auto f = [&](bElem &brick, bElem *arr) -> void {
    outfile << brick;
    outfile << std::endl;
  };

  iter_grid<dims>(dimlist, padding, ghost, arr, grid_ptr, brick, f);
  outfile.close();
}

/**
 * @brief Copy values from an array to bricks without ghost or padding
 * @tparam dims
 * @tparam T
 * @param dimlist
 * @param arr
 * @param grid_ptr
 * @param brick
 *
 * For parameters see copyToBrick(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost, bElem *arr, unsigned *grid_ptr, T &brick)
 */
template<unsigned dims, typename T>
inline void write_brick_into_file_verify(const std::vector<long> &dimlist, bElem *arr, unsigned *grid_ptr, T &brick, std::string &filename) {
  std::vector<long> padding(dimlist.size(), 0);
  std::vector<long> ghost(dimlist.size(), 0);

  write_brick_into_file_verify<dims>(dimlist, padding, ghost, arr, grid_ptr, brick, filename);
}

template<unsigned dims, typename T>
inline void
read_brick_from_file_verify(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost,
            bElem *arr, unsigned *grid_ptr, T &brick, std::string &filename) {
  std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Unable to open file " << filename  << "for reading data from." << std::endl;
        return;
    }  

  auto f = [&](bElem &brick, bElem *arr) -> void {
    if (!(infile >> brick)) {
      std::cerr << "Error reading data from file "<< filename << std::endl;
      exit(1);
    }   
  };

  iter_grid<dims>(dimlist, padding, ghost, arr, grid_ptr, brick, f);
  infile.close();
}


template<unsigned dims, typename T>
inline void read_brick_from_file_verify(const std::vector<long> &dimlist, bElem *arr, unsigned *grid_ptr, T &brick, std::string &filename) {
  std::vector<long> padding(dimlist.size(), 0);
  std::vector<long> ghost(dimlist.size(), 0);

  read_brick_from_file_verify<dims>(dimlist, padding, ghost, arr, grid_ptr, brick, filename);
}
#endif
