/**
 * @file
 * @brief Verify content from bricks with another brick
 */

#ifndef BRICK_BRICKVERIFY_H
#define BRICK_BRICKVERIFY_H

#include <iostream>
#include <cmath>
#include "bricksetup.h"
#include "cmpconst.h"

extern bool verifyBrick_b;     ///< Thread-private verifier accumulator
#define _PARFOR _Pragma("omp parallel for collapse(2)")
#pragma omp threadprivate(verifyBrick_b)

/**
 * @brief Compare values between bricks and an array
 * @tparam dims number of dimensions
 * @tparam T type for brick
 * @param dimlist dimensions, contiguous first
 * @param padding padding applied to array format (skipped)
 * @param ghost padding applied to array and brick (skipped)
 * @param arr array input
 * @param grid_ptr the grid array contains indices of bricks
 * @param brick the brick data structure
 * @return False when not equal (with tolerance)
 */
template<unsigned dims, typename T>
inline bool
compareBrick(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost,
    bElem *arr, unsigned *grid_ptr, T &brick) {
  bool ret = true;
  auto f = [&ret](bElem &brick, const bElem *arr) -> void {
    double diff = std::abs(brick - *arr);
    bool r = (diff < BRICK_TOLERANCE) || (diff < (std::abs(brick) + std::abs(*arr)) * BRICK_TOLERANCE);
    verifyBrick_b = (verifyBrick_b && r);
  };

#pragma omp parallel default(none)
  {
    verifyBrick_b = true;
  }

  iter_grid<dims>(dimlist, padding, ghost, arr, grid_ptr, brick, f);

#pragma omp parallel default(none) shared(ret)
  {
#pragma omp critical
    {
      ret = ret && verifyBrick_b;
    }
  }

  return ret;
}

/**
 * @brief Verify all values between 2 bricks without ghost or padding
 * @tparam dims
 * @tparam T
 * @param dimlist
 * @param arr
 * @param grid_ptr
 * @param brick
 * @return
 *
 * For parameters see verifyBrick(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost, bElem *arr, unsigned *grid_ptr, T &brick)
 */
template<unsigned dims, typename T>
inline bool
verifyBrick(const std::vector<long> &dimlist, bElem *arr, unsigned *grid_ptr,
             T &brick) {
  std::vector<long> padding(dimlist.size(), 0); // (size, init value)
  std::vector<long> ghost(dimlist.size(), 0);

  return compareBrick<dims, T>(dimlist, padding, ghost, arr, grid_ptr, brick);
}
#endif //BRICK_BRICKVERIFY_H
