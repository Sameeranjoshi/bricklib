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
 * @brief verify values between 2 bricks 
 * @tparam dims number of dimensions
 * @tparam T1 type for brick1
 * @tparam T2 type for brick2
 * @param dimlist dimensions, contiguous first
 * @param padding padding applied to array format (skipped)
 * @param ghost padding applied to array and brick (skipped)
 * @param grid_ptr1 the grid array contains indices of brick1
 * @param grid_ptr2 the grid array contains indices of brick2
 * @param brick1 the brick data structure
 * @param brick2 the brick data structure
 * @return False when not equal (with tolerance)
 */
template<unsigned dims, typename T1, typename T2>
inline bool
verifyBrick(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost,
    unsigned *grid_ptr1, T1 &brick1, unsigned *grid_ptr2, T2 &brick2) {
  bool ret = true;
  auto f = [&ret](bElem brick1, bElem brick2) -> void {
    double diff = std::abs(brick1 - brick2);
    bool r = (diff < BRICK_TOLERANCE) || (diff < (std::abs(brick1) + std::abs(brick2)) * BRICK_TOLERANCE);
    verifyBrick_b = (verifyBrick_b && r);
  };

#pragma omp parallel default(none)
  {
    verifyBrick_b = true;
  }
  iter_grid_verify<dims>(dimlist, padding, ghost, grid_ptr1, brick1, grid_ptr2, brick2, f);

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
 * @tparam T1
 * @tparam T2
 * @param dimlist
 * @param grid_ptr1
 * @param brick1
 * @param grid_ptr2
 * @param brick2
 * @return
 *
 * For parameters see verifyBrick(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost, unsigned *grid_ptr1, T1 &brick1, unsingned *grid_ptr2, T2 &brick2)
 */
template<unsigned dims, typename T1, typename T2>
inline bool
verifyBrick(const std::vector<long> &dimlist, unsigned *grid_ptr1,
             T1 &brick1, unsigned *grid_ptr2, T2 &brick2) {
  std::vector<long> padding(dimlist.size(), 0); // (size, init value)
  std::vector<long> ghost(dimlist.size(), 0);

  return verifyBrick<dims, T1, T2>(dimlist, padding, ghost, grid_ptr1, brick1, grid_ptr2, brick2);
}
#endif //BRICK_BRICKVERIFY_H
