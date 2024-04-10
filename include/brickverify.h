/**
 * @file
 * @brief Verify content from bricks with another brick
 */

#ifndef BRICK_BRICKVERIFY_H
#define BRICK_BRICKVERIFY_H

#include <iostream>
#include <cmath>
#include <iomanip>
#include "bricksetup.h"
#include "cmpconst.h"

extern bool verifyBrick_b;     ///< Thread-private verifier accumulator
extern bool verifyBrick_b_numerical;     ///< Thread-private verifier accumulator
extern int k0_l,k1_l,k2_l,k3_l,k4_l,k5_l,k6_l,k7_l,k8_l,k9_l, k10_l, k11_l, k12_l, k13_l, k14_l, k15_l, k16_l;

#define _PARFOR _Pragma("omp parallel for collapse(2)")
#pragma omp threadprivate(verifyBrick_b)
#pragma omp threadprivate(verifyBrick_b_numerical)
#pragma omp threadprivate(k0_l,k1_l,k2_l,k3_l,k4_l,k5_l,k6_l,k7_l,k8_l,k9_l, k10_l, k11_l, k12_l, k13_l, k14_l, k15_l, k16_l)
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
  int cnt = 0;
  int cnt_local=0;
  auto f = [&ret, &cnt_local](bElem brick1, bElem brick2) -> void {
    double diff = std::abs(brick1 - brick2);
    bool r = (diff < BRICK_TOLERANCE) || (diff < (std::abs(brick1) + std::abs(brick2)) * BRICK_TOLERANCE);
    verifyBrick_b = (verifyBrick_b && r);
    if (r == false){
      cnt_local++;
      std::cout << "\n" << std::fixed << std::setprecision(15) << brick1 << " - " << brick2 << "\n";
      double diff = std::abs(brick1 - brick2); 
      std::cout << "\n diff= " << diff << " tolerance= "<< BRICK_TOLERANCE << "\n";
    }
  };

#pragma omp parallel default(none)
  {
    verifyBrick_b = true;
  }
  iter_grid_verify<dims>(dimlist, padding, ghost, grid_ptr1, brick1, grid_ptr2, brick2, f);

#pragma omp parallel default(none) shared(ret, cnt)
  {
#pragma omp critical
    {
      if(verifyBrick_b == false){
        cnt++;
      }
      ret = ret && verifyBrick_b;
    }
  }
  std::cout << "\n Cnt of false = " << cnt;
  std::cout << "\n CNT local = " << cnt_local;
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


// Verify using numerical methods techniques

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
verifyBrick_numerical(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost,
    unsigned *grid_ptr1, T1 &brick1/*original*/, unsigned *grid_ptr2, T2 &brick2/*CDC*/) {

  int f_k0,f_k1,f_k2,f_k3,f_k4,f_k5,f_k6,f_k7,f_k8,f_k9, f_k10, f_k11, f_k12, f_k13, f_k14, f_k15, f_k16 = 0;
  // int shared_k=0;

  auto f = [&](bElem brick1/*original*/, bElem brick2/*CDC*/) -> void {
    // Don't normalize as we are not doing it elsewhere.
    // if we normalize we don't see interesting results, only 4,5,6 show up.
    // while((int)brick1!= 0 ){ //normalizes decimal numbers to original
    //   brick2=brick2/10.0;
    //   brick1=brick1/10.0;
    // }
    
    double kor = fabs(brick1 - brick2); // original - CDC
    std::cout << kor << std::endl;
    // if (kor<=0.0000000001) ki_l++;
         if (kor<=0.00000000000000001) k16_l++;
    else if (kor<=0.0000000000000001) k15_l++;
    else if (kor<=0.000000000000001) k14_l++;
    else if (kor<=0.00000000000001) k13_l++;
    else if (kor<=0.0000000000001) k12_l++;
    else if (kor<=0.000000000001) k11_l++;
    else if (kor<=0.00000000001) k10_l++;
    else if (kor<=0.0000000001) k9_l++;
    else if (kor<=0.000000001) k8_l++;
    else if (kor<=0.00000001) k7_l++;
    else if (kor<=0.0000001) k6_l++;
    else if (kor<=0.000001) k5_l++;
    else if (kor<=0.00001) k4_l++;
    else if (kor<=0.0001) k3_l++;
    else if (kor<=0.001) k2_l++;
    else if (kor<=0.01) k1_l++;
                  else k0_l++;
      // std::cout << "\nbrick1 = " << std::fixed << std::setprecision(15) << brick1;
      // std::cout << "\nbrick2 = " << std::fixed << std::setprecision(15) << brick2;
      // std::cout << "\n(b1-b2)= " << std::fixed << std::setprecision(15) << kor;    
    // SDD stands for significant decimal digit
    // std::cout <<  ", SDD0= " << k0_l << ", SDD1= " << k1_l << ", SDD2= " << k2_l << ", SDD3= " << k3_l << ", SDD4= " << k4_l << ", SDD5= " << k5_l << ", SDD6= " << k6_l << ", SDD7= " << k7_l << ", SDD8= " << k8_l << ", SDDi= " << ki_l << std::endl;
  };

#pragma omp parallel default(none)
  {
    k0_l = 0;
    k1_l = 0;
    k2_l = 0;
    k3_l = 0;
    k4_l = 0;
    k5_l = 0;
    k6_l = 0;
    k7_l = 0;
    k8_l = 0;
    k9_l = 0;
    k10_l = 0;
    k11_l = 0;
    k12_l = 0;
    k13_l = 0;
    k14_l = 0;
    k15_l = 0;
    k16_l = 0;
  }
  iter_grid_verify<dims>(dimlist, padding, ghost, grid_ptr1, brick1, grid_ptr2, brick2, f);

#pragma omp parallel default(none) shared(f_k0,f_k1,f_k2,f_k3,f_k4,f_k5,f_k6,f_k7,f_k8,f_k9, f_k10, f_k11, f_k12, f_k13, f_k14, f_k15, f_k16)
  {
#pragma omp critical
    {
      f_k0 = f_k0 + k0_l;
      f_k1 = f_k1 + k1_l;
      f_k2 = f_k2 + k2_l;
      f_k3 = f_k3 + k3_l;
      f_k4 = f_k4 + k4_l;
      f_k5 = f_k5 + k5_l;
      f_k6 = f_k6 + k6_l;
      f_k7 = f_k7 + k7_l;
      f_k8 = f_k8 + k8_l;
      f_k9 = f_k9 + k9_l;
      f_k10 = f_k10 + k10_l;
      f_k11 = f_k11 + k11_l;
      f_k12 = f_k12 + k12_l;
      f_k13 = f_k13 + k13_l;
      f_k14 = f_k14 + k14_l;
      f_k15 = f_k15 + k15_l;
      f_k16 = f_k16 + k16_l;                                                
    }
  }

std::cout << "\n\n Out of total " << brick1.bInfo->nbricks << " bricks "
          << " SDD0= " << k0_l
          << " SDD1= " << k1_l
          << " SDD2= " << k2_l
          << " SDD3= " << k3_l
          << " SDD4= " << k4_l
          << " SDD5= " << k5_l
          << " SDD6= " << k6_l
          << " SDD7= " << k7_l
          << " SDD8= " << k8_l
          << " SDD9= " << k9_l
          << " SDD10= " << k10_l
          << " SDD11= " << k11_l
          << " SDD12= " << k12_l
          << " SDD13= " << k13_l
          << " SDD14= " << k14_l
          << " SDD15= " << k15_l
          << " SDD16= " << k16_l << std::endl;


  // if k0,k1,k2,k3,k4 all 0 means verified, else error
  // why ?
  // lesser the SDD errors means it's a red flag, data is not precise
  // farther the error can be tolerated.
  return (k0_l == 0 && k1_l == 0 && k2_l == 0 && k3_l==0 && k4_l==0);
  
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
verifyBrick_numerical(const std::vector<long> &dimlist, unsigned *grid_ptr1,
             T1 &brick1, unsigned *grid_ptr2, T2 &brick2) {
  std::vector<long> padding(dimlist.size(), 0); // (size, init value)
  std::vector<long> ghost(dimlist.size(), 0);

  return verifyBrick<dims, T1, T2>(dimlist, padding, ghost, grid_ptr1, brick1, grid_ptr2, brick2);
}



template<unsigned dims, typename T1, typename T2>
inline void
print_both_Bricks_verify(const std::vector<long> &dimlist, const std::vector<long> &padding, const std::vector<long> &ghost,
    unsigned *grid_ptr1, T1 &brick1, unsigned *grid_ptr2, T2 &brick2) {
  bool ret = true;
  auto f = [&ret](bElem brick1, bElem brick2) -> void {
      std::cout << "\n" << std::fixed << std::setprecision(15) << brick1 << " - " << brick2 ;
      double diff = std::abs(brick1 - brick2); 
      std::cout << "\n diff= " << diff << " tolerance= "<< BRICK_TOLERANCE << "";    
  };

  iter_grid_verify<dims>(dimlist, padding, ghost, grid_ptr1, brick1, grid_ptr2, brick2, f);
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
inline void
print_both_Bricks_verify(const std::vector<long> &dimlist, unsigned *grid_ptr1,
             T1 &brick1, unsigned *grid_ptr2, T2 &brick2) {
  std::vector<long> padding(dimlist.size(), 0); // (size, init value)
  std::vector<long> ghost(dimlist.size(), 0);

  return print_both_Bricks_verify<dims, T1, T2>(dimlist, padding, ghost, grid_ptr1, brick1, grid_ptr2, brick2);
}


#endif //BRICK_BRICKVERIFY_H
