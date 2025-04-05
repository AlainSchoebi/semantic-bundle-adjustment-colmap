#ifndef COLMAP_SRC_UTIL_ROTATION_EXTENSION_H_
#define COLMAP_SRC_UTIL_ROTATION_EXTENSION_H_

#include <Eigen/Core>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

#include "glog/logging.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace ceres {

////////////////////////////////////////////////////////////////////////////////
// Definitions
////////////////////////////////////////////////////////////////////////////////

// Inverse pose computation
template <typename T>
inline void PoseInverse(const T q[4], const T t[3], T q_inv[4], T t_inv[3]);

// Quaternion inverse computation
template <typename T>
inline void QuaternionInverseRotation(const T q[4], T q_inverse[4]);

// Apply a transformation to a 3D point
template <typename T>
inline void PoseTransformPoint(const T q[4], const T t[3], const T pt[3],
                               T result[3]);

// Pose or 3D transformation chaining
template <typename T>
inline void PoseProduct(const T q_A[4], const T t_A[3], const T q_B[4],
                        const T t_B[3], T q_AB[4], T t_AB[3]);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void PoseInverse(const T q[4], const T t[3], T q_inv[4], T t_inv[3]) {
  // Inverse quaternion
  QuaternionInverseRotation<T>(q, q_inv);

  // Inverse rotation matrix
  T R_vec_inv[9];                          // row-major rotation matrix vector
  QuaternionToRotation(q_inv, R_vec_inv);  // row-major
  MatrixAdapter<T, 3, 1> R_inv = RowMajorAdapter3x3<T>(R_vec_inv);

  // Get inverse translation vector
  t_inv[0] = -(R_inv(0, 0) * t[0] + R_inv(0, 1) * t[1] + R_inv(0, 2) * t[2]);
  t_inv[1] = -(R_inv(1, 0) * t[0] + R_inv(1, 1) * t[1] + R_inv(1, 2) * t[2]);
  t_inv[2] = -(R_inv(2, 0) * t[0] + R_inv(2, 1) * t[1] + R_inv(2, 2) * t[2]);
}

template <typename T>
inline void QuaternionInverseRotation(const T q[4], T q_inverse[4]) {
  // 'scale' is 1 / norm(q).
  const T scale =
      T(1) / sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

  // Make unit-norm version of q.
  const T unit[4] = {
      scale * q[0],
      scale * q[1],
      scale * q[2],
      scale * q[3],
  };

  q_inverse[0] = unit[0];   // w
  q_inverse[1] = -unit[1];  // x
  q_inverse[2] = -unit[2];  // y
  q_inverse[3] = -unit[3];  // z
}

template <typename T>
inline void PoseTransformPoint(const T q[4], const T t[3], const T pt[3],
                               T result[3]) {
  QuaternionRotatePoint(q, pt, result);
  result[0] += t[0];
  result[1] += t[1];
  result[2] += t[2];
}

template <typename T>
inline void PoseProduct(const T q_A[4], const T t_A[3], const T q_B[4],
                        const T t_B[3], T q_AB[4], T t_AB[3]) {
  DCHECK_NE(q_A, q_AB) << "Inplace PoseProdut is not supported.";
  DCHECK_NE(t_A, t_AB) << "Inplace PoseProdut is not supported.";
  DCHECK_NE(q_B, q_AB) << "Inplace PoseProdut is not supported.";
  DCHECK_NE(t_B, t_AB) << "Inplace PoseProdut is not supported.";

  QuaternionProduct(q_A, q_B, q_AB);        // R_AB = R_A @ R_B
  PoseTransformPoint(q_A, t_A, t_B, t_AB);  // t_AB = R_A @ t_B + t_A
}

}  // namespace ceres

#endif  // COLMAP_SRC_UTIL_ROTATION_EXTENSION_H_
