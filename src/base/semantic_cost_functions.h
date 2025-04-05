// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_BASE_SEMANTIC_COST_FUNCTIONS_H_
#define COLMAP_SRC_BASE_SEMANTIC_COST_FUNCTIONS_H_

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "util/rotation_extension.h"

namespace colmap {

// ReprojectionStatus enum
enum ReprojectionStatus { OUT_OF_BOUNDS = -1, INVALID_DEPTH = -2, VALID = 10 };

////////////////////////////////////////////////////////////////////////////////
// BaseSemanticBACostFunction
////////////////////////////////////////////////////////////////////////////////
// CameraModel must be SimplePinholeCameraModel (other models not supported)
template <typename CameraModel = SimplePinholeCameraModel>
class BaseSemanticBACostFunction {
 public:
  explicit BaseSemanticBACostFunction(
      const std::string image_name_1, const std::string image_name_2,
      const double* const camera_params_1, const double* const camera_params_2,
      const Eigen::Vector2i& point2D_1,
      const std::unordered_map<std::string, const Eigen::MatrixXf>* const
          depth_maps_ptr,
      const std::unordered_map<std::string, const Eigen::MatrixXf>* const
          semantic_maps_ptr,
      const double depth_error_threshold)
      : point2D_1_(point2D_1),
        depth_1_(depth_maps_ptr->at(image_name_1)(point2D_1(1), point2D_1(0))),
        semantic_1_(
            semantic_maps_ptr->at(image_name_1)(point2D_1(1), point2D_1(0))),
        depth_map_2_(&depth_maps_ptr->at(image_name_2)),
        semantic_map_2_(&semantic_maps_ptr->at(image_name_2)),
        image_name_1_(image_name_1),
        image_name_2_(image_name_2),
        camera_params_1_(camera_params_1),
        camera_params_2_(camera_params_2),
        depth_error_threshold_(depth_error_threshold) {}

  virtual void printType() {
    std::cout << "BaseSemanticBACostFunction" << std::endl;
  }

  // Create ceres cost function
  static ceres::CostFunction* Create() { return nullptr; }

  // Operator ()
  template <typename T = double>
  bool operator()() {}

  // Semantic error computation
  template <typename T = double>  // T must be double (not Jet)
  bool compute_semantic_error(const T* const qvec_1, const T* const tvec_1,
                              const T* const qvec_2, const T* const tvec_2,
                              const T* const camera_params_1,
                              const T* const camera_params_2,
                              ReprojectionStatus* status,
                              Eigen::Ref<Eigen::Vector3d> return_point3D,
                              Eigen::Ref<Eigen::Vector2i> return_point2D_2,
                              T* semantic_error) const {
    // Ascertain T is double
    if (!std::is_same<T, double>::value) {
      std::cerr << "ERROR: T is not double." << std::endl;
      throw std::runtime_error("ERROR: T is not double.");
    }

    // Image points in pixels
    const T x_1 = static_cast<T>(this->point2D_1_(0));
    const T y_1 = static_cast<T>(this->point2D_1_(1));

    // Normalized world coordinates, i.e. (u, v, 1), focal length 1
    T u_1;
    T v_1;

    // Image to normalized world coordinates
    CameraModel::ImageToWorld<T>(camera_params_1, x_1, y_1, &u_1, &v_1);

    // Scale the 2D normalized point using the depth value -> 3D point
    T x_3D = u_1 * this->depth_1_;
    T y_3D = v_1 * this->depth_1_;
    T z_3D = static_cast<T>(this->depth_1_);
  // .radius_
    T point3D[3] = {x_3D, y_3D, z_3D};

    // Get pose inverse
    T qvec_1_inverse[4];
    T tvec_1_inverse[3];
    ceres::PoseInverse<T>(qvec_1, tvec_1, qvec_1_inverse, tvec_1_inverse);

    // Transform 3D point to world coordinates
    T point3D_w_world[3];
    ceres::PoseTransformPoint(qvec_1_inverse, tvec_1_inverse, point3D,
                              point3D_w_world);

    // Save 3D point coordinates
    return_point3D(0) = static_cast<double>(point3D_w_world[0]);
    return_point3D(1) = static_cast<double>(point3D_w_world[1]);
    return_point3D(2) = static_cast<double>(point3D_w_world[2]);

    // Transform 3D point to second camera frame
    T point3D_w_camera_2[3];
    ceres::PoseTransformPoint(qvec_2, tvec_2, point3D_w_world,
                              point3D_w_camera_2);

    // Project to normalized 2D coordinates, i.e. (u, v, 1)
    T uv_w_camera_2[2];
    uv_w_camera_2[0] = point3D_w_camera_2[0] / point3D_w_camera_2[2];
    uv_w_camera_2[1] = point3D_w_camera_2[1] / point3D_w_camera_2[2];
    T measured_depth_2 = point3D_w_camera_2[2];

    // Get image coordinates (i.e. pixel values)
    T x_2, y_2;

    // Distort and transform to pixel space of camera 2.
    CameraModel::WorldToImage(camera_params_2, uv_w_camera_2[0],
                              uv_w_camera_2[1], &x_2, &y_2);

    // Round the pixels values [IMPROVEMENT? don't round and interpolate]
    Eigen::Vector2i point2D_2;
    point2D_2(0) = static_cast<int>(std::round(x_2));
    point2D_2(1) = static_cast<int>(std::round(y_2));

    // Save 2D point
    return_point2D_2(0) = static_cast<int>(point2D_2(0));
    return_point2D_2(1) = static_cast<int>(point2D_2(1));

    // Check if the point2D_2 lies inside the image bounds of image 2
    int H = this->depth_map_2_->rows(), W = this->depth_map_2_->cols();
    if (point2D_2(0) < 0 || point2D_2(0) >= W || point2D_2(1) < 0 ||
        point2D_2(1) >= H) {
      *status = OUT_OF_BOUNDS;
      semantic_error[0] = T(0.);

#ifdef PRINT
      std::cout << "Skipped residual as the pixel landed outside of the image."
                << "The coordinates were x: " << point2D_2(0)
                << ", y: " << point2D_2(1)
                << ", which is outside of bounds (H, W) = (" << H << ", " << W
                << ")." << std::endl;
#endif
      return true;
    }

    // Access depth value (cast from float to double)
    T depth_2 =
        static_cast<T>(this->depth_map_2_->coeff(point2D_2(1), point2D_2(0)));

    // Check if the depth is valid
    if (std::fabs(depth_2 - measured_depth_2) > this->depth_error_threshold_) {
      *status = INVALID_DEPTH;
      semantic_error[0] = T(0.);
#ifdef PRINT
      std::cout << "Skipped residual as the depth error was "
                << std::fabs(depth_2 - measured_depth_2)
                << " > this->depth_error_threshold_ = "
                << this->depth_error_threshold_
                << ". The respective depths were " << depth_2 << " and "
                << measured_depth_2 << std::endl;
#endif
      return true;
    }

    // Acess semantic mask
    *status = VALID;
    float semantic_2 = this->semantic_map_2_->coeff(point2D_2(1), point2D_2(0));
    if (this->semantic_1_ == semantic_2) {
      semantic_error[0] = T(0);  // [IMPROVEMENT? don't use a 0/1 error]
    } else {
      semantic_error[0] = T(1);  // [IMPROVEMENT? don't use a 0/1 error]
    }

    return true;
  }

 protected:
  const Eigen::Vector2i point2D_1_;

  const std::string image_name_1_;
  const std::string image_name_2_;

  const double* camera_params_1_;
  const double* camera_params_2_;

  const float depth_1_;
  const float semantic_1_;
  const Eigen::MatrixXf* const depth_map_2_;
  const Eigen::MatrixXf* const semantic_map_2_;

  const double depth_error_threshold_;

 public:
  static const int DIM_RESIDUAL = 1;
};

////////////////////////////////////////////////////////////////////////////////
// SemanticBACostFunction
////////////////////////////////////////////////////////////////////////////////
// CameraModel must be SimplePinholeCameraModel (other models not supported)
template <typename CameraModel = SimplePinholeCameraModel>
class SemanticBACostFunction : public BaseSemanticBACostFunction<CameraModel> {
 public:
  // Constructor
  template <typename... Params>
  explicit SemanticBACostFunction(const Params&... params)
      : BaseSemanticBACostFunction<CameraModel>(params...) {}

  void printType() override {
    std::cout << "SemanticBACostFunction" << std::endl;
  }

  // Create ceres cost function
  template <typename... Params>
  static ceres::CostFunction* Create(const double numeric_relative_step_size,
                                     const Params&... params) {
    // Numerically differentiable cost function
    ceres::NumericDiffOptions numeric_diff_options;
    numeric_diff_options.relative_step_size = numeric_relative_step_size;

    return new ceres::NumericDiffCostFunction<
        SemanticBACostFunction<CameraModel>, ceres::CENTRAL, DIM_RESIDUAL, 4, 3,
        4, 3>(new SemanticBACostFunction<CameraModel>(params...),
              ceres::TAKE_OWNERSHIP, DIM_RESIDUAL, numeric_diff_options);
  }

  template <typename T = double>  // T must be double (not Jet)
  bool operator()(const T* const qvec_1, const T* const tvec_1,
                  const T* const qvec_2, const T* const tvec_2,
                  T* residuals) const {
    // Define return values
    Eigen::Vector3d point3D;
    Eigen::Vector2i point2D_2;
    ReprojectionStatus status;
    T semantic_error;

    // Compute semantic error
    this->compute_semantic_error<T>(
        qvec_1, tvec_1, qvec_2, tvec_2, this->camera_params_1_,
        this->camera_params_2_, &status, point3D, point2D_2, &semantic_error);

    // Set the residual
    residuals[0] = semantic_error;
    return true;
  }
};

////////////////////////////////////////////////////////////////////////////////
// ConstantFirstPoseSemanticBACostFunction
////////////////////////////////////////////////////////////////////////////////
// CameraModel must be SimplePinholeCameraModel (other models not supported)
template <typename CameraModel = SimplePinholeCameraModel>
class ConstantFirstPoseSemanticBACostFunction
    : public BaseSemanticBACostFunction<CameraModel> {
 public:
  // Constructor
  template <typename... Params>
  explicit ConstantFirstPoseSemanticBACostFunction(
      const Eigen::Vector4d& qvec_1, const Eigen::Vector3d& tvec_1,
      const Params&... params)
      : BaseSemanticBACostFunction<CameraModel>(params...),
        qvec_1_{qvec_1(0), qvec_1(1), qvec_1(2), qvec_1(3)},
        tvec_1_{tvec_1(0), tvec_1(1), tvec_1(2)} {}

  void printType() override {
    std::cout << "ConstantFirstPoseSemanticBACostFunction" << std::endl;
  }

  // Create ceres cost function
  template <typename... Params>
  static ceres::CostFunction* Create(const double numeric_relative_step_size,
                                     const Eigen::Vector4d& qvec_1,
                                     const Eigen::Vector3d& tvec_1,
                                     const Params&... params) {
    // Numerically differentiable cost function
    ceres::NumericDiffOptions numeric_diff_options;
    numeric_diff_options.relative_step_size = numeric_relative_step_size;
    return new ceres::NumericDiffCostFunction<
        ConstantFirstPoseSemanticBACostFunction<CameraModel>, ceres::CENTRAL,
        DIM_RESIDUAL, 4, 3>(
        new ConstantFirstPoseSemanticBACostFunction<CameraModel>(qvec_1, tvec_1,
                                                                 params...),
        ceres::TAKE_OWNERSHIP, DIM_RESIDUAL, numeric_diff_options);
  }

  template <typename T = double>  // T must be double (not Jet)
  bool operator()(const T* const qvec_2, const T* const tvec_2,
                  T* residuals) const {
    // Define return values
    Eigen::Vector3d point3D;
    Eigen::Vector2i point2D_2;
    ReprojectionStatus status;
    T semantic_error;

    // Compute semantic error
    this->compute_semantic_error<T>(
        this->qvec_1_, this->tvec_1_, qvec_2, tvec_2, this->camera_params_1_,
        this->camera_params_2_, &status, point3D, point2D_2, &semantic_error);

    // Set the residual
    residuals[0] = semantic_error;
    return true;
  }

 private:
  const double qvec_1_[4];
  const double tvec_1_[3];
};

////////////////////////////////////////////////////////////////////////////////
// ConstantSecondPoseSemanticBACostFunction
////////////////////////////////////////////////////////////////////////////////
// CameraModel must be SimplePinholeCameraModel (other models not supported)
template <typename CameraModel = SimplePinholeCameraModel>
class ConstantSecondPoseSemanticBACostFunction
    : public BaseSemanticBACostFunction<CameraModel> {
 public:
  // Constructor
  template <typename... Params>
  explicit ConstantSecondPoseSemanticBACostFunction(
      const Eigen::Vector4d& qvec_2, const Eigen::Vector3d& tvec_2,
      const Params&... params)
      : BaseSemanticBACostFunction<CameraModel>(params...),
        qvec_2_{qvec_2(0), qvec_2(1), qvec_2(2), qvec_2(3)},
        tvec_2_{tvec_2(0), tvec_2(1), tvec_2(2)} {}

  void printType() override {
    std::cout << "ConstantSecondPoseSemanticBACostFunction" << std::endl;
  }

  // Create ceres cost function
  template <typename... Params>
  static ceres::CostFunction* Create(const double numeric_relative_step_size,
                                     const Eigen::Vector4d& qvec_2,
                                     const Eigen::Vector3d& tvec_2,
                                     const Params&... params) {
    // Numerically differentiable cost function
    ceres::NumericDiffOptions numeric_diff_options;
    numeric_diff_options.relative_step_size = numeric_relative_step_size;
    return new ceres::NumericDiffCostFunction<
        ConstantSecondPoseSemanticBACostFunction<CameraModel>, ceres::CENTRAL,
        DIM_RESIDUAL, 4, 3>(
        new ConstantSecondPoseSemanticBACostFunction<CameraModel>(
            qvec_2, tvec_2, params...),
        ceres::TAKE_OWNERSHIP, DIM_RESIDUAL, numeric_diff_options);
  }

  template <typename T = double>  // T must be double (not Jet)
  bool operator()(const T* const qvec_1, const T* const tvec_1,
                  T* residuals) const {
    // Define return values
    Eigen::Vector3d point3D;
    Eigen::Vector2i point2D_2;
    ReprojectionStatus status;
    T semantic_error;

    // Compute semantic error
    this->compute_semantic_error<T>(
        qvec_1, tvec_1, this->qvec_2_, this->tvec_2_, this->camera_params_1_,
        this->camera_params_2_, &status, point3D, point2D_2, &semantic_error);

    // Set the residual
    residuals[0] = semantic_error;

    return true;
  }

 private:
  const double qvec_2_[4];
  const double tvec_2_[3];
};

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_SEMANTIC_COST_FUNCTIONS_H_
