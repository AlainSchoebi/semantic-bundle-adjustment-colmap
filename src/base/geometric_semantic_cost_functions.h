#ifndef COLMAP_SRC_BASE_GEOMETRIC_SEMANTIC_COST_FUNCTIONS_H_
#define COLMAP_SRC_BASE_GEOMETRIC_SEMANTIC_COST_FUNCTIONS_H_

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "util/cylinder.h"
#include "util/cylinder_by_2_points.h"
#include "util/matrix_vis.h"
#include "util/rotation_extension.h"
#include "util/utils.h"
#include "util/xywh.h"

namespace colmap {

constexpr int DIM_RESIDUAL = 1;

////////////////////////////////////////////////////////////////////////////////
// GSBACostFunction
////////////////////////////////////////////////////////////////////////////////

class GSBACostFunction {
 public:
  explicit GSBACostFunction(const double* const camera_params,
                            const Eigen::MatrixXb* const semantic_map_bool)
      : camera_params_(camera_params), semantic_map_bool_(semantic_map_bool) {}

  // Create ceres cost function
  static ceres::CostFunction* Create(
      const double numeric_relative_step_size,
      const double* const camera_params,
      const Eigen::MatrixXb* const semantic_map_bool) {
    // Numerical options
    ceres::NumericDiffOptions numeric_diff_options;
    numeric_diff_options.relative_step_size = numeric_relative_step_size;

    // Build numerically differentiable cost function
    return new ceres::NumericDiffCostFunction<GSBACostFunction, ceres::CENTRAL,
                                              DIM_RESIDUAL, 4, 3, 4, 3, 1, 1>(
        new GSBACostFunction(camera_params, semantic_map_bool),
        ceres::TAKE_OWNERSHIP, DIM_RESIDUAL, numeric_diff_options);
  }

  template <typename T = double>
  bool operator()(const T* const camera_qvec, const T* const camera_tvec,
                  const T* const qvec, const T* const tvec,
                  const T* const radius, const T* const height,
                  T* residuals) const {
    // Ascertain T is double
    if (!std::is_same<T, double>::value) {
      std::cerr << "ERROR: T is not double." << std::endl;
      throw std::runtime_error("ERROR: T is not double.");
    }

    // Create new temporary cylinder
    Cylinder cylinder(qvec, tvec, *radius, *height);

    // Compute IoU
    double iou = cylinder.ComputeSemanticIoU<double>(camera_qvec, camera_tvec,
                                                     this->camera_params_,
                                                     this->semantic_map_bool_);

#ifdef ADDITIONAL_PRINT
    std::cout << " - Evaluate: camera tvec = (" << camera_tvec[0] << ", "
              << camera_tvec[1] << ", " << camera_tvec[2]
              << "), camera qvec = (" << camera_qvec[0] << ", "
              << camera_qvec[1] << ", " << camera_qvec[2] << ", "
              << camera_qvec[3] << "). \n       - cylinder tvec = (" << tvec[0]
              << ", " << tvec[1] << ", " << tvec[2] << "), qvec = (" << qvec[0]
              << ", " << qvec[1] << ", " << qvec[2] << ", " << qvec[3] << ")"
              << std::endl;
    std::cout << "        ==> IoU=" << iou << std::endl << std::endl;
#endif

#ifdef _DEBUG
    std::cout << "Evaluate: " << cylinder << " ==> IoU=" << iou << std::endl;
#endif

    // Compute the error
    residuals[0] = 1 - iou;
    return true;
  }

 private:
  static const int DIM_RESIDUAL = 1;

  const double* camera_params_;
  const Eigen::MatrixXb* const semantic_map_bool_;
};

////////////////////////////////////////////////////////////////////////////////
// ConstantPoseGSBACostFunction
////////////////////////////////////////////////////////////////////////////////

class ConstantPoseGSBACostFunction {
 public:
  explicit ConstantPoseGSBACostFunction(
      const double* const camera_qvec, const double* const camera_tvec,
      const double* const camera_params,
      const Eigen::MatrixXb* const semantic_map_bool)
      : camera_qvec_(camera_qvec),
        camera_tvec_(camera_tvec),
        camera_params_(camera_params),
        semantic_map_bool_(semantic_map_bool) {}

  // Create ceres cost function
  static ceres::CostFunction* Create(
      const double numeric_relative_step_size, const double* const camera_qvec,
      const double* const camera_tvec, const double* const camera_params,
      const Eigen::MatrixXb* const semantic_map_bool) {
    // Numerical options
    ceres::NumericDiffOptions numeric_diff_options;
    numeric_diff_options.relative_step_size = numeric_relative_step_size;

    // Build numerically differentiable cost function
    return new ceres::NumericDiffCostFunction<
        ConstantPoseGSBACostFunction, ceres::CENTRAL, DIM_RESIDUAL, 4, 3, 1, 1>(
        new ConstantPoseGSBACostFunction(camera_qvec, camera_tvec,
                                         camera_params, semantic_map_bool),
        ceres::TAKE_OWNERSHIP, DIM_RESIDUAL, numeric_diff_options);
  }

  template <typename T = double>
  bool operator()(const T* const qvec, const T* const tvec,
                  const T* const radius, const T* const height,
                  T* residuals) const {
    // Ascertain T is double
    if (!std::is_same<T, double>::value) {
      std::cerr << "ERROR: T is not double." << std::endl;
      throw std::runtime_error("ERROR: T is not double.");
    }

    // Create new temporary cylinder
    Cylinder cylinder(qvec, tvec, *radius, *height);

    // Compute IoU
    double iou = cylinder.ComputeSemanticIoU<double>(
        this->camera_qvec_, this->camera_tvec_, this->camera_params_,
        this->semantic_map_bool_);

#ifdef _DEBUG
    std::cout << "Evaluate: " << cylinder << " ==> IoU=" << iou << std::endl;
#endif

    // Compute the error
    residuals[0] = 1 - iou;
    return true;
  }

 private:
  static const int DIM_RESIDUAL = 1;

  const double* camera_qvec_;
  const double* camera_tvec_;
  const double* camera_params_;
  const Eigen::MatrixXb* const semantic_map_bool_;
};

////////////////////////////////////////////////////////////////////////////////
// ConstantCylinderGSBACostFunction
////////////////////////////////////////////////////////////////////////////////

template <typename CylinderClass = Cylinder>
class ConstantCylinderGSBACostFunction {
 public:
  explicit ConstantCylinderGSBACostFunction(
      const CylinderClass& cylinder, const double* const camera_params,
      const Eigen::MatrixXb* const semantic_map_bool)
      : cylinder_(cylinder),
        camera_params_(camera_params),
        semantic_map_bool_(semantic_map_bool) {}

  template <typename T = Cylinder>
  // Create ceres cost function
  static ceres::CostFunction* Create(
      const double numeric_relative_step_size, const T& cylinder,
      const double* const camera_params,
      const Eigen::MatrixXb* const semantic_map_bool) {
    // Numerical options
    ceres::NumericDiffOptions numeric_diff_options;
    numeric_diff_options.relative_step_size = numeric_relative_step_size;

    // Build numerically differentiable cost function
    return new ceres::NumericDiffCostFunction<
        ConstantCylinderGSBACostFunction<T>, ceres::CENTRAL, DIM_RESIDUAL, 4,
        3>(new ConstantCylinderGSBACostFunction<T>(cylinder, camera_params,
                                                   semantic_map_bool),
           ceres::TAKE_OWNERSHIP, DIM_RESIDUAL, numeric_diff_options);
  }

  template <typename T = double>
  bool operator()(const T* const camera_qvec, const T* const camera_tvec,
                  T* residuals) const {
    // Ascertain T is double
    if (!std::is_same<T, double>::value) {
      std::cerr << "ERROR: T is not double." << std::endl;
      throw std::runtime_error("ERROR: T is not double.");
    }

    // Compute IoU
    double iou = this->cylinder_.ComputeSemanticIoU<double>(
        camera_qvec, camera_tvec, this->camera_params_,
        this->semantic_map_bool_);

#ifdef _DEBUG
    std::cout << "Evaluate constant cylinder  ==> IoU=" << iou << std::endl;
#endif

    // Compute the error
    residuals[0] = 1 - iou;
    return true;
  }

 private:
  static const int DIM_RESIDUAL = 1;

  const CylinderClass& cylinder_;
  const double* camera_params_;

  const Eigen::MatrixXb* const semantic_map_bool_;
};

////////////////////////////////////////////////////////////////////////////////
// GSBACostFunctionBy2Points
////////////////////////////////////////////////////////////////////////////////

class GSBACostFunctionBy2Points {
 public:
  explicit GSBACostFunctionBy2Points(
      const double* const camera_params,
      const Eigen::MatrixXb* const semantic_map_bool)
      : camera_params_(camera_params), semantic_map_bool_(semantic_map_bool) {}

  // Create ceres cost function
  static ceres::CostFunction* Create(
      const double numeric_relative_step_size,
      const double* const camera_params,
      const Eigen::MatrixXb* const semantic_map_bool) {
    // Numerical options
    ceres::NumericDiffOptions numeric_diff_options;
    numeric_diff_options.relative_step_size = numeric_relative_step_size;

    // Build numerically differentiable cost function
    return new ceres::NumericDiffCostFunction<
        GSBACostFunctionBy2Points, ceres::CENTRAL, DIM_RESIDUAL, 4, 3, 3, 3, 1>(
        new GSBACostFunctionBy2Points(camera_params, semantic_map_bool),
        ceres::TAKE_OWNERSHIP, DIM_RESIDUAL, numeric_diff_options);
  }

  template <typename T = double>
  bool operator()(const T* const camera_qvec, const T* const camera_tvec,
                  const T* const tvec_1, const T* const tvec_2,
                  const T* const radius, T* residuals) const {
    // Create new temporary cylinder
    CylinderBy2Points cylinder(tvec_1, tvec_2, *radius);

    // Compute IoU
    double iou = cylinder.ComputeSemanticIoU<double>(camera_qvec, camera_tvec,
                                                     this->camera_params_,
                                                     this->semantic_map_bool_);

#ifdef _DEBUG
    std::cout << "Evaluate: " << cylinder << " ==> IoU=" << iou << std::endl;
#endif

    // Compute the error
    residuals[0] = 1 - iou;
    return true;
  }

 private:
  static const int DIM_RESIDUAL = 1;

  const double* camera_params_;
  const Eigen::MatrixXb* const semantic_map_bool_;
};

////////////////////////////////////////////////////////////////////////////////
// ConstantPoseGSBACostFunctionBy2Points
////////////////////////////////////////////////////////////////////////////////

class ConstantPoseGSBACostFunctionBy2Points {
 public:
  explicit ConstantPoseGSBACostFunctionBy2Points(
      const double* const camera_qvec, const double* const camera_tvec,
      const double* const camera_params,
      const Eigen::MatrixXb* const semantic_map_bool)
      : camera_qvec_(camera_qvec),
        camera_tvec_(camera_tvec),
        camera_params_(camera_params),
        semantic_map_bool_(semantic_map_bool) {}

  // Create ceres cost function
  static ceres::CostFunction* Create(
      const double numeric_relative_step_size, const double* const camera_qvec,
      const double* const camera_tvec, const double* const camera_params,
      const Eigen::MatrixXb* const semantic_map_bool) {
    // Numerical options
    ceres::NumericDiffOptions numeric_diff_options;
    numeric_diff_options.relative_step_size = numeric_relative_step_size;

    // Build numerically differentiable cost function
    return new ceres::NumericDiffCostFunction<
        ConstantPoseGSBACostFunctionBy2Points, ceres::CENTRAL, DIM_RESIDUAL, 3,
        3, 1>(new ConstantPoseGSBACostFunctionBy2Points(
                  camera_qvec, camera_tvec, camera_params, semantic_map_bool),
              ceres::TAKE_OWNERSHIP, DIM_RESIDUAL, numeric_diff_options);
  }

  template <typename T = double>
  bool operator()(const T* const tvec_1, const T* const tvec_2,
                  const T* const radius, T* residuals) const {
    // Create new temporary cylinder
    CylinderBy2Points cylinder(tvec_1, tvec_2, *radius);

    // Compute IoU
    double iou = cylinder.ComputeSemanticIoU<double>(
        this->camera_qvec_, this->camera_tvec_, this->camera_params_,
        this->semantic_map_bool_);

#ifdef _DEBUG
    std::cout << "Evaluate: " << cylinder << " ==> IoU=" << iou << std::endl;
#endif

    // Compute the error
    residuals[0] = 1 - iou;
    return true;
  }

 private:
  static const int DIM_RESIDUAL = 1;

  const double* camera_qvec_;
  const double* camera_tvec_;
  const double* camera_params_;
  const Eigen::MatrixXb* const semantic_map_bool_;
};
}  // namespace colmap

#endif  // COLMAP_SRC_BASE_GEOMETRIC_SEMANTIC_COST_FUNCTIONS_H_
