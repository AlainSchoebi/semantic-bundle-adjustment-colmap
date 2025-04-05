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

#ifndef COLMAP_SRC_OPTIM_GEOMETRIC_SEMANTIC_BUNDLE_ADJUSTMENT_H_
#define COLMAP_SRC_OPTIM_GEOMETRIC_SEMANTIC_BUNDLE_ADJUSTMENT_H_

#include <memory>
#include <unordered_set>

#include <Eigen/Core>

#include <ceres/ceres.h>

#include "PBA/pba.h"
#include "base/camera_rig.h"
#include "base/geometric_semantic_cost_functions.h"
#include "base/reconstruction.h"
#include "util/alignment.h"

namespace colmap {
enum class CylinderParametrization { DEFAULT, BY2POINTS };

struct GeometricSemanticBundleAdjustmentOptions {
  // Loss function types: Trivial (non-robust) and Cauchy (robust) loss.
  enum class LossFunctionType { TRIVIAL, SOFT_L1, CAUCHY };
  LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

  // Data path
  // The folder containing the depth maps and semantic maps as .tiff files.
  std::string data_path = "";

  // Output path
  // Coming from the options --output_path, not actually a GSBA option.
  std::string output_path = "";

  // Input geometry file
  // The input text file containing the description of the initial geometry
  std::string input_geometry = "";

  // Semantic class label of the tree trunk
  double trunk_semantic_class = 250.;

  // Visualization path
  // Output folder containing that will contain some visualization files.
  std::string visualization_path = "{output_path}/run";

  // Refine geometry
  // Boolean indicating whether the geometry components (i.e. the cylinder)
  // should be refined or not.
  bool refine_geometry = true;

  // Cylinder parametrization
  // The type of cylinder parametrization.
  std::string cylinder_parametrization = "default";

  CylinderParametrization GetCylinderParametrization() const {
    if (cylinder_parametrization == "default") {
      return CylinderParametrization::DEFAULT;
    } else if (cylinder_parametrization == "by_2_points") {
      return CylinderParametrization::BY2POINTS;
    }

    std::cerr << "ERROR: '" << cylinder_parametrization
              << "' is not a valid cylinder parametrization." << std::endl;
    throw std::runtime_error("ERROR: '" + cylinder_parametrization +
                             "' is not a valid cylinder parametrization.");
  }

  // Include landmarks error
  // Boolean indicating whether the landmarks also contribute to the cost
  // function or not.
  bool include_landmark_error = false;

  // Landmark error weight
  // The weight associated to the landmark error, which is then divided by the
  // total number of 2D features.
  double landmark_error_weight = 1;

  // Numeric relative step size
  // Numeric differentiation step size (multiplied by parameter block's
  // order of magnitude). If parameters are close to zero, the step size
  // is set to sqrt(machine_epsilon).
  double numeric_relative_step_size = 1e-3;

  // Scaling factor determines residual at which robustification takes place.
  double loss_function_scale = 1.0;

  // Whether to refine the extrinsic parameter group.
  bool refine_extrinsics = true;

  // Whether to print a final summary.
  bool print_summary = true;

  // Minimum number of residuals to enable multi-threading. Note that
  // single-threaded is typically better for small bundle adjustment problems
  // due to the overhead of threading.
  int min_num_residuals_for_multi_threading = 50000;

  // Ceres-Solver options.
  ceres::Solver::Options solver_options;

  GeometricSemanticBundleAdjustmentOptions() {
    solver_options.function_tolerance = 0.0;
    solver_options.gradient_tolerance = 0.0;
    solver_options.parameter_tolerance = 0.0;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.max_num_iterations = 100;
    solver_options.max_linear_solver_iterations = 200;
    solver_options.max_num_consecutive_invalid_steps = 10;
    solver_options.max_consecutive_nonmonotonic_steps = 10;
    solver_options.num_threads = -1;
    // For the parameters to be updated when used in the callback functions
    // during the optimization procedure
    solver_options.update_state_every_iteration = true;
#if CERES_VERSION_MAJOR < 2 solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
  }

  // Create a new loss function based on the specified options. The caller
  // takes ownership of the loss function.
  ceres::LossFunction* CreateLossFunction() const;

  bool Check() const;
};

// Configuration container to setup bundle adjustment problems.
class GeometricSemanticBundleAdjustmentConfig {
 public:
  GeometricSemanticBundleAdjustmentConfig();

  size_t NumImages() const;
  size_t NumPoints() const;
  size_t NumConstantCameras() const;
  size_t NumConstantPoses() const;
  size_t NumConstantTvecs() const;
  size_t NumVariablePoints() const;
  size_t NumConstantPoints() const;

  // Determine the number of residuals for the given reconstruction. The number
  // of residuals equals the number of observations times two.
  size_t NumResiduals(const Reconstruction& reconstruction) const;

  // Add / remove images from the configuration.
  void AddImage(const image_t image_id);
  bool HasImage(const image_t image_id) const;
  void RemoveImage(const image_t image_id);

  // Set cameras of added images as constant or variable. By default all
  // cameras of added images are variable. Note that the corresponding images
  // have to be added prior to calling these methods.
  void SetConstantCamera(const camera_t camera_id);
  void SetVariableCamera(const camera_t camera_id);
  bool IsConstantCamera(const camera_t camera_id) const;

  // Set the pose of added images as constant. The pose is defined as the
  // rotational and translational part of the projection matrix.
  void SetConstantPose(const image_t image_id);
  void SetVariablePose(const image_t image_id);
  bool HasConstantPose(const image_t image_id) const;

  // Set the translational part of the pose, hence the constant pose
  // indices may be in [0, 1, 2] and must be unique. Note that the
  // corresponding images have to be added prior to calling these methods.
  void SetConstantTvec(const image_t image_id, const std::vector<int>& idxs);
  void RemoveConstantTvec(const image_t image_id);
  bool HasConstantTvec(const image_t image_id) const;

  // Add / remove points from the configuration. Note that points can either
  // be variable or constant but not both at the same time.
  void AddVariablePoint(const point3D_t point3D_id);
  void AddConstantPoint(const point3D_t point3D_id);
  bool HasPoint(const point3D_t point3D_id) const;
  bool HasVariablePoint(const point3D_t point3D_id) const;
  bool HasConstantPoint(const point3D_t point3D_id) const;
  void RemoveVariablePoint(const point3D_t point3D_id);
  void RemoveConstantPoint(const point3D_t point3D_id);

  // Access configuration data.
  const std::unordered_set<image_t>& Images() const;
  const std::unordered_set<point3D_t>& VariablePoints() const;
  const std::unordered_set<point3D_t>& ConstantPoints() const;
  const std::vector<int>& ConstantTvec(const image_t image_id) const;

 private:
  std::unordered_set<camera_t> constant_camera_ids_;
  std::unordered_set<image_t> image_ids_;
  std::unordered_set<point3D_t> variable_point3D_ids_;
  std::unordered_set<point3D_t> constant_point3D_ids_;
  std::unordered_set<image_t> constant_poses_;
  std::unordered_map<image_t, std::vector<int>> constant_tvecs_;
};

// Bundle adjustment based on Ceres-Solver. Enables most flexible configurations
// and provides best solution quality.
template <typename CylinderClass>
class GeometricSemanticBundleAdjuster {
 public:
  GeometricSemanticBundleAdjuster(
      const GeometricSemanticBundleAdjustmentOptions& options,
      const GeometricSemanticBundleAdjustmentConfig& config);

  bool Solve(Reconstruction* reconstruction);

  // Get the Ceres solver summary for the last call to `Solve`.
  const ceres::Solver::Summary& Summary() const;

 private:
  void SetUp(Reconstruction* reconstruction,
             ceres::LossFunction* loss_function);

  void SetUpGeometryError(Reconstruction* reconstruction);

  void SetUpLandmarkError(Reconstruction* reconstruction);

  void Assert(const Reconstruction* const reconstruction) const;

  void TearDown(Reconstruction* reconstruction);

  void AddImageToProblem(const image_t image_id, Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function);

  // For landmarks error
  void AddImageToProblemForLandmarks(const image_t image_id,
                                     Reconstruction* reconstruction,
                                     ceres::LossFunction* loss_function);
  void AddPointToProblemUsualForLandmarks(const point3D_t point3D_id,
                                          Reconstruction* reconstruction,
                                          ceres::LossFunction* loss_function);
  void ParameterizePointsForLandmarks(Reconstruction* reconstruction);

  void PrintSetupInformation(Reconstruction* reconstruction);

  void PrintCylinders(const Reconstruction* const reconstruction);

  void ExportCylindersToCSV(Reconstruction* reconstruction,
                            const std::string& csv_name,
                            const std::string& csv_name_circles);

  void SetUpOutputFolders();

  bool ReadDepthAndSemanticMaps(const Reconstruction* const reconstruction);

  // Manifolds
  void SetUpManifolds(Reconstruction* reconstruction);
  void SetUpCylinderManifolds();

 protected:
  const GeometricSemanticBundleAdjustmentOptions options_;
  GeometricSemanticBundleAdjustmentConfig config_;
  std::unique_ptr<ceres::Problem> problem_;
  ceres::Solver::Summary summary_;
  std::unordered_set<camera_t> camera_ids_;
  std::unordered_map<point3D_t, size_t> point3D_num_observations_;

  // Depth maps and semantic maps
  std::unordered_map<std::string, const Eigen::MatrixXf> depth_maps_;
  std::unordered_map<std::string, const Eigen::MatrixXf> semantic_maps_;
  std::unordered_map<std::string, const Eigen::MatrixXb> semantic_maps_bool_;

  // Cylinders
  std::vector<CylinderClass> cylinders_;

  // Output paths
  std::string image_folder_;
  std::string optim_steps_folder_;

 public:
  friend class ceres::IterationCallback;
  template <typename U>
  friend class MyClass;
};

void PrintSolverSummary(const ceres::Solver::Summary& summary);
void PrintGeometricSemanticSolverSummary(const ceres::Solver::Summary& summary);

template <typename CylinderClass>
class MyClass : public ceres::IterationCallback {
 public:
  MyClass(Reconstruction* reconstruction,
          GeometricSemanticBundleAdjuster<CylinderClass>* gsba);

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary);

 private:
  Reconstruction* const callback_reconstruction_;
  GeometricSemanticBundleAdjuster<CylinderClass>* const callback_gsba_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_GEOMETRIC_SEMANTIC_BUNDLE_ADJUSTMENT_H_
