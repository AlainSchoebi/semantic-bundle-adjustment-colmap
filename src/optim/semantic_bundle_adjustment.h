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

#ifndef COLMAP_SRC_OPTIM_SEMANTIC_BUNDLE_ADJUSTMENT_H_
#define COLMAP_SRC_OPTIM_SEMANTIC_BUNDLE_ADJUSTMENT_H_

#include <memory>
#include <unordered_set>

#include <Eigen/Core>

#include <ceres/ceres.h>

#include "PBA/pba.h"
#include "base/camera_rig.h"
#include "base/reconstruction.h"
#include "util/alignment.h"

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// SemanticBundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

struct SemanticBundleAdjustmentOptions {
  // Loss function types: Trivial (non-robust) and Cauchy (robust) loss.
  enum class LossFunctionType { TRIVIAL, SOFT_L1, CAUCHY };
  LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

  // Data path
  // The data folder containing the depth maps and semantic maps as .tiff files.
  std::string data_path;

  // Output path
  // Coming from the options --output_path, not actually a GSBA option.
  std::string output_path = "";

  // Visualization path
  // Output folder containing that will contain some visualization files.
  std::string visualization_path = "{output_path}/run";

  // Export CSV
  // Boolean indicating whether or not visualizlation .csv files should be
  // exported at every optimization iteration.
  bool export_csv = false;

  // Depth error threshold
  // The threshold when reprojecting pixels to another image for comparing the
  // depth values.
  double depth_error_threshold = 2;

  // Error computation pixel step
  // The number of pixels horizontally and vertically between two pixels
  // coordinates for which the semantic error is being computed.
  int error_computation_pixel_step = 10;

  // Numeric relative step size
  // Numeric differentiation step size (multiplied by parameter block's
  // order of magnitude). If parameters are close to zero, the step size
  // is set to sqrt(machine_epsilon).
  double numeric_relative_step_size = 1e-3;

  // Scaling factor determines residual at which robustification takes place.
  double loss_function_scale = 1.0;

  // Whether to refine the focal length parameter group.
  bool refine_focal_length = true;

  // Whether to refine the principal point parameter group.
  bool refine_principal_point = false;

  // Whether to refine the extra parameter group.
  bool refine_extra_params = true;

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

  SemanticBundleAdjustmentOptions() {
    solver_options.function_tolerance = 1e-8;
    solver_options.gradient_tolerance = 1e-8;
    solver_options.parameter_tolerance = 1e-8;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.max_num_iterations = 100;
    solver_options.max_linear_solver_iterations = 200;
    solver_options.max_num_consecutive_invalid_steps = 10;
    solver_options.max_consecutive_nonmonotonic_steps = 10;
    solver_options.num_threads = -1;
    // For the parameters to be updated when used in the callback functions
    // during the optimization procedure
    solver_options.update_state_every_iteration = true;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
  }

  // Create a new loss function based on the specified options. The caller
  // takes ownership of the loss function.
  ceres::LossFunction* CreateLossFunction() const;

  bool Check() const;
};

////////////////////////////////////////////////////////////////////////////////
// SemanticBundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////
// Configuration container to setup bundle adjustment problems.
class SemanticBundleAdjustmentConfig {
 public:
  SemanticBundleAdjustmentConfig();

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

////////////////////////////////////////////////////////////////////////////////
// SemanticBundleAdjuster
////////////////////////////////////////////////////////////////////////////////
// Bundle adjustment based on Ceres-Solver. Enables most flexible configurations
// and provides best solution quality.
class SemanticBundleAdjuster {
 public:
  SemanticBundleAdjuster(const SemanticBundleAdjustmentOptions& options,
                         const SemanticBundleAdjustmentConfig& config);

  bool Solve(Reconstruction* reconstruction);

  // Get the Ceres solver summary for the last call to `Solve`.
  const ceres::Solver::Summary& Summary() const;

 private:
  void SetUp(Reconstruction* reconstruction,
             ceres::LossFunction* loss_function);

  void SetUpOutputFolders();

  void Assert(const Reconstruction* const reconstruction) const;

  void TearDown(Reconstruction* reconstruction);

  void PrintSetupInformation(Reconstruction* reconstruction);

  void AddImagePairToProblem(const image_t image_id_1, const image_t image_id_2,
                             Reconstruction* reconstruction,
                             ceres::LossFunction* loss_function);

  void SetUpManifolds(Reconstruction* reconstruction);

  bool ReadDepthAndSemanticMaps(const Reconstruction* const reconstruction);

  void ExportSemanticErrorToCSV(const Reconstruction* const reconstruction,
                                const std::string path_prefix) const;

 protected:
  const SemanticBundleAdjustmentOptions options_;
  SemanticBundleAdjustmentConfig config_;
  std::unique_ptr<ceres::Problem> problem_;
  ceres::Solver::Summary summary_;
  std::unordered_set<camera_t> camera_ids_;
  std::unordered_map<point3D_t, size_t> point3D_num_observations_;

  // Depth maps and semantic maps
  std::unordered_map<std::string, const Eigen::MatrixXf> depth_maps_;
  std::unordered_map<std::string, const Eigen::MatrixXf> semantic_maps_;

  // Output paths
  std::string optim_steps_folder_;

  // Counters
  int semantic_residual_block_counter = 0;
  int skipped_semantic_residual_block_counter = 0;

 public:
  friend class ceres::IterationCallback;
  friend class SBACallbackFunctor;
};

void PrintSolverSummary(const ceres::Solver::Summary& summary);
void PrintSemanticSolverSummary(const ceres::Solver::Summary& summary);

////////////////////////////////////////////////////////////////////////////////
// SBACallbackFunctor
////////////////////////////////////////////////////////////////////////////////
struct SBACallbackFunctor : public ceres::IterationCallback {
  SBACallbackFunctor(Reconstruction* reconstruction,
                     SemanticBundleAdjuster* sba);

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary);

 private:
  Reconstruction* callback_reconstruction_;
  SemanticBundleAdjuster* const callback_sba_;
};

}  // namespace colmap
#endif  // COLMAP_SRC_OPTIM_SEMANTIC_BUNDLE_ADJUSTMENT_H_
