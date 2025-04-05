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

#include "optim/semantic_bundle_adjustment.h"

#include <iomanip>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/projection.h"
#include "base/semantic_cost_functions.h"
#include "util/matrix_vis.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"
#include "util/utils.h"

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// SemanticBundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

ceres::LossFunction* SemanticBundleAdjustmentOptions::CreateLossFunction()
    const {
  ceres::LossFunction* loss_function = nullptr;
  switch (loss_function_type) {
    case LossFunctionType::TRIVIAL:
      loss_function = new ceres::TrivialLoss();
      break;
    case LossFunctionType::SOFT_L1:
      loss_function = new ceres::SoftLOneLoss(loss_function_scale);
      break;
    case LossFunctionType::CAUCHY:
      loss_function = new ceres::CauchyLoss(loss_function_scale);
      break;
  }
  CHECK_NOTNULL(loss_function);
  return loss_function;
}

bool SemanticBundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GE(loss_function_scale, 0);
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// SemanticBundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////

SemanticBundleAdjustmentConfig::SemanticBundleAdjustmentConfig() {}

size_t SemanticBundleAdjustmentConfig::NumImages() const {
  return image_ids_.size();
}

size_t SemanticBundleAdjustmentConfig::NumPoints() const {
  return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t SemanticBundleAdjustmentConfig::NumConstantCameras() const {
  return constant_camera_ids_.size();
}

size_t SemanticBundleAdjustmentConfig::NumConstantPoses() const {
  return constant_poses_.size();
}

size_t SemanticBundleAdjustmentConfig::NumConstantTvecs() const {
  return constant_tvecs_.size();
}

size_t SemanticBundleAdjustmentConfig::NumVariablePoints() const {
  return variable_point3D_ids_.size();
}

size_t SemanticBundleAdjustmentConfig::NumConstantPoints() const {
  return constant_point3D_ids_.size();
}

size_t SemanticBundleAdjustmentConfig::NumResiduals(
    const Reconstruction& reconstruction) const {
  // Count the number of observations for all added images.
  size_t num_observations = 0;
  for (const image_t image_id : image_ids_) {
    num_observations += reconstruction.Image(image_id).NumPoints3D();
  }

  // Count the number of observations for all added 3D points that are not
  // already added as part of the images above.

  auto NumObservationsForPoint = [this,
                                  &reconstruction](const point3D_t point3D_id) {
    size_t num_observations_for_point = 0;
    const auto& point3D = reconstruction.Point3D(point3D_id);
    for (const auto& track_el : point3D.Track().Elements()) {
      if (image_ids_.count(track_el.image_id) == 0) {
        num_observations_for_point += 1;
      }
    }
    return num_observations_for_point;
  };

  for (const auto point3D_id : variable_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }
  for (const auto point3D_id : constant_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }

  return 2 * num_observations;
}

void SemanticBundleAdjustmentConfig::AddImage(const image_t image_id) {
  image_ids_.insert(image_id);
}

bool SemanticBundleAdjustmentConfig::HasImage(const image_t image_id) const {
  return image_ids_.find(image_id) != image_ids_.end();
}

void SemanticBundleAdjustmentConfig::RemoveImage(const image_t image_id) {
  image_ids_.erase(image_id);
}

void SemanticBundleAdjustmentConfig::SetConstantCamera(
    const camera_t camera_id) {
  constant_camera_ids_.insert(camera_id);
}

void SemanticBundleAdjustmentConfig::SetVariableCamera(
    const camera_t camera_id) {
  constant_camera_ids_.erase(camera_id);
}

bool SemanticBundleAdjustmentConfig::IsConstantCamera(
    const camera_t camera_id) const {
  return constant_camera_ids_.find(camera_id) != constant_camera_ids_.end();
}

void SemanticBundleAdjustmentConfig::SetConstantPose(const image_t image_id) {
  CHECK(HasImage(image_id));
  CHECK(!HasConstantTvec(image_id));
  constant_poses_.insert(image_id);
}

void SemanticBundleAdjustmentConfig::SetVariablePose(const image_t image_id) {
  constant_poses_.erase(image_id);
}

bool SemanticBundleAdjustmentConfig::HasConstantPose(
    const image_t image_id) const {
  return constant_poses_.find(image_id) != constant_poses_.end();
}

void SemanticBundleAdjustmentConfig::SetConstantTvec(
    const image_t image_id, const std::vector<int>& idxs) {
  CHECK_GT(idxs.size(), 0);
  CHECK_LE(idxs.size(), 3);
  CHECK(HasImage(image_id));
  CHECK(!HasConstantPose(image_id));
  CHECK(!VectorContainsDuplicateValues(idxs))
      << "Tvec indices must not contain duplicates";
  constant_tvecs_.emplace(image_id, idxs);
}

void SemanticBundleAdjustmentConfig::RemoveConstantTvec(
    const image_t image_id) {
  constant_tvecs_.erase(image_id);
}

bool SemanticBundleAdjustmentConfig::HasConstantTvec(
    const image_t image_id) const {
  return constant_tvecs_.find(image_id) != constant_tvecs_.end();
}

const std::unordered_set<image_t>& SemanticBundleAdjustmentConfig::Images()
    const {
  return image_ids_;
}

const std::unordered_set<point3D_t>&
SemanticBundleAdjustmentConfig::VariablePoints() const {
  return variable_point3D_ids_;
}

const std::unordered_set<point3D_t>&
SemanticBundleAdjustmentConfig::ConstantPoints() const {
  return constant_point3D_ids_;
}

const std::vector<int>& SemanticBundleAdjustmentConfig::ConstantTvec(
    const image_t image_id) const {
  return constant_tvecs_.at(image_id);
}

void SemanticBundleAdjustmentConfig::AddVariablePoint(
    const point3D_t point3D_id) {
  CHECK(!HasConstantPoint(point3D_id));
  variable_point3D_ids_.insert(point3D_id);
}

void SemanticBundleAdjustmentConfig::AddConstantPoint(
    const point3D_t point3D_id) {
  CHECK(!HasVariablePoint(point3D_id));
  constant_point3D_ids_.insert(point3D_id);
}

bool SemanticBundleAdjustmentConfig::HasPoint(
    const point3D_t point3D_id) const {
  return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
}

bool SemanticBundleAdjustmentConfig::HasVariablePoint(
    const point3D_t point3D_id) const {
  return variable_point3D_ids_.find(point3D_id) != variable_point3D_ids_.end();
}

bool SemanticBundleAdjustmentConfig::HasConstantPoint(
    const point3D_t point3D_id) const {
  return constant_point3D_ids_.find(point3D_id) != constant_point3D_ids_.end();
}

void SemanticBundleAdjustmentConfig::RemoveVariablePoint(
    const point3D_t point3D_id) {
  variable_point3D_ids_.erase(point3D_id);
}

void SemanticBundleAdjustmentConfig::RemoveConstantPoint(
    const point3D_t point3D_id) {
  constant_point3D_ids_.erase(point3D_id);
}

////////////////////////////////////////////////////////////////////////////////
// SemanticBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

SemanticBundleAdjuster::SemanticBundleAdjuster(
    const SemanticBundleAdjustmentOptions& options,
    const SemanticBundleAdjustmentConfig& config)
    : options_(options), config_(config) {
  CHECK(options_.Check());
}

void SemanticBundleAdjuster::PrintSetupInformation(
    Reconstruction* reconstruction) {
  // Print information
  std::cout << "Setup the Semantic BA problem, with:" << std::endl;

  // Debug or release mode
#ifdef _DEBUG
  std::cout << "  - DEBUG mode." << std::endl;
#else
  std::cout << "  - RELEASE mode." << std::endl;
#endif

  // Print data folder (h5 folder)
  if (!ExistsDir(this->options_.data_path)) {
    std::cerr << "ERROR: `datafolder` is not a directory." << std::endl;
    throw std::runtime_error("ERROR: `datafolder` is not valid.");
  }
  if (!ExistsDir(this->options_.data_path + "/depth_tiff") ||
      !ExistsDir(this->options_.data_path + "/color_tiff") ||
      !ExistsDir(this->options_.data_path + "/semantic_tiff")) {
    std::cerr << "The provided `data_folder` does not contain a "
                 "`depth_tiff`, a `color_tiff` and a `semantic_tiff` folder."
              << std::endl;
    throw std::runtime_error("ERROR: `datafolder` is not valid.");
  }
  std::cout << "  - Data folder containg the .tiff images: "
            << this->options_.data_path << std::endl;

  // Visualization folder (if provided)
  if (this->options_.visualization_path.empty()) {
    std::cout << "  - Output folder containg visualization: "
              << "(not provided, so no visualization)" << std::endl;
  } else {
    std::cout << "  - Output folder containg visualization: "
              << this->options_.visualization_path << std::endl;
  }

  // Print parameters information
  std::cout << "  - Depth error threshold: "
            << this->options_.depth_error_threshold << std::endl;
  std::cout << "  - Numeric relative step size: "
            << this->options_.numeric_relative_step_size << std::endl;
  std::cout << "  - Error computation pixel step: "
            << this->options_.error_computation_pixel_step << std::endl;

  // Print reconstruction information
  std::cout << "  - SemanticBundleAdjustmentConfig:" << std::endl;

  // Print camera model information
  for (const image_t image_id : this->config_.Images()) {
    Image& image = reconstruction->Image(image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());
    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    std::cout << "    - CameraModel: " << typeid(CameraModel).name()   \
              << " with " << CameraModel::kNumParams << " parameters"; \
    break;  // important no \ here !
      CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
    }

    // Print example of camera parameters
    const std::vector<double>& params = camera.Params();
    if (!params.empty()) {
      std::cout << ", e.g. [";

      auto it = params.begin();
      while (true) {
        std::cout << *it;
        if (++it != params.end()) {
          std::cout << ", ";
        } else {
          break;
        }
      }
      std::cout << "]" << std::endl;
    } else {
      std::cout << std::endl;
    }
    break;
  }

  // Images
  std::cout << "    - NumImages: " << this->config_.NumImages() << " (";
  int c = 0;
  for (const image_t image_id : this->config_.Images()) {
    if (c++ != 0) std::cout << ", ";
    std::cout << reconstruction->Image(image_id).Name();
  }
  std::cout << ")" << std::endl;

  // Constant cameras
  std::cout << "    - NumConstantCameras: "
            << this->config_.NumConstantCameras();
  if (this->config_.NumConstantCameras() > 0) {
    std::cout << " (corresponding images: ";
    c = 0;
    for (const image_t image_id : this->config_.Images()) {
      camera_t camera_id = reconstruction->Image(image_id).CameraId();

      if (this->config_.IsConstantCamera(image_id)) {
        if (c++ != 0) std::cout << ", ";
        std::cout << reconstruction->Image(image_id).Name();
      }
    }
    std::cout << ")";
  }
  std::cout << std::endl;

  // Constant poses
  if (this->options_.refine_extrinsics) {
    std::cout << "    - NumConstantPoses: " << this->config_.NumConstantPoses();
  } else {
    std::cout << "    - NumConstantPoses: " << this->config_.Images().size()
              << " (all since NOT refining extrinsics)";
  }
  if (this->config_.NumConstantPoses() > 0 ||
      !this->options_.refine_extrinsics) {
    std::cout << " (";
    c = 0;
    for (const image_t image_id : this->config_.Images()) {
      const bool constant_pose = !this->options_.refine_extrinsics ||
                                 this->config_.HasConstantPose(image_id);
      if (constant_pose) {
        if (c++ != 0) std::cout << ", ";
        std::cout << reconstruction->Image(image_id).Name();
      }
    }
    std::cout << ")";
  }
  std::cout << std::endl;

  // Constant translation vectors
  std::cout << "    - NumConstantTvecs: " << this->config_.NumConstantTvecs();
  if (this->config_.NumConstantTvecs() > 0) {
    std::cout << " (";
    c = 0;
    for (const image_t image_id : this->config_.Images()) {
      if (this->config_.HasConstantTvec(image_id)) {
        if (c++ != 0) std::cout << ", ";
        std::cout << reconstruction->Image(image_id).Name();
        if (c > 10) {
          std::cout << ", ...";
          break;
        }
      }
    }
    std::cout << ")";
  }
  std::cout << std::endl << std::endl;
}

bool SemanticBundleAdjuster::Solve(Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

  PrintSetupInformation(reconstruction);

  problem_ = std::make_unique<ceres::Problem>();

  ceres::LossFunction* loss_function = options_.CreateLossFunction();
  SetUp(reconstruction, loss_function);

  // Evaluate initial cost
  double cost;
  this->problem_->Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr,
                           nullptr, nullptr);

  const SemanticBundleAdjustmentOptions::LossFunctionType& type =
      this->options_.loss_function_type;
  std::string loss_function_string = "UNKNOWN";

  switch (type) {
    case SemanticBundleAdjustmentOptions::LossFunctionType::TRIVIAL:
      loss_function_string = "TRIVIAL (i.e. 0.5 err^2)";
      break;
    case SemanticBundleAdjustmentOptions::LossFunctionType::SOFT_L1:
      loss_function_string = "SOFT_L1 (i.e. 0.5 soft_l1(err^2))";
      break;
    case SemanticBundleAdjustmentOptions::LossFunctionType::CAUCHY:
      loss_function_string = "CAUCHY (i.e. 0.5 cauchy(err^2))";
      break;
    default:
      break;
  }

  std::cout << "\nFinished setting up the problem, which has: \n"
            << "  - " << this->problem_->NumResiduals()
            << " residuals (i.e. number of pixels for which the semantic error "
               "is computed)\n"
            << "  - " << this->problem_->NumResidualBlocks()
            << " residual blocks\n"
            << "  - " << this->problem_->NumParameters() << " parameters\n"
            << "  - " << this->problem_->NumParameterBlocks()
            << " parameter blocks\n"
            << "  - Loss function: " << loss_function_string << "\n"
            << "  - Initial cost of: " << cost
            << " (semantic error: " << cost * 2 << ")\n\n";

  if (problem_->NumResiduals() == 0) {
    return false;
  }

  ceres::Solver::Options solver_options = options_.solver_options;
  const bool has_sparse =
      solver_options.sparse_linear_algebra_library_type != ceres::NO_SPARSE;

  // Create a functor with captured parameters
  SBACallbackFunctor sbaCallbackFunctor(reconstruction, this);
  solver_options.callbacks.push_back(&sbaCallbackFunctor);

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 1000;
  const size_t num_images = config_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver && has_sparse) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  if (problem_->NumResiduals() <
      options_.min_num_residuals_for_multi_threading) {
    solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR
  } else {
    solver_options.num_threads =
        GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads =
        GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  }

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

  ceres::Solve(solver_options, problem_.get(), &summary_);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options_.print_summary) {
    PrintHeading2("Semantic Bundle Adjustment Report");
    PrintSemanticSolverSummary(summary_);
  }

  // Create output text new folder
  std::string output_text = this->options_.output_path + "/text";
  createFolders(output_text);

  // Write save reconstruction
  reconstruction->Write(this->options_.output_path);
  reconstruction->WriteText(output_text);
  std::cout << "Successfully saved refined reconstruction to: '"
            << this->options_.output_path << "'." << std::endl;

  TearDown(reconstruction);

  return true;
}

void PrintSemanticSolverSummary(const ceres::Solver::Summary& summary) {
  std::cout << std::right << std::setw(16) << "Residuals : ";
  std::cout << std::left << summary.num_residuals_reduced << std::endl;

  std::cout << std::right << std::setw(16) << "Parameters : ";
  std::cout << std::left << summary.num_effective_parameters_reduced
            << std::endl;

  std::cout << std::right << std::setw(16) << "Iterations : ";
  std::cout << std::left
            << summary.num_successful_steps + summary.num_unsuccessful_steps
            << std::endl;

  std::cout << std::right << std::setw(16) << "Time : ";
  std::cout << std::left << summary.total_time_in_seconds << " [s]"
            << std::endl;

  std::cout << std::right << std::setw(16) << "Initial cost : ";
  std::cout << std::right << std::setprecision(6) << summary.initial_cost << " "
            << std::endl;

  std::cout << std::right << std::setw(16) << "Final cost : ";
  std::cout << std::right << std::setprecision(6) << summary.final_cost << " "
            << std::endl;

  std::cout << std::right << std::setw(16) << "Termination : ";

  std::string termination = "";

  switch (summary.termination_type) {
    case ceres::CONVERGENCE:
      termination = "Convergence";
      break;
    case ceres::NO_CONVERGENCE:
      termination = "No convergence";
      break;
    case ceres::FAILURE:
      termination = "Failure";
      break;
    case ceres::USER_SUCCESS:
      termination = "User success";
      break;
    case ceres::USER_FAILURE:
      termination = "User failure";
      break;
    default:
      termination = "Unknown";
      break;
  }

  std::cout << std::right << termination << std::endl;
  std::cout << std::endl;
}

const ceres::Solver::Summary& SemanticBundleAdjuster::Summary() const {
  return summary_;
}

void SemanticBundleAdjuster::Assert(
    const Reconstruction* const reconstruction) const {
  // Assert that the cameras all have constant intrinsics
  for (const image_t& image_id : config_.Images()) {
    const Image& image = reconstruction->Image(image_id);
    if (!config_.IsConstantCamera(image.CameraId())) {
      std::cerr << "ERROR: camera intrinsics of image '" << image.Name()
                << "' are not set to constant. This is not supported."
                << std::endl;
      throw std::runtime_error(
          "ERROR: camera intrinsics of image '" + image.Name() +
          "' are not set to constant. This is not supported.");
    }
  }

  // Assert that all the camera models are SimplePinholeCameraModel
  for (const image_t& image_id : config_.Images()) {
    const Image& image = reconstruction->Image(image_id);
    const Camera& camera = reconstruction->Camera(image.CameraId());

    if (camera.ModelId() != SimplePinholeCameraModel::kModelId) {
      std::cerr << "ERROR: the only supported camera model is "
                   "SimplePinholeCameraModel."
                << std::endl;
      throw std::runtime_error(
          "ERROR: the only supported camera model is "
          "SimplePinholeCameraModel.");
    }
  }

  // Assert that refine_extrinsics is set to true
  if (!this->options_.refine_extrinsics) {
    std::cerr << "ERROR: the argument 'refine_extrinsics' must be set to true."
              << std::endl;
    throw std::runtime_error(
        "ERROR: the argument 'refine_extrinsics' must be set to true.");
  }

  std::cout << "All cameras have constant intrinsics and are of type "
               "'SimplePinholeCameraModel'. \n\n";
}

void SemanticBundleAdjuster::SetUp(Reconstruction* reconstruction,
                                   ceres::LossFunction* loss_function) {
  // Assert that the cameras all have constant intrinsics and are of type
  // SimplePinholeCameraModel
  Assert(reconstruction);

  // Read depth maps and semantic maps
  ReadDepthAndSemanticMaps(reconstruction);

  // Build the semantic error cost function
  for (const image_t image_id_1 : config_.Images()) {
    for (const image_t image_id_2 : config_.Images()) {
      AddImagePairToProblem(image_id_1, image_id_2, reconstruction,
                            loss_function);
    }
  }

  // Set up the manifolds of the camera extrinsics
  SetUpManifolds(reconstruction);

  // Set up output folders
  SetUpOutputFolders();
}

void SemanticBundleAdjuster::SetUpManifolds(Reconstruction* reconstruction) {

  for (const image_t image_id : this->config_.Images()) {
    Image& image = reconstruction->Image(image_id);

    this->camera_ids_.insert(image.CameraId());  // Is a set so no dupplicates

    double* qvec_data = image.Qvec().data();
    double* tvec_data = image.Tvec().data();

    const bool constant_pose = !this->options_.refine_extrinsics ||
                               this->config_.HasConstantPose(image_id);
    // Set pose parameterization.
    if (!constant_pose) {
      SetQuaternionManifold(this->problem_.get(), qvec_data);
      if (config_.HasConstantTvec(image_id)) {
        const std::vector<int>& constant_tvec_idxs =
            this->config_.ConstantTvec(image_id);
        SetSubsetManifold(3, constant_tvec_idxs, this->problem_.get(),
                          tvec_data);
      }
    }
  }
}

void SemanticBundleAdjuster::TearDown(Reconstruction*) {
  // Nothing to do
}

void SemanticBundleAdjuster::AddImagePairToProblem(
    const image_t image_id_1, const image_t image_id_2,
    Reconstruction* reconstruction, ceres::LossFunction* loss_function) {
  // Skip if same image id
  if (image_id_1 == image_id_2) return;

  int residuals_counter = 0;
  int skipped_residuals_counter = 0;

  int correspondences_counter = 0;  // all
  int skipped_correspondences_counter =
      0;  // the ones for which the depth is 0 in the first image
  int added_correspondences_counter = 0;  // the added ones

  // Get the two images and cameras
  Image& image_1 = reconstruction->Image(image_id_1);
  Image& image_2 = reconstruction->Image(image_id_2);
  Camera& camera_1 = reconstruction->Camera(image_1.CameraId());
  Camera& camera_2 = reconstruction->Camera(image_2.CameraId());

  // Get the image names
  const std::string image_1_name = image_1.Name();
  const std::string image_2_name = image_2.Name();

  // Access the depth maps
  const Eigen::MatrixXf* const depth_1 = &this->depth_maps_[image_1_name];
  const Eigen::MatrixXf* const depth_2 = &this->depth_maps_[image_2_name];

  // Access the semantic maps
  const Eigen::MatrixXf* const semantic_1 = &this->semantic_maps_[image_1_name];
  const Eigen::MatrixXf* const semantic_2 = &this->semantic_maps_[image_2_name];

  // CostFunction assumes unit quaternions.
  image_1.NormalizeQvec();
  image_2.NormalizeQvec();

  // Get the poses q, t
  double* qvec_1_data = image_1.Qvec().data();
  double* qvec_2_data = image_2.Qvec().data();
  double* tvec_1_data = image_1.Tvec().data();
  double* tvec_2_data = image_2.Tvec().data();

  // Camera parameters
  double* camera_1_params_data = camera_1.ParamsData();
  double* camera_2_params_data = camera_2.ParamsData();

  // Booleans for constant pose
  const bool constant_pose_1 =
      !options_.refine_extrinsics || config_.HasConstantPose(image_id_1);
  const bool constant_pose_2 =
      !options_.refine_extrinsics || config_.HasConstantPose(image_id_2);

#define PRINT
#ifdef PRINT

  std::cout << "\nImages: '" << image_1_name << "' ("
            << (constant_pose_1 ? "constant" : "variable") << ")   ==>  '"
            << image_2_name << "' ("
            << (constant_pose_2 ? "constant" : "variable") << "):" << std::endl;

  std::cout << "  - Image 1: '" << image_1_name << "' ("
            << (constant_pose_1 ? "constant" : "variable")
            << ") has:" << std::endl
            << "    - qvec_1(w,x,y,z): [" << qvec_1_data[0] << ", "
            << qvec_1_data[1] << ", " << qvec_1_data[2] << ", "
            << qvec_1_data[3] << "]" << std::endl
            << "    - tvec_1: [" << tvec_1_data[0] << ", " << tvec_1_data[1]
            << ", " << tvec_1_data[2] << "]" << std::endl
            << "    - camera_1_params_data(f, c_x, c_y): ["
            << camera_1_params_data[0] << ", " << camera_1_params_data[1]
            << ", " << camera_1_params_data[2] << "]" << std::endl;

  std::cout << "  - Image 2: '" << image_2_name << "' ("
            << (constant_pose_2 ? "constant" : "variable")
            << ") has:" << std::endl
            << "    - qvec_2(w,x,y,z): [" << qvec_2_data[0] << ", "
            << qvec_2_data[1] << ", " << qvec_2_data[2] << ", "
            << qvec_2_data[3] << "]" << std::endl
            << "    - tvec_2: [" << tvec_2_data[0] << ", " << tvec_2_data[1]
            << ", " << tvec_2_data[2] << "]" << std::endl
            << "    - camera_2_params_data(f, c_x, c_y): ["
            << camera_2_params_data[0] << ", " << camera_2_params_data[1]
            << ", " << camera_2_params_data[2] << "]" << std::endl;
#endif

  if (constant_pose_1 && constant_pose_2) {
    std::cout << "Images '" << image_1_name << "' ==> '" << image_2_name
              << ":\n"
              << "Two camera poses are constant --> Skipping." << std::endl;
    return;
  }

  // Image 1 dimensions
  int height_1 = depth_1->rows();
  int width_1 = depth_1->cols();

  // Loop through the pixels of image 1
  for (int y = 0; y < height_1;
       y = y + this->options_.error_computation_pixel_step) {
    for (int x = 0; x < width_1;
         x = x + this->options_.error_computation_pixel_step) {
      correspondences_counter++;

      // Get 2D point in image 1
      Eigen::Vector2i point2D(x, y);

      // Get its corresponding depth
      const float depth = depth_1->coeff(y, x);

      // Skip pixels with zero depth value
      if (depth < 1e-4) {
        this->skipped_semantic_residual_block_counter++;
        skipped_residuals_counter++;
        skipped_correspondences_counter++;
        continue;
      }

      added_correspondences_counter++;

      // Access depth and semantic maps
      const std::unordered_map<std::string, const Eigen::MatrixXf>* const
          depth_maps_ptr = &this->depth_maps_;

      const std::unordered_map<std::string, const Eigen::MatrixXf>* const
          semantic_maps_ptr = &this->semantic_maps_;

      // Create pointer to the cost function
      ceres::CostFunction* cost_function = nullptr;

      // Constant pose 1 and variable pose 2
      if (constant_pose_1) {
        cost_function =
            ConstantFirstPoseSemanticBACostFunction<SimplePinholeCameraModel>::
                Create(this->options_.numeric_relative_step_size,
                       image_1.Qvec(), image_1.Tvec(), image_1_name,
                       image_2_name, camera_1_params_data, camera_2_params_data,
                       point2D, depth_maps_ptr, semantic_maps_ptr,
                       this->options_.depth_error_threshold);

        // Add the reisual block to the problem
        problem_->AddResidualBlock(cost_function, loss_function, qvec_2_data,
                                   tvec_2_data);
        this->semantic_residual_block_counter++;
        residuals_counter++;

      }
      // Variable pose 1 and constant pose 2
      else if (constant_pose_2) {
        cost_function =
            ConstantSecondPoseSemanticBACostFunction<SimplePinholeCameraModel>::
                Create(this->options_.numeric_relative_step_size,
                       image_2.Qvec(), image_2.Tvec(), image_1_name,
                       image_2_name, camera_1_params_data, camera_2_params_data,
                       point2D, depth_maps_ptr, semantic_maps_ptr,
                       this->options_.depth_error_threshold);

        // Add the reisual block to the problem
        this->problem_->AddResidualBlock(cost_function, loss_function,
                                         qvec_1_data, tvec_1_data);
        this->semantic_residual_block_counter++;
        residuals_counter++;

      }
      // Variable pose 1 and variable pose 2
      else {
        cost_function =
            SemanticBACostFunction<SimplePinholeCameraModel>::Create(
                this->options_.numeric_relative_step_size, image_1_name,
                image_2_name, camera_1_params_data, camera_2_params_data,
                point2D, depth_maps_ptr, semantic_maps_ptr,
                this->options_.depth_error_threshold);

        // Add the reisual block to the problem
        this->problem_->AddResidualBlock(cost_function, loss_function,
                                         qvec_1_data, tvec_1_data, qvec_2_data,
                                         tvec_2_data);
        this->semantic_residual_block_counter++;
        residuals_counter++;
      }

    }  // end for loop
  }    // end for loop

  // Print summary
  std::cout
      << "Current summary:" << std::endl
      << "  - Looped through " << correspondences_counter
      << " pixels (height: " << height_1 << ", width: " << width_1
      << ", step: " << this->options_.error_computation_pixel_step << ")"
      << std::endl
      << "  - " << std::setw(6) << skipped_correspondences_counter
      << " correspondences were skipped due to a zero depth in the first image"
      << std::endl
      << "  - " << std::setw(6) << added_correspondences_counter
      << " correspondences were added" << std::endl
      << "Current total: " << std::setw(6)
      << this->semantic_residual_block_counter << " correspondences added, "
      << std::setw(6) << this->skipped_semantic_residual_block_counter
      << " corresponences skipped." << std::endl;

  // Evaluate initial cost
  double cost;
  this->problem_->Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr,
                           nullptr, nullptr);
  std::cout << "Current total cost is: " << cost
            << " (i.e. semantic error: " << cost * 2 << ")\n"
            << std::endl;
}

void SemanticBundleAdjuster::ExportSemanticErrorToCSV(
    const Reconstruction* const reconstruction,
    const std::string path_prefix) const {
  // Loop through every pair of images
  for (const image_t& image_id_1 : config_.Images()) {
    for (const image_t& image_id_2 : config_.Images()) {
      // Skip if same image id
      if (image_id_1 == image_id_2) continue;

      // Get the two images and cameras
      const Image& image_1 = reconstruction->Image(image_id_1);
      const Image& image_2 = reconstruction->Image(image_id_2);
      const Camera& camera_1 = reconstruction->Camera(image_1.CameraId());
      const Camera& camera_2 = reconstruction->Camera(image_2.CameraId());

      // Get the image names
      const std::string image_1_name = image_1.Name();
      const std::string image_2_name = image_2.Name();

      // Get the poses q, t
      const double* qvec_1_data = image_1.Qvec().data();
      const double* qvec_2_data = image_2.Qvec().data();
      const double* tvec_1_data = image_1.Tvec().data();
      const double* tvec_2_data = image_2.Tvec().data();

      // Camera parameters
      const double* camera_1_params_data = camera_1.ParamsData();
      const double* camera_2_params_data = camera_2.ParamsData();

      // Access depth and semantic maps
      const std::unordered_map<std::string, const Eigen::MatrixXf>* const
          depth_maps_ptr = &this->depth_maps_;
      const std::unordered_map<std::string, const Eigen::MatrixXf>* const
          semantic_maps_ptr = &this->semantic_maps_;

      // CSV file entries
      std::vector<std::tuple<int, float, Eigen::Vector2i, Eigen::Vector2i,
                             Eigen::Vector3d>>
          entries = {};

      // Image 1 dimensions
      const int height_1 = this->depth_maps_.at(image_1_name).rows();
      const int width_1 = this->depth_maps_.at(image_1_name).cols();

      // Loop through the pixels of image 1
      for (int y = 0; y < height_1;
           y = y + this->options_.error_computation_pixel_step) {
        for (int x = 0; x < width_1;
             x = x + this->options_.error_computation_pixel_step) {
          // Get 2D point in image 1
          Eigen::Vector2i point2D(x, y);

          // Define cost function
          BaseSemanticBACostFunction<SimplePinholeCameraModel>* cost_function =
              new BaseSemanticBACostFunction<SimplePinholeCameraModel>(
                  image_1_name, image_2_name, camera_1_params_data,
                  camera_2_params_data, point2D, depth_maps_ptr,
                  semantic_maps_ptr, this->options_.depth_error_threshold);

          // Compute semantic error
          double residuals[1];
          ReprojectionStatus status;
          Eigen::Vector3d point3D_w_world;
          Eigen::Vector2i point_1_vis = point2D;
          Eigen::Vector2i point_2_vis;
          double semantic_error;
          cost_function->compute_semantic_error<double>(
              qvec_1_data, tvec_1_data, qvec_2_data, tvec_2_data,
              camera_1_params_data, camera_2_params_data, &status,
              point3D_w_world, point_2_vis, &semantic_error);

          // Save values to the entries vector
          int status_integer = static_cast<int>(status);
          entries.emplace_back(status_integer, semantic_error, point_1_vis,
                               point_2_vis, point3D_w_world);
        }  // for loop x
      }    // for loop y

      // File path for the CSV file
      std::string filename_csv =
          path_prefix + "_" + image_1_name + "_to_" + image_2_name + ".csv";
      // Open the file for writing
      std::ofstream file(filename_csv);

      if (file.is_open()) {
        // Write header
        file << "Type,SemanticError,X1,Y1,X2,Y2,X3D,Y3D,Z3D\n";

        // Write data to the file
        for (const auto& entry : entries) {
          int type = std::get<0>(entry);
          float semantic_error = std::get<1>(entry);
          Eigen::Vector2i point1 = std::get<2>(entry);
          Eigen::Vector2i point2 = std::get<3>(entry);
          Eigen::Vector3d point3d = std::get<4>(entry);
          file << type << "," << semantic_error << "," << point1.x() << ","
               << point1.y() << "," << point2.x() << "," << point2.y() << ","
               << point3d.x() << "," << point3d.y() << "," << point3d.z()
               << "\n";
        }
        // Close the file
        file.close();
        std::cout << "  - CSV file '" << filename_csv
                  << "' has been created.\n";
      } else {
        std::cerr << "  - Unable to open the file '" << filename_csv
                  << "' for writing.\n";
      }

    }  // for loop image_id_2
  }    // for loop image_id_1
}

bool SemanticBundleAdjuster::ReadDepthAndSemanticMaps(
    const Reconstruction* const reconstruction) {
  std::cout << "Reading depth and semantic maps for "
            << this->config_.Images().size() << " images:" << std::endl;

  // Loop through the images
  for (const image_t image_id : this->config_.Images()) {
    // Get image name
    const Image& image = reconstruction->Image(image_id);
    const std::string image_name = image.Name();

    // Get depth and semantic maps paths
    std::string depth_path =
        this->options_.data_path + "/depth_tiff/" +
        image_name.substr(0, image_name.find_last_of('.')) + "_depth.tiff";
    std::string semantic_path =
        this->options_.data_path + "/semantic_tiff/" +
        image_name.substr(0, image_name.find_last_of('.')) + "_semantic.tiff";

    // Ensure the files exist
    if (!ExistsFile(depth_path)) {
      std::cerr << "ERROR: the depth file '" << depth_path
                << "' does not exist." << std::endl;
      throw std::runtime_error("ERROR: the depth file '" + depth_path +
                               "' does not exist.");
    }
    if (!ExistsFile(semantic_path)) {
      std::cerr << "ERROR: the semantic file '" << semantic_path
                << "' does not exist." << std::endl;
      throw std::runtime_error("ERROR: the semantic file '" + depth_path +
                               "' does not exist.");
    }

    // Get matrices
    std::cout << "  - depth from '" << depth_path << "'" << std::endl;
    Eigen::MatrixXf depth_map = matrixFromTiff(depth_path);

    std::cout << "  - semantic from " << semantic_path << "'" << std::endl;
    Eigen::MatrixXf semantic_map = matrixFromTiff(semantic_path);

    // Save them to the BundleAdjuster
    this->depth_maps_.insert({image_name, depth_map});
    this->semantic_maps_.insert({image_name, semantic_map});
  }
  std::cout << std::endl;

  return true;
}

void SemanticBundleAdjuster::SetUpOutputFolders() {
  std::string out = this->options_.visualization_path;
  this->optim_steps_folder_ = out + "/optim_steps";

  std::cout << "\nOutput folders: " << std::endl;
  std::cout << "  - Optimization steps folder: " << this->optim_steps_folder_
            << std::endl;

  // Empty output folder
  createFolders(this->options_.output_path, out, this->optim_steps_folder_);
}

////////////////////////////////////////////////////////////////////////////////
// SBACallbackFunctor
////////////////////////////////////////////////////////////////////////////////

SBACallbackFunctor::SBACallbackFunctor(Reconstruction* reconstruction,
                                       SemanticBundleAdjuster* sba)
    : callback_reconstruction_(reconstruction), callback_sba_(sba) {}

ceres::CallbackReturnType SBACallbackFunctor::operator()(
    const ceres::IterationSummary& summary) {
  // Print optimization step
  PrintHeading2("Optimization Iteration " + std::to_string(summary.iteration) +
                " Update");
  // Print optimization informations
  std::cout << std::left << std::setw(16) << "Cost: ";
  std::cout << std::left << std::setprecision(6) << summary.cost << std::endl;
  std::cout << std::left << std::setw(16) << "Cost change: ";
  std::cout << std::left << summary.cost_change << std::endl;
  std::cout << std::left << std::setw(16) << "Gradient norm: ";
  std::cout << std::left << summary.gradient_max_norm << std::setprecision(3)
            << std::endl;

  // Visualization
  // Create new folder
  std::string folder_step = this->callback_sba_->optim_steps_folder_ +
                            "/step_" + std::to_string(summary.iteration);
  std::string folder_step_text = folder_step + "/text";
  createFolders(folder_step, folder_step_text);

  // Export CSV visualization
  if (this->callback_sba_->options_.export_csv) {
    this->callback_sba_->ExportSemanticErrorToCSV(
        this->callback_reconstruction_, folder_step + "/vis");
  }

  // Export camera poses
  callback_reconstruction_->Write(folder_step);
  callback_reconstruction_->WriteText(folder_step_text);
  std::cout << std::endl;

  return ceres::SOLVER_CONTINUE;
}

}  // namespace colmap
