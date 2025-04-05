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

#include "optim/geometric_semantic_bundle_adjustment.h"

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
#include "util/xywh.h"

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// GeometricSemanticBundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

ceres::LossFunction*
GeometricSemanticBundleAdjustmentOptions::CreateLossFunction() const {
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

bool GeometricSemanticBundleAdjustmentOptions::Check() const {
  CHECK_OPTION_GE(loss_function_scale, 0);
  GetCylinderParametrization();
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// GeometricSemanticBundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////

GeometricSemanticBundleAdjustmentConfig::
    GeometricSemanticBundleAdjustmentConfig() {}

size_t GeometricSemanticBundleAdjustmentConfig::NumImages() const {
  return image_ids_.size();
}

size_t GeometricSemanticBundleAdjustmentConfig::NumPoints() const {
  return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t GeometricSemanticBundleAdjustmentConfig::NumConstantCameras() const {
  return constant_camera_ids_.size();
}

size_t GeometricSemanticBundleAdjustmentConfig::NumConstantPoses() const {
  return constant_poses_.size();
}

size_t GeometricSemanticBundleAdjustmentConfig::NumConstantTvecs() const {
  return constant_tvecs_.size();
}

size_t GeometricSemanticBundleAdjustmentConfig::NumVariablePoints() const {
  return variable_point3D_ids_.size();
}

size_t GeometricSemanticBundleAdjustmentConfig::NumConstantPoints() const {
  return constant_point3D_ids_.size();
}

size_t GeometricSemanticBundleAdjustmentConfig::NumResiduals(
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

void GeometricSemanticBundleAdjustmentConfig::AddImage(const image_t image_id) {
  image_ids_.insert(image_id);
}

bool GeometricSemanticBundleAdjustmentConfig::HasImage(
    const image_t image_id) const {
  return image_ids_.find(image_id) != image_ids_.end();
}

void GeometricSemanticBundleAdjustmentConfig::RemoveImage(
    const image_t image_id) {
  image_ids_.erase(image_id);
}

void GeometricSemanticBundleAdjustmentConfig::SetConstantCamera(
    const camera_t camera_id) {
  constant_camera_ids_.insert(camera_id);
}

void GeometricSemanticBundleAdjustmentConfig::SetVariableCamera(
    const camera_t camera_id) {
  constant_camera_ids_.erase(camera_id);
}

bool GeometricSemanticBundleAdjustmentConfig::IsConstantCamera(
    const camera_t camera_id) const {
  return constant_camera_ids_.find(camera_id) != constant_camera_ids_.end();
}

void GeometricSemanticBundleAdjustmentConfig::SetConstantPose(
    const image_t image_id) {
  CHECK(HasImage(image_id));
  CHECK(!HasConstantTvec(image_id));
  constant_poses_.insert(image_id);
}

void GeometricSemanticBundleAdjustmentConfig::SetVariablePose(
    const image_t image_id) {
  constant_poses_.erase(image_id);
}

bool GeometricSemanticBundleAdjustmentConfig::HasConstantPose(
    const image_t image_id) const {
  return constant_poses_.find(image_id) != constant_poses_.end();
}

void GeometricSemanticBundleAdjustmentConfig::SetConstantTvec(
    const image_t image_id, const std::vector<int>& idxs) {
  CHECK_GT(idxs.size(), 0);
  CHECK_LE(idxs.size(), 3);
  CHECK(HasImage(image_id));
  CHECK(!HasConstantPose(image_id));
  CHECK(!VectorContainsDuplicateValues(idxs))
      << "Tvec indices must not contain duplicates";
  constant_tvecs_.emplace(image_id, idxs);
}

void GeometricSemanticBundleAdjustmentConfig::RemoveConstantTvec(
    const image_t image_id) {
  constant_tvecs_.erase(image_id);
}

bool GeometricSemanticBundleAdjustmentConfig::HasConstantTvec(
    const image_t image_id) const {
  return constant_tvecs_.find(image_id) != constant_tvecs_.end();
}

const std::unordered_set<image_t>&
GeometricSemanticBundleAdjustmentConfig::Images() const {
  return image_ids_;
}

const std::unordered_set<point3D_t>&
GeometricSemanticBundleAdjustmentConfig::VariablePoints() const {
  return variable_point3D_ids_;
}

const std::unordered_set<point3D_t>&
GeometricSemanticBundleAdjustmentConfig::ConstantPoints() const {
  return constant_point3D_ids_;
}

const std::vector<int>& GeometricSemanticBundleAdjustmentConfig::ConstantTvec(
    const image_t image_id) const {
  return constant_tvecs_.at(image_id);
}

void GeometricSemanticBundleAdjustmentConfig::AddVariablePoint(
    const point3D_t point3D_id) {
  CHECK(!HasConstantPoint(point3D_id));
  variable_point3D_ids_.insert(point3D_id);
}

void GeometricSemanticBundleAdjustmentConfig::AddConstantPoint(
    const point3D_t point3D_id) {
  CHECK(!HasVariablePoint(point3D_id));
  constant_point3D_ids_.insert(point3D_id);
}

bool GeometricSemanticBundleAdjustmentConfig::HasPoint(
    const point3D_t point3D_id) const {
  return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
}

bool GeometricSemanticBundleAdjustmentConfig::HasVariablePoint(
    const point3D_t point3D_id) const {
  return variable_point3D_ids_.find(point3D_id) != variable_point3D_ids_.end();
}

bool GeometricSemanticBundleAdjustmentConfig::HasConstantPoint(
    const point3D_t point3D_id) const {
  return constant_point3D_ids_.find(point3D_id) != constant_point3D_ids_.end();
}

void GeometricSemanticBundleAdjustmentConfig::RemoveVariablePoint(
    const point3D_t point3D_id) {
  variable_point3D_ids_.erase(point3D_id);
}

void GeometricSemanticBundleAdjustmentConfig::RemoveConstantPoint(
    const point3D_t point3D_id) {
  constant_point3D_ids_.erase(point3D_id);
}

////////////////////////////////////////////////////////////////////////////////
// GeometricSemanticBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

// Explicit instantiation
template class GeometricSemanticBundleAdjuster<Cylinder>;
template class GeometricSemanticBundleAdjuster<CylinderBy2Points>;

template <typename CylinderClass>
GeometricSemanticBundleAdjuster<CylinderClass>::GeometricSemanticBundleAdjuster(
    const GeometricSemanticBundleAdjustmentOptions& options,
    const GeometricSemanticBundleAdjustmentConfig& config)
    : options_(options), config_(config) {
  CHECK(options_.Check());
}

template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::PrintSetupInformation(
    Reconstruction* reconstruction) {
  // Print information
  std::cout << "Setup the Geometric Semantic BA problem, with:" << std::endl;

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

  // Geometry path
  if (!ExistsFile(this->options_.input_geometry)) {
    std::cerr << "ERROR: the file '" << this->options_.input_geometry
              << "' containing the cylinders descriptions does not exist."
              << std::endl;
    throw std::runtime_error(
        "ERROR: the file '" + this->options_.input_geometry +
        +"' containing the cylinders descriptions does not exist.");
  }
  std::cout << "  - Input initial geometry description text file: "
            << this->options_.input_geometry << std::endl;

  // Output visualization folder (is set up later)
  std::cout << "  - Output visualization folder: "
            << this->options_.visualization_path << std::endl;

  // Print parameters information
  std::cout << "  - Trunk semantic class: "
            << this->options_.trunk_semantic_class << std::endl;

  std::cout << "  - Numeric relative step size: "
            << this->options_.numeric_relative_step_size << std::endl;

  std::cout << "  - Refine extrinsics: " << std::boolalpha
            << this->options_.refine_extrinsics << std::noboolalpha
            << std::endl;

  std::cout << "  - Refine geometry: " << std::boolalpha
            << this->options_.refine_geometry << std::noboolalpha << std::endl;
  std::cout << "  - Cylinder parametrization: ";
  switch (this->options_.GetCylinderParametrization()) {
    case CylinderParametrization::DEFAULT:
      std::cout << "'default' (i.e. orientation, position, radius and height)";
      break;
    case CylinderParametrization::BY2POINTS:
      std::cout
          << "'by_2_points' (i.e. top position, bottom position and radius)";
      break;
    default:
      std::cerr << std::endl
                << "ERROR: CylinderParametrization not implemented."
                << std::endl;
      throw std::runtime_error(
          "ERROR: CylinderParametrization not implemented.");
      break;
  }
  std::cout << std::endl;

  std::cout << "  - Include landmark error: " << std::boolalpha
            << this->options_.include_landmark_error << std::noboolalpha
            << std::endl;
  std::cout << "  - Landmark error weight: "
            << this->options_.landmark_error_weight << std::endl;

  // Print reconstruction information
  std::cout << "  - GeometricSemanticBundleAdjustmentConfig:" << std::endl;

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

template <typename CylinderClass>
bool GeometricSemanticBundleAdjuster<CylinderClass>::Solve(
    Reconstruction* reconstruction) {
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

  const GeometricSemanticBundleAdjustmentOptions::LossFunctionType& type =
      this->options_.loss_function_type;
  std::string loss_function_string = "UNKNOWN";

  switch (type) {
    case GeometricSemanticBundleAdjustmentOptions::LossFunctionType::TRIVIAL:
      loss_function_string = "TRIVIAL (i.e. 0.5 err^2)";
      break;
    case GeometricSemanticBundleAdjustmentOptions::LossFunctionType::SOFT_L1:
      loss_function_string = "SOFT_L1 (i.e. 0.5 soft_l1(err^2))";
      break;
    case GeometricSemanticBundleAdjustmentOptions::LossFunctionType::CAUCHY:
      loss_function_string = "CAUCHY (i.e. 0.5 cauchy(err^2))";
      break;
    default:
      break;
  }

  std::cout << "\nFinished setting up the problem, which has: \n"
            << "  - " << this->problem_->NumResiduals() << " residuals\n"
            << "  - " << this->problem_->NumResidualBlocks()
            << " residual blocks\n"
            << "  - " << this->problem_->NumParameters() << " parameters\n"
            << "  - " << this->problem_->NumParameterBlocks()
            << " parameter blocks\n"
            << "  - Loss function: " << loss_function_string << "\n"
            << "  - Initial cost of: " << cost << "\n\n";

  if (problem_->NumResiduals() == 0) {
    return false;
  }

  ceres::Solver::Options solver_options = options_.solver_options;
  const bool has_sparse =
      solver_options.sparse_linear_algebra_library_type != ceres::NO_SPARSE;

  // Create a functor with captured parameters
  MyClass<CylinderClass> gsbaCallbackFunctor(reconstruction, this);
  //  GSBACallbackFunctor<CylinderClass> gsbaCallbackFunctor(reconstruction,
  //  this);
  solver_options.callbacks.push_back(&gsbaCallbackFunctor);

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

  // Print summary
  if (options_.print_summary) {
    PrintHeading2("Geometric Semantic Bundle Adjustment Report");
    PrintGeometricSemanticSolverSummary(summary_);
    this->PrintCylinders(reconstruction);
  }

  // Export refined cylinders
  exportCylindersToText(this->cylinders_,
                        this->options_.output_path + "/cylinders.txt");
  std::cout << "Successfully saved refined cylinders to: '"
            << this->options_.output_path + "/cylinders.txt"
            << "'." << std::endl;

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

void PrintGeometricSemanticSolverSummary(
    const ceres::Solver::Summary& summary) {
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

template <typename CylinderClass>
const ceres::Solver::Summary&
GeometricSemanticBundleAdjuster<CylinderClass>::Summary() const {
  return summary_;
}
template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::Assert(
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

  // Assert that the loss function is LossFunctionType::TRIVIAL;
  if (this->options_.loss_function_type !=
      GeometricSemanticBundleAdjustmentOptions::LossFunctionType::TRIVIAL) {
    std::cerr << "ERROR: the only supported loss function is "
                 "'LossFunctionType::TRIVIAL'."
              << std::endl;
    throw std::runtime_error(
        "ERROR: the only supported loss function is "
        "'LossFunctionType::TRIVIAL'.");
  }

  std::cout << "All cameras have constant intrinsics and are of type "
               "'SimplePinholeCameraModel'. The loss function is 'TRIVIAL'.\n"
            << std::endl;
}

template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::SetUpGeometryError(
    Reconstruction* reconstruction) {
  std::cout << "Adding residuals blocks for the "
            << (options_.refine_geometry ? "constant " : "")
            << "geometry:" << std::endl;

  // Weight of the residual blocks
  ceres::LossFunction* cylinder_loss_function = new ceres::ScaledLoss(
      nullptr, 1. / config_.Images().size(), ceres::Ownership::TAKE_OWNERSHIP);
  for (const image_t image_id : config_.Images()) {
    AddImageToProblem(image_id, reconstruction, cylinder_loss_function);
  }
}

template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::SetUpLandmarkError(
    Reconstruction* reconstruction) {
  std::cout << "\nAdding residuals blocks for the landmarks since "
               "'include_landmark_error' is set to true:"
            << std::endl;

  int num_residuals_blocks = this->problem_->NumResidualBlocks();

  // Count total number of 2D features
  int total_num_2d_features = 0;
  for (const image_t image_id : config_.Images()) {
    Image& image = reconstruction->Image(image_id);
    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        total_num_2d_features++;
      }
    }
  }

  for (const image_t image_id : config_.Images()) {
    // Count number of 2D features in the image
    int num_2d_features = 0;
    Image& image = reconstruction->Image(image_id);
    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        num_2d_features++;
      }
    }

    double weight = options_.landmark_error_weight / total_num_2d_features;
    std::cout << "Image: '" << image.Name() << "' weight=" << weight
              << ", num_features=" << num_2d_features << std::endl;
    ceres::LossFunction* landmark_loss_function = new ceres::ScaledLoss(
        nullptr, weight, ceres::Ownership::TAKE_OWNERSHIP);

    AddImageToProblemForLandmarks(image_id, reconstruction,
                                  landmark_loss_function);
  }
  std::cout << "  - Added "
            << this->problem_->NumResidualBlocks() - num_residuals_blocks
            << " resdiuals blocks." << std::endl;

  // Add point to problem: When are these being used?
  num_residuals_blocks = this->problem_->NumResidualBlocks();
  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblemUsualForLandmarks(point3D_id, reconstruction,
                                       options_.CreateLossFunction());
  }
  for (const auto point3D_id : config_.ConstantPoints()) {
    AddPointToProblemUsualForLandmarks(point3D_id, reconstruction,
                                       options_.CreateLossFunction());
  }

  if (num_residuals_blocks != this->problem_->NumResidualBlocks()) {
    std::cerr << "ERROR: investigate when the COLMAP AddPointToProblem are "
                 "actually adding residuals blocks!"
              << std::endl;

    throw std::runtime_error(
        "ERROR: investigate when the COLMAP AddPointToProblem are "
        "actually adding residuals blocks!");
  }

  // Parametrize the points
  ParameterizePointsForLandmarks(reconstruction);
}

// Assert that the cameras all have constant intrinsics and are of type
template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::SetUp(
    Reconstruction* reconstruction, ceres::LossFunction* loss_function) {
  // Assert that the cameras all have constant intrinsics and are of type
  // SimplePinholeCameraModel
  Assert(reconstruction);

  // Read and register the cylinders
  pushBackCylindersReadFromText(this->options_.input_geometry,
                                this->cylinders_);

  // Read depth maps and semantic maps
  ReadDepthAndSemanticMaps(reconstruction);

  // Print initial cylinders
  PrintCylinders(reconstruction);

  // Build the residuals for the geometry
  SetUpGeometryError(reconstruction);

  // -- Usual BA: add the landmark reconstruction error to the cost function
  // Warning: AddPointsToProblem assumes that AddImageToProblem is called first.
  // Do not change order of instructions!
  if (options_.include_landmark_error) {
    SetUpLandmarkError(reconstruction);
  }

  // Set up the manifolds of the cylinders
  SetUpCylinderManifolds();

  // Set up the manifolds of the camera extrinsics
  SetUpManifolds(reconstruction);

  // Set up output folders
  SetUpOutputFolders();
}

template <>
void GeometricSemanticBundleAdjuster<Cylinder>::AddImageToProblem(
    const image_t image_id, Reconstruction* reconstruction,
    ceres::LossFunction* loss_function) {
  // Get image and camera
  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  // Get the image name
  const std::string image_name = image.Name();

  // CostFunction assumes unit quaternions.
  image.NormalizeQvec();

  double* camera_qvec_data = image.Qvec().data();
  double* camera_tvec_data = image.Tvec().data();
  double* camera_params_data = camera.ParamsData();

  const bool constant_pose =
      !options_.refine_extrinsics || config_.HasConstantPose(image_id);

  if (constant_pose && !this->options_.refine_geometry) {
    std::cout << "  - No residual block for image '" << image_name
              << "' since constant camera pose and constant geometry."
              << std::endl;
    return;
  }

  for (Cylinder& cylinder : this->cylinders_) {
    std::cout << "  - Residual block for image '" << image_name << "' ";

    // Constant camera but variable geometry
    if (constant_pose) {
      // Create pointer to the cost function
      ceres::CostFunction* cost_function = nullptr;
      cost_function = ConstantPoseGSBACostFunction::Create(
          this->options_.numeric_relative_step_size, camera_qvec_data,
          camera_tvec_data, camera_params_data,
          &this->semantic_maps_bool_[image_name]);

      // Add the residual block to the problem
      problem_->AddResidualBlock(cost_function, loss_function,
                                 cylinder.Qvec().data(), cylinder.Tvec().data(),
                                 &cylinder.RadiusData(),
                                 &cylinder.HeightData());

      std::cout << "with constant camera pose." << std::endl;

    }
    // Variable camera and geometry
    else if (options_.refine_geometry) {
      // Create pointer to the cost function
      ceres::CostFunction* cost_function = nullptr;
      cost_function = GSBACostFunction::Create(
          this->options_.numeric_relative_step_size, camera_params_data,
          &this->semantic_maps_bool_[image_name]);

      // Add the residual block to the problem
      problem_->AddResidualBlock(cost_function, loss_function, camera_qvec_data,
                                 camera_tvec_data, cylinder.Qvec().data(),
                                 cylinder.Tvec().data(), &cylinder.RadiusData(),
                                 &cylinder.HeightData());

      std::cout << "." << std::endl;
    }
    // Constant geometry but variable camera
    else {
      // Create pointer to the cost function
      ceres::CostFunction* cost_function = nullptr;
      cost_function = ConstantCylinderGSBACostFunction<Cylinder>::Create(
          this->options_.numeric_relative_step_size, cylinder,
          camera_params_data, &this->semantic_maps_bool_[image_name]);

      // Add the reisual block to the problem
      problem_->AddResidualBlock(cost_function, loss_function, camera_qvec_data,
                                 camera_tvec_data);

      std::cout << "with constant cylinder." << std::endl;
    }
  }
}

template <>
void GeometricSemanticBundleAdjuster<CylinderBy2Points>::AddImageToProblem(
    const image_t image_id, Reconstruction* reconstruction,
    ceres::LossFunction* loss_function) {
  // Get image and camera
  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  // Get the image name
  const std::string image_name = image.Name();

  // CostFunction assumes unit quaternions.
  image.NormalizeQvec();

  double* camera_qvec_data = image.Qvec().data();
  double* camera_tvec_data = image.Tvec().data();
  double* camera_params_data = camera.ParamsData();

  const bool constant_pose =
      !options_.refine_extrinsics || config_.HasConstantPose(image_id);

  if (constant_pose && !this->options_.refine_geometry) {
    std::cout << "  - No residual block for image '" << image_name
              << "' since constant camera pose and constant geometry."
              << std::endl;
    return;
  }

  for (CylinderBy2Points& cylinder : this->cylinders_) {
    std::cout << "  - Residual block for image '" << image_name << "' ";

    // Constant camera but variable geometry
    if (constant_pose) {
      // Create pointer to the cost function
      ceres::CostFunction* cost_function = nullptr;
      cost_function = ConstantPoseGSBACostFunctionBy2Points::Create(
          this->options_.numeric_relative_step_size, camera_qvec_data,
          camera_tvec_data, camera_params_data,
          &this->semantic_maps_bool_[image_name]);

      // Add the residual block to the problem
      problem_->AddResidualBlock(
          cost_function, loss_function, cylinder.Tvec1().data(),
          cylinder.Tvec2().data(), &cylinder.RadiusData());

      std::cout << "with constant camera pose." << std::endl;

    }
    // Variable camera and geometry
    else if (options_.refine_geometry) {
      // Create pointer to the cost function
      ceres::CostFunction* cost_function = nullptr;
      cost_function = GSBACostFunctionBy2Points::Create(
          this->options_.numeric_relative_step_size, camera_params_data,
          &this->semantic_maps_bool_[image_name]);

      // Add the residual block to the problem
      problem_->AddResidualBlock(cost_function, loss_function, camera_qvec_data,
                                 camera_tvec_data, cylinder.Tvec1().data(),
                                 cylinder.Tvec2().data(),
                                 &cylinder.RadiusData());

      std::cout << "." << std::endl;
    }
    // Constant geometry but variable camera
    else {
      // Create pointer to the cost function
      ceres::CostFunction* cost_function = nullptr;
      cost_function =
          ConstantCylinderGSBACostFunction<CylinderBy2Points>::Create(
              this->options_.numeric_relative_step_size, cylinder,
              camera_params_data, &this->semantic_maps_bool_[image_name]);

      // Add the reisual block to the problem
      problem_->AddResidualBlock(cost_function, loss_function, camera_qvec_data,
                                 camera_tvec_data);

      std::cout << "with constant cylinder." << std::endl;
    }
  }
}

template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::ExportCylindersToCSV(
    Reconstruction* reconstruction, const std::string& csv_name,
    const std::string& csv_name_circles) {
  // Format and file path for the CSV file
  std::vector<std::tuple<std::string, Eigen::Vector2d, Eigen::Vector2d,
                         Eigen::Vector2d, Eigen::Vector2d>>
      entries = {};
  std::vector<std::tuple<std::string, Eigen::Matrix3d>> entries_circles = {};

  for (const image_t image_id : config_.Images()) {
    // Get the two images and cameras
    Image& image = reconstruction->Image(image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());
    std::string image_name = image.Name();

    // Camera qvec, tvec and params
    image.NormalizeQvec();
    double* camera_qvec = image.Qvec().data();
    double* camera_tvec = image.Tvec().data();
    double* camera_params = camera.ParamsData();

    for (CylinderClass& cylinder : this->cylinders_) {
      Cylinder c = cylinder.ToCylinder();

      // Project the circles of the cylinder
      Eigen::Matrix3d C_2D_1 = Eigen::Matrix3d::Zero();
      Eigen::Matrix3d C_2D_2 = Eigen::Matrix3d::Zero();
      c.ProjectCircles<double>(camera_qvec, camera_tvec, camera_params, C_2D_1,
                               C_2D_2);
      double p1[2], p2[2], p3[2], p4[2];
      c.ProjectToQuadrilateral<double>(camera_qvec, camera_tvec, camera_params,
                                       p1, p2, p3, p4);

      entries.emplace_back(image_name, Eigen::Vector2d(p1), Eigen::Vector2d(p2),
                           Eigen::Vector2d(p3), Eigen::Vector2d(p4));

      if (!C_2D_1.isZero()) entries_circles.emplace_back(image_name, C_2D_1);
      if (!C_2D_2.isZero()) entries_circles.emplace_back(image_name, C_2D_2);
    }
  }

  // Open the file for writing
  std::ofstream file(csv_name);

  if (file.is_open()) {
    // Write header
    file << "Image,X1,Y1,X2,Y2,X3,Y3,X4,Y4\n";

    // Write data to the file
    for (const auto& entry : entries) {
      std::string image = std::get<0>(entry);
      Eigen::Vector2d p1 = std::get<1>(entry);
      Eigen::Vector2d p2 = std::get<2>(entry);
      Eigen::Vector2d p3 = std::get<3>(entry);
      Eigen::Vector2d p4 = std::get<4>(entry);
      file << image << "," << p1.x() << "," << p1.y() << "," << p2.x() << ","
           << p2.y() << "," << p3.x() << "," << p3.y() << "," << p4.x() << ","
           << p4.y() << "\n";
    }
    // Close the file
    file.close();
    std::cout << "CSV file '" << csv_name << "' has been created.\n";
  } else {
    std::cerr << "Unable to open '" << csv_name << "' for writing.\n";
  }

  // Open the file for writing
  std::ofstream file_circles(csv_name_circles);

  if (file_circles.is_open()) {
    // Write header
    file_circles << "Image,0,1,2,3,4,5,6,7,8\n";

    // Write data to the file
    for (const auto& entry : entries_circles) {
      std::string image = std::get<0>(entry);
      Eigen::Matrix3d C = std::get<1>(entry);
      // In column major order according to Eigen (but symmetric anyways ;)
      file_circles << image << "," << C(0) << "," << C(1) << "," << C(2) << ","
                   << C(3) << "," << C(4) << "," << C(5) << "," << C(6) << ","
                   << C(7) << "," << C(8) << "\n";
    }
    // Close the file
    file_circles.close();
    std::cout << "CSV file '" << csv_name_circles << "' has been created.\n";
  } else {
    std::cerr << "Unable to open '" << csv_name_circles << "' for writing.\n";
  }
}

template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::PrintCylinders(
    const Reconstruction* const reconstruction) {
  std::cout << "Problem has " << this->cylinders_.size()
            << " cylinder(s):" << std::endl;

  for (CylinderClass& cylinder : this->cylinders_) {
    std::cout << "  - " << cylinder << ", with:" << std::endl;

    double total_iou = 0;
    for (const image_t image_id : config_.Images()) {
      // Get the two images and cameras
      const Image& image = reconstruction->Image(image_id);
      const Camera& camera = reconstruction->Camera(image.CameraId());
      const std::string image_name = image.Name();

      // Camera qvec, tvec and params
      const double* camera_qvec = image.Qvec().data();
      const double* camera_tvec = image.Tvec().data();
      const double* camera_params = camera.ParamsData();

      double iou = cylinder.ComputeSemanticIoU<double>(
          camera_qvec, camera_tvec, camera_params,
          &this->semantic_maps_bool_[image_name]);
      total_iou += iou;
      std::cout << "    - Image '" << image_name
                << "': IoU=" << std::setprecision(3) << iou << std::endl;
    }
    // Print mean IoU
    double mean_iou = total_iou / config_.Images().size();
    std::cout << "    ==> Mean IoU=" << std::setprecision(3) << mean_iou
              << std::endl;
  }
  std::cout << std::endl;
}

template <>
void GeometricSemanticBundleAdjuster<Cylinder>::SetUpCylinderManifolds() {
  std::cout << "\nCylinder manifolds:" << std::endl;
  if (!options_.refine_geometry) {
    std::cout << "  - No manifolds since the cylinders are constant."
              << std::endl;
    return;
  }

  for (size_t i = 0; i < this->cylinders_.size(); i++) {
    // constant pointer by reference to the actual Cylinder which is not
    // constant
    Cylinder& cylinder = this->cylinders_[i];

    // Parameter block for qvec
    double* param_block_qvec = cylinder.Qvec().data();

    // Check if the problem has the parameter block
    if (problem_->HasParameterBlock(param_block_qvec)) {
      std::cout
          << "  - Cylinder " << i
          << ": qvec is an optimization variable --> Set quaternion manifold."
          << std::endl;

      // Set quaternion manifold
      problem_->SetManifold(param_block_qvec, new ceres::QuaternionManifold);
    }

    // Parameter block for radius
    double* param_block_radius = &cylinder.RadiusData();

    // Check if the problem has the parameter block
    if (problem_->HasParameterBlock(param_block_radius)) {
      std::cout
          << "  - Cylinder " << i
          << ": radius is an optimization variable --> Set bounds [0, +inf)."
          << std::endl;

      // Set bounds [0, +inf)
      int index = 0;
      problem_->SetParameterLowerBound(param_block_radius, index, 0);
    }

    // Parameter block for height
    double* param_block_height = &cylinder.HeightData();

    // Check if the problem has the parameter block
    if (problem_->HasParameterBlock(param_block_height)) {
      std::cout
          << "  - Cylinder " << i
          << ": height is an optimization variable --> Set bounds [0, +inf)."
          << std::endl;

      // Set bounds [0, +inf)
      int index = 0;
      problem_->SetParameterLowerBound(param_block_radius, index, 0);
    }
  }
}

template <>
void GeometricSemanticBundleAdjuster<
    CylinderBy2Points>::SetUpCylinderManifolds() {
  std::cout << "\nCylinder manifolds:" << std::endl;
  if (!options_.refine_geometry) {
    std::cout << "  - No manifolds since the cylinders are constant."
              << std::endl;
    return;
  }

  for (size_t i = 0; i < this->cylinders_.size(); i++) {
    CylinderBy2Points& cylinder = this->cylinders_[i];

    // Parameter block for radius
    double* param_block_radius = &cylinder.RadiusData();

    // Check if the problem has the parameter block
    if (problem_->HasParameterBlock(param_block_radius)) {
      std::cout
          << "  - Cylinder " << i
          << ": radius is an optimization variable --> Set bounds [0, +inf)."
          << std::endl;

      // Set bounds [0, +inf)
      int index = 0;
      problem_->SetParameterLowerBound(param_block_radius, index, 0);
    }
  }
}

template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::SetUpManifolds(
    Reconstruction* reconstruction) {
  std::cout << "\nCamera manifolds:" << std::endl;

  // Add the cameras to the problem
  for (const image_t image_id : this->config_.Images()) {
    Image& image = reconstruction->Image(image_id);
    this->camera_ids_.insert(image.CameraId());  // Is a set so no dupplicates
  }

  // Set intrinsics to constant
  bool set_intrinsics_to_constant = false;
  for (const camera_t camera_id : camera_ids_) {
    Camera& camera = reconstruction->Camera(camera_id);
    if (problem_->HasParameterBlock(camera.ParamsData())) {
      problem_->SetParameterBlockConstant(camera.ParamsData());
      std::cout << "  - Camera with id " << camera.CameraId()
                << " params is an optimization variable --> Set to constant."
                << std::endl;
      set_intrinsics_to_constant = true;
    }
  }
  if (!set_intrinsics_to_constant) {
    std::cout << "  - Did not have to set the camera intrinsics parameters to "
                 "constant since they are not part of the problem."
              << std::endl;
  }

  // Set extrinsics manifolds
  for (const image_t image_id : this->config_.Images()) {
    Image& image = reconstruction->Image(image_id);

    double* qvec_data = image.Qvec().data();
    double* tvec_data = image.Tvec().data();

    const bool constant_pose = !this->options_.refine_extrinsics ||
                               this->config_.HasConstantPose(image_id);
    // Set pose parameterization.
    if (!constant_pose) {
      std::cout << "  - Image '" << image.Name()
                << "' qvec: is an optimization variable --> Set quaternion "
                   "manifold."
                << std::endl;
      SetQuaternionManifold(this->problem_.get(), qvec_data);
      if (config_.HasConstantTvec(image_id)) {
        std::cout << "  - Image '" << image.Name()
                  << "' tvec: is constrained --> Set subset manifold."
                  << std::endl;
        const std::vector<int>& constant_tvec_idxs = this->config_.ConstantTvec(
            image_id);  // The ids of the 3 components
                        // of the t vector i guess?
        SetSubsetManifold(3, constant_tvec_idxs, this->problem_.get(),
                          tvec_data);
      }
    } else {
      std::cout << "  - Image '" << image.Name() << "' is constant."
                << std::endl;
    }
  }
}

template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::TearDown(Reconstruction*) {
  // Nothing to do
}

template <typename CylinderClass>
bool GeometricSemanticBundleAdjuster<CylinderClass>::ReadDepthAndSemanticMaps(
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

    // Build boolean semantic maps for the tree trunks
    Eigen::MatrixXb semantic_map_bool(semantic_map.rows(), semantic_map.cols());
    semantic_map_bool.setConstant(false);

    semantic_map_bool =
        (semantic_map.array() == this->options_.trunk_semantic_class).select(true, semantic_map_bool);
    semantic_maps_bool_.insert({image_name, semantic_map_bool});
  }
  std::cout << std::endl;

  return true;
}

template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::SetUpOutputFolders() {
  std::string out = this->options_.visualization_path;
  this->image_folder_ = out + "/images";
  this->optim_steps_folder_ = out + "/optim_steps";

  std::cout << "\nOutput folders: " << std::endl;
  std::cout << "  - Image folder: " << this->image_folder_ << std::endl;
  std::cout << "  - Optimization steps folder: " << this->optim_steps_folder_
            << std::endl;

  // Empty output folder
  createFolders(this->options_.output_path, out, this->image_folder_,
                this->optim_steps_folder_);
}

////////////////////////////////////////////////////////////////////////////////
// Usual BA methods
////////////////////////////////////////////////////////////////////////////////

template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::
    AddImageToProblemForLandmarks(const image_t image_id,
                                  Reconstruction* reconstruction,
                                  ceres::LossFunction* loss_function) {
  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  // CostFunction assumes unit quaternions.
  image.NormalizeQvec();

  double* qvec_data = image.Qvec().data();
  double* tvec_data = image.Tvec().data();
  double* camera_params_data = camera.ParamsData();

  const bool constant_pose =
      !options_.refine_extrinsics || config_.HasConstantPose(image_id);

  // Add residuals to bundle adjustment problem.
  size_t num_observations = 0;
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D()) {
      continue;
    }

    num_observations += 1;
    point3D_num_observations_[point2D.Point3DId()] += 1;

    Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
    assert(point3D.Track().Length() > 1);

    ceres::CostFunction* cost_function = nullptr;

    if (constant_pose) {
      cost_function = BundleAdjustmentConstantPoseCostFunction<
          SimplePinholeCameraModel>::Create(image.Qvec(), image.Tvec(),
                                            point2D.XY());
      problem_->AddResidualBlock(cost_function, loss_function,
                                 point3D.XYZ().data(), camera_params_data);
    } else {
      cost_function =
          BundleAdjustmentCostFunction<SimplePinholeCameraModel>::Create(
              point2D.XY());

      problem_->AddResidualBlock(cost_function, loss_function, qvec_data,
                                 tvec_data, point3D.XYZ().data(),
                                 camera_params_data);
    }
  }
}

template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::
    AddPointToProblemUsualForLandmarks(const point3D_t point3D_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
  Point3D& point3D = reconstruction->Point3D(point3D_id);

  // Is 3D point already fully contained in the problem? I.e. its entire track
  // is contained in `variable_image_ids`, `constant_image_ids`,
  // `constant_x_image_ids`.
  if (point3D_num_observations_[point3D_id] == point3D.Track().Length()) {
    return;
  }

  for (const auto& track_el : point3D.Track().Elements()) {
    // Skip observations that were already added in `FillImages`.
    if (config_.HasImage(track_el.image_id)) {
      continue;
    }

    point3D_num_observations_[point3D_id] += 1;

    Image& image = reconstruction->Image(track_el.image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());
    const Point2D& point2D = image.Point2D(track_el.point2D_idx);

    // We do not want to refine the camera of images that are not
    // part of `constant_image_ids_`, `constant_image_ids_`,
    // `constant_x_image_ids_`.
    if (camera_ids_.count(image.CameraId()) == 0) {
      camera_ids_.insert(image.CameraId());
      config_.SetConstantCamera(image.CameraId());
    }

    ceres::CostFunction* cost_function = nullptr;

    cost_function = BundleAdjustmentConstantPoseCostFunction<
        SimplePinholeCameraModel>::Create(image.Qvec(), image.Tvec(),
                                          point2D.XY());
    problem_->AddResidualBlock(cost_function, loss_function,
                               point3D.XYZ().data(), camera.ParamsData());
  }
}

template <typename CylinderClass>
void GeometricSemanticBundleAdjuster<CylinderClass>::
    ParameterizePointsForLandmarks(Reconstruction* reconstruction) {
  for (const auto elem : point3D_num_observations_) {
    Point3D& point3D = reconstruction->Point3D(elem.first);
    if (point3D.Track().Length() > elem.second) {
      problem_->SetParameterBlockConstant(point3D.XYZ().data());
    }
  }

  for (const point3D_t point3D_id : config_.ConstantPoints()) {
    Point3D& point3D = reconstruction->Point3D(point3D_id);
    problem_->SetParameterBlockConstant(point3D.XYZ().data());
  }
}

////////////////////////////////////////////////////////////////////////////////
// GSBACallbackFunctor
////////////////////////////////////////////////////////////////////////////////

template <typename CylinderClass>
MyClass<CylinderClass>::MyClass(
    Reconstruction* reconstruction,
    GeometricSemanticBundleAdjuster<CylinderClass>* gsba)
    : callback_reconstruction_(reconstruction), callback_gsba_(gsba) {}

template <typename CylinderClass>
ceres::CallbackReturnType MyClass<CylinderClass>::operator()(
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

  // Print camera extrinsics information
  std::cout << "Images: " << std::endl;
  for (const image_t image_id : this->callback_gsba_->config_.Images()) {
    Image& image = this->callback_reconstruction_->Image(image_id);
    double* qvec_data = image.Qvec().data();
    double* tvec_data = image.Tvec().data();
    const bool constant_pose =
        !this->callback_gsba_->options_.refine_extrinsics ||
        this->callback_gsba_->config_.HasConstantPose(image_id);
    if (!constant_pose) {
      std::cout << "  - Image '" << image.Name() << "' (variable):" << std::endl
                << "    - qvec(w,x,y,z): [" << qvec_data[0] << ", "
                << qvec_data[1] << ", " << qvec_data[2] << ", " << qvec_data[3]
                << "]" << std::endl
                << "    - tvec: [" << tvec_data[0] << ", " << tvec_data[1]
                << ", " << tvec_data[2] << "]" << std::endl;
    } else {
      std::cout << "  - Image '" << image.Name() << "' is constant."
                << std::endl;
    }
  }
  // Print cylinders information
  this->callback_gsba_->PrintCylinders(this->callback_reconstruction_);

  // Save images for visualization
  for (const image_t image_id : this->callback_gsba_->config_.Images()) {
    Image& image = this->callback_reconstruction_->Image(image_id);
    Camera& camera = this->callback_reconstruction_->Camera(image.CameraId());
    const std::string image_name = image.Name();

    double* camera_qvec_data = image.Qvec().data();
    double* camera_tvec_data = image.Tvec().data();
    double* camera_params_data = camera.ParamsData();

    int H = this->callback_gsba_->semantic_maps_[image_name].rows();
    int W = this->callback_gsba_->semantic_maps_[image_name].cols();
    for (CylinderClass& cylinder : this->callback_gsba_->cylinders_) {
      Eigen::MatrixXb img(H, W);
      XYWH xywh;
      cylinder.ToCylinder().ProjectToMask(img, xywh, camera_qvec_data,
                                          camera_tvec_data, camera_params_data);
      saveMatrixToJpg<bool>(img, this->callback_gsba_->image_folder_ + "/" +
                                     image_name + "_cylinder_mask_optim_step_" +
                                     std::to_string(summary.iteration) +
                                     ".jpg");
    }
  }

  // Create new folder
  std::string folder_step = this->callback_gsba_->optim_steps_folder_ +
                            "/step_" + std::to_string(summary.iteration);
  std::string folder_step_text = folder_step + "/text";
  createFolders(folder_step, folder_step_text);

  // Export camera poses
  callback_reconstruction_->Write(folder_step);
  callback_reconstruction_->WriteText(folder_step_text);

  // Export cylinder
  exportCylindersToText(this->callback_gsba_->cylinders_,
                        folder_step + "/cylinders.txt");
  return ceres::SOLVER_CONTINUE;
}

}  // namespace colmap
