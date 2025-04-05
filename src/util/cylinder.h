#ifndef COLMAP_SRC_UTIL_CYLINDER_H_
#define COLMAP_SRC_UTIL_CYLINDER_H_

#include <Eigen/Core>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "util/matrix_vis.h"
#include "util/rotation_extension.h"
#include "util/utils.h"
#include "util/xywh.h"

namespace colmap {

template <typename T = int>
inline void drawQuadrilateral(Eigen::Ref<Eigen::MatrixX<T>> matrix, XYWH& xywh,
                              const Eigen::Ref<const Eigen::Vector2d> p1,
                              const Eigen::Ref<const Eigen::Vector2d> p2,
                              const Eigen::Ref<const Eigen::Vector2d> p3,
                              const Eigen::Ref<const Eigen::Vector2d> p4) {
  // Dimensions of the matrix
  const int H = matrix.rows(), W = matrix.cols();

  // Initialize the matrix to 'false'
  matrix.setConstant(T(0));

  // Find bounding box of interest
  xywh.setToBoundPoints(p1, p2, p3, p4);
  xywh.shrinkToFitInToFitIn(XYWH(0, 0, W, H));

  // Quadrilateral completely out of the image
  if (xywh == 0) {
#ifdef _DEBUG
    std::cout << "WARNING: Cylinder is completely out of the image."
              << std::endl;
#endif
  }

  // Set 'true' inside the bounding box
  matrix.block(xywh.y(), xywh.x(), xywh.h(), xywh.w()).setConstant(T(1));

  // Actual boundaries
  auto drawEdges = [&W, &H, &matrix](
                       const Eigen::Ref<const Eigen::Vector2d> p_first,
                       const Eigen::Ref<const Eigen::Vector2d> p_second) {
    XYWH xywh12;
    xywh12.setToBoundPoints(p_first, p_second);
    xywh12.shrinkToFitInToFitIn(XYWH(0, 0, W, H));

    // Don't draw edges if completely out of the image
    if (xywh12 == 0) return;

    Eigen::MatrixXd xValues =
        Eigen::RowVectorXd::LinSpaced(xywh12.w(), xywh12.x(), xywh12.x_end())
            .replicate(xywh12.h(), 1);
    Eigen::MatrixXd yValues =
        Eigen::VectorXd::LinSpaced(xywh12.h(), xywh12.y(), xywh12.y_end())
            .replicate(1, xywh12.w());

    Eigen::MatrixXd crossprod =
        (xValues.array() - p_first.x()).array() * (p_second.y() - p_first.y()) -
        (yValues.array() - p_first.y()).array() * (p_second.x() - p_first.x());

    matrix.block(xywh12.y(), xywh12.x(), xywh12.h(), xywh12.w()) =
        (crossprod.array() > 0)
            .select(T(0), matrix.block(xywh12.y(), xywh12.x(), xywh12.h(),
                                       xywh12.w()));
  };

  drawEdges(p1, p2);
  drawEdges(p2, p3);
  drawEdges(p3, p4);
  drawEdges(p4, p1);

  // Avoid full boxes at the corners
  auto correctCorners = [&xywh, &W, &H,
                         &matrix](const Eigen::Ref<const Eigen::Vector2d> p) {
    bool border_touch = false;
    // Check if touching border
    if (p.x() - xywh.x() < 1 || xywh.x_end() - p.x() < 1 ||
        p.y() - xywh.y() < 1 || xywh.y_end() - p.y() < 1) {
      // Touching border -> nothing to do
      return;
    }
    // Not touching border -> erase white rectangle
    else {
      std::vector<Eigen::Vector2i> corners = xywh.Corners();
      std::sort(corners.begin(), corners.end(),
                [&p](const Eigen::Vector2i& a, const Eigen::Vector2i& b) {
                  return (p - a.cast<double>()).norm() <
                         (p - b.cast<double>()).norm();
                });
      Eigen::Vector2d closest_corner = corners[0].cast<double>();

      XYWH xywh_corner;
      Eigen::Vector2d pp = p.cast<double>();  // copy for shrink
      xywh_corner.setToBoundPoints(closest_corner, pp);
      xywh_corner.shrinkToFitInToFitIn(XYWH(0, 0, W, H));
      matrix
          .block(xywh_corner.y(), xywh_corner.x(), xywh_corner.h(),
                 xywh_corner.w())
          .setConstant(T(0));
    }
  };

  correctCorners(p1);
  correctCorners(p2);
  correctCorners(p3);
  correctCorners(p4);
}

#ifdef DRAW_HALFSPACE
// Check wheter a matrix of points is inside an ellipse or not.
// Inefficient way to draw a halfspace. Not being used.
template <typename T = int>
inline void drawHalfspace(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> matrix,
    const Eigen::Ref<const Eigen::Vector2d> p1,
    const Eigen::Ref<const Eigen::Vector2d> p2) {
  // Given matrix dimensions M and N
  const int H = matrix.rows(), W = matrix.cols();

  // Create matrices for x and y values
  Eigen::MatrixXd xValues =
      Eigen::RowVectorXd::LinSpaced(W, 0, W - 1.).replicate(H, 1);
  Eigen::MatrixXd yValues =
      Eigen::VectorXd::LinSpaced(H, 0, H - 1.).replicate(1, W);

  // Compute the difference between points and p1
  Eigen::MatrixXd diffX = xValues.array() - p1.x();
  Eigen::MatrixXd diffY = yValues.array() - p1.y();

  // Compute the cross product with (p2 - p1)
  Eigen::MatrixXd crossprod =
      diffX.array() * (p2.y() - p1.y()) - diffY.array() * (p2.x() - p1.x());

  // Set the matrix to zero outside of the halfspace
  matrix = (crossprod.array() > 0).select(T(0), matrix);
}
#endif

template <typename T = double>
inline bool projectCircle(const T circle_qvec[4], const T circle_tvec[3],
                          const T camera_qvec[4], const T camera_tvec[3],
                          const T camera_params[3], const T radius,
                          Eigen::Ref<Eigen::Matrix3d> C_2D) {
  // Set the homogeneous circle matrix to zero in case of errors
  C_2D = Eigen::Matrix3d::Zero();

  if (radius <= T(0)) {
    std::cerr << "Radius can't be zero or negative !" << std::endl;
    return false;
  }

  // Compute transformation from camera coordinates to circle coordinates
  T camera_w_circle_qvec[4];
  T camera_w_circle_tvec[4];
  ceres::PoseProduct<T>(camera_qvec, camera_tvec, circle_qvec, circle_tvec,
                        camera_w_circle_qvec, camera_w_circle_tvec);

  T camera_w_circle_Rvec[9];
  ceres::QuaternionToRotation<T>(camera_w_circle_qvec, camera_w_circle_Rvec);

  // Build the T matrix: T = np.c_[R[:, :2], t[:, None]] # (3,3)
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> T_matrix(
      camera_w_circle_Rvec);

  // Replace the last column by the translation vector
  T_matrix(0, 2) = camera_w_circle_tvec[0];
  T_matrix(1, 2) = camera_w_circle_tvec[1];
  T_matrix(2, 2) = camera_w_circle_tvec[2];

  // Build the intrinsics matrix K
  Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
  K(0, 0) = camera_params[0];  // f
  K(1, 1) = camera_params[0];  // f
  K(0, 2) = camera_params[1];  // c_1
  K(1, 2) = camera_params[2];  // c_2
  K(2, 2) = 1;                 // homogeneous 1

  // Compute the homography matrix H = K @ T
  Eigen::Matrix3d H = K * T_matrix;

  // Invert the homography matrix
  if (H.determinant() == 0) {
    std::cerr << "Homography matrix is not invertible." << std::endl;
    return false;
  }
  Eigen::Matrix3d H_inv = H.inverse();

  Eigen::Matrix3d C_3D = Eigen::Matrix3d::Zero();
  C_3D(0, 0) = 1 / radius / radius;
  C_3D(1, 1) = 1 / radius / radius;
  C_3D(2, 2) = -1;

  C_2D = H_inv.transpose() * C_3D * H_inv;
  C_2D = -C_2D / C_2D(2, 2);

  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Cylinder
////////////////////////////////////////////////////////////////////////////////

class Cylinder {
 public:
  // Default constructor
  explicit Cylinder()
      : eig_qvec_(0, 0, 0, 1), eig_tvec_(0, 0, 0), radius_(1), height_(1) {}

  explicit Cylinder(const double qvec[4], const double tvec[3],
                    const double radius, const double height)
      : eig_qvec_(qvec[0], qvec[1], qvec[2], qvec[3]),
        eig_tvec_(tvec[0], tvec[1], tvec[2]),
        radius_(radius),
        height_(height) {
    this->Check();
  }

  explicit Cylinder(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                    const double radius, const double height)
      : eig_qvec_(qvec), eig_tvec_(tvec), radius_(radius), height_(height) {
    this->Check();
  }

  // Copy
  Cylinder(const Cylinder& c)
      : eig_qvec_(c.eig_qvec_),
        eig_tvec_(c.eig_tvec_),
        radius_(c.radius_),
        height_(c.height_) {}

  // Destructor
  ~Cylinder() {}

  // Check the validity of the cylinder
  bool Check() {
// Note: normalization cannot be checked due to the quaternion manifold during
// the optimization
#ifdef CHECK_NORMALIZED_QUATERNION
    double qvec_norm = qvec_[0] * qvec_[0] + qvec_[1] * qvec_[1] +
                       qvec_[2] * qvec_[2] + qvec_[3] * qvec_[3];
    if (std::fabs(qvec_norm - 1) > 1e-7) {
      throw std::runtime_error(
          "Can't use a not normalized quaternion to define a cylinder.");

      if (std::fabs(this->eig_qvec_.norm() - 1) > 1e-7) {
        std::cerr << "ERROR: Can't use a not normalized quaternion to define
            a "
                     "cylinder."
                  << std::endl;
        throw std::runtime_error(
            "Can't use a not normalized quaternion to define a cylinder.");
      }
    }
#endif
    if (this->radius_ <= 0) {
#ifdef _DEBUG
      std::cerr << "WARNING: Can't have a negative or zero radius cylinder."
                << std::endl;
#endif
      this->radius_ = 1e-4;
    }
    if (this->height_ <= 0) {
#ifdef _DEBUG
      std::cerr << "WARNING: Can't have a negative or zero height cylinder."
                << std::endl;
#endif
      this->height_ = 1e-4;
    }
    return true;
  }

  Cylinder ToCylinder() const { return Cylinder(*this); }

  // Serialize to string
  std::string to_string() const {
    std::ostringstream oss;
    oss << "q " << this->Qvec()(0) << " " << this->Qvec()(1) << " "
        << this->Qvec()(2) << " " << this->Qvec()(3) << " t " << this->Tvec()(0)
        << " " << this->Tvec()(1) << " " << this->Tvec()(2) << " r "
        << this->RadiusData() << " h " << this->HeightData();
    return oss.str();
  }

  // Deserialize from string
  Cylinder(const std::string& str) {
    auto fail = []() {
      std::cerr << "ERROR: creating Cylinder from string failed." << std::endl;
      throw std::runtime_error("ERROR: creating Cylinder from string failed.");
    };

    std::istringstream iss(str);
    std::string temp;
    iss >> temp;  // q
    if (temp != "q") fail();
    iss >> this->eig_qvec_(0);
    iss >> this->eig_qvec_(1);
    iss >> this->eig_qvec_(2);
    iss >> this->eig_qvec_(3);

    iss >> temp;  // t
    if (temp != "t") fail();
    iss >> this->eig_tvec_(0);
    iss >> this->eig_tvec_(1);
    iss >> this->eig_tvec_(2);

    iss >> temp;  // r
    if (temp != "r") fail();
    iss >> this->radius_;

    iss >> temp;  // h
    if (temp != "h") fail();
    iss >> this->height_;

    this->Check();
  }

  template <typename T = double>
  bool ProjectCircles(const T camera_qvec[4], const T camera_tvec[3],
                      const T camera_params[3],
                      Eigen::Ref<Eigen::Matrix3d> C_2D_1,
                      Eigen::Ref<Eigen::Matrix3d> C_2D_2) const {
    bool success_1 = projectCircle<double>(
        this->Qvec().data(), this->Tvec().data(), camera_qvec, camera_tvec,
        camera_params, this->radius_, C_2D_1);

    double upper_tvec[3];
    this->GetUpperTvec(upper_tvec);
    bool success_2 = projectCircle<double>(
        this->Qvec().data(), upper_tvec, camera_qvec, camera_tvec,
        camera_params, this->radius_, C_2D_2);
    return success_1 && success_2;
  }

  // Get edge points
  // Note: camera qvec, tvec represent the transformation from world to
  // camera. That is, tvec does NOT represent the center camera in the world
  // coordinates.
  bool GetEdgePoints(const double* camera_qvec, const double* camera_tvec,
                     double p1_w_world[3], double p2_w_world[3],
                     double p3_w_world[3], double p4_w_world[3]) const {
    // Get camera pose inverse (from camera to world)
    double camera_w_world_qvec[4];
    double camera_w_world_tvec[3];
    ceres::PoseInverse<double>(camera_qvec, camera_tvec, camera_w_world_qvec,
                               camera_w_world_tvec);

    // Get pose inverse (from world to cylinder coordinates)
    double qvec_inverse[4];
    double tvec_inverse[3];
    ceres::PoseInverse<double>(this->Qvec().data(), this->Tvec().data(),
                               qvec_inverse, tvec_inverse);

    // Transform camera center to the cylinder coordinates
    double camera_tvec_w_cylinder[3];
    ceres::PoseTransformPoint(qvec_inverse, tvec_inverse, camera_w_world_tvec,
                              camera_tvec_w_cylinder);

    // Project the camera center to the cylinder's plane
    camera_tvec_w_cylinder[2] = 0;  // set z component to zero

    // Compute the distance from the projected camera center to the center
    // of the cylinder
    double dist =
        std::sqrt(camera_tvec_w_cylinder[0] * camera_tvec_w_cylinder[0] +
                  camera_tvec_w_cylinder[1] * camera_tvec_w_cylinder[1]);

    // Ascertain that the camera center is outside of the "infinite"
    // cylinder
    if (dist <= this->radius_) {
      std::cerr << "ERROR: Camera center is indie of the infinite cylinder."
                << std::endl;
      throw std::runtime_error(
          "Camera center is indie of the infinite cylinder");
      return false;
    }

    // Direction vector from the center of the lower circle of the circle
    // pointing towards the projected camera center
    // (camera_tvec_w_cylinder), with a norm of the radius of the cylinder
    double direction_vector[3];
    direction_vector[0] = camera_tvec_w_cylinder[0] / dist * this->radius_;
    direction_vector[1] = camera_tvec_w_cylinder[1] / dist * this->radius_;
    direction_vector[2] = 0;

    // Compute the angle between relevant for the tangent computation
    double beta = std::acos(this->radius_ / dist);

    // Compute the four edge points in the cylinder coordinates
    double angle_axis_positive[3] = {0, 0, beta};  // z-axis with angle of beta
    double p1[3];
    ceres::AngleAxisRotatePoint(angle_axis_positive, direction_vector, p1);

    double angle_axis_negative[3] = {0, 0, -beta};  // z-axis with angle of beta
    double p2[3];
    ceres::AngleAxisRotatePoint(angle_axis_negative, direction_vector, p2);

    double p3[3] = {p2[0], p2[1], p2[2] + this->height_};
    double p4[3] = {p1[0], p1[1], p1[2] + this->height_};

    // Transform the four edge points to the world frame
    ceres::PoseTransformPoint(this->Qvec().data(), this->Tvec().data(), p1,
                              p1_w_world);
    ceres::PoseTransformPoint(this->Qvec().data(), this->Tvec().data(), p2,
                              p2_w_world);
    ceres::PoseTransformPoint(this->Qvec().data(), this->Tvec().data(), p3,
                              p3_w_world);
    ceres::PoseTransformPoint(this->Qvec().data(), this->Tvec().data(), p4,
                              p4_w_world);

    return true;
  }

  // Project
  template <typename T = double>
  void ProjectToQuadrilateral(const T* const camera_qvec,
                              const T* const camera_tvec,
                              const T* const camera_params, double p1_2D[2],
                              double p2_2D[2], double p3_2D[2],
                              double p4_2D[2]) const {
    // Ascertain T is double
    if (!std::is_same<T, double>::value) {
      std::cout << "ERROR: T is not double." << std::endl;
      throw std::runtime_error("An error occurred.");
    }

    // Get the 4 edges points
    double p1_3D[3], p2_3D[3], p3_3D[3], p4_3D[3];
    bool success = this->GetEdgePoints(camera_qvec, camera_tvec, p1_3D, p2_3D,
                                       p3_3D, p4_3D);
    if (!success) {
      std::cerr << "ERROR: GetEdgePoints() failed." << std::endl;
      throw std::runtime_error("GetEdgePoints() failed.");
    }

    // Project the 3D points
    // Throws error if points lie behind the camera
    try {
      simplePinholeProject<T>(camera_qvec, camera_tvec, camera_params, p1_3D,
                              p1_2D);
      simplePinholeProject<T>(camera_qvec, camera_tvec, camera_params, p2_3D,
                              p2_2D);
      simplePinholeProject<T>(camera_qvec, camera_tvec, camera_params, p3_3D,
                              p3_2D);
      simplePinholeProject<T>(camera_qvec, camera_tvec, camera_params, p4_3D,
                              p4_2D);
    } catch (const std::exception& e) {
      throw;
    }
    // Order the edge points
    // Note: using image coordinates here, so y-axis pointing downwards.
    Eigen::Vector2d v0 = Eigen::Vector2d(p2_2D) - Eigen::Vector2d(p1_2D);
    Eigen::Vector2d v1 = Eigen::Vector2d(p3_2D) - Eigen::Vector2d(p1_2D);
    // Reverse order if v0 x v1 < 0
    if (v0.x() * v1.y() - v0.y() * v1.x() < 0) {
      std::swap(p2_2D[0], p4_2D[0]);
      std::swap(p2_2D[1], p4_2D[1]);
    }

    return;
  }

  // Only projected to a quadrilateral boolean mask, i.e. the two circles
  // are not reprojected [can be improved, at the expense of efficiency]
  void ProjectToMask(Eigen::Ref<Eigen::MatrixXb> mask, XYWH& xywh,
                     const double camera_qvec[4], const double camera_tvec[3],
                     const double camera_params[3]) const {
    // Get edge points
    double p1[2], p2[2], p3[2], p4[2];
    try {
      this->ProjectToQuadrilateral<double>(camera_qvec, camera_tvec,
                                           camera_params, p1, p2, p3, p4);
    } catch (const std::exception& e) {
      throw;
    }

    // Draw the quadrilateral
    drawQuadrilateral<bool>(mask, xywh, Eigen::Vector2d(p1),
                            Eigen::Vector2d(p2), Eigen::Vector2d(p3),
                            Eigen::Vector2d(p4));
  }

  template <typename T = double>
  double ComputeSemanticIoU(
      const T* camera_qvec, const T* const camera_tvec,
      const T* const camera_params,
      const Eigen::MatrixXb* const semantic_map_bool) const {
    // Get projected mask
    const int H = semantic_map_bool->rows();
    const int W = semantic_map_bool->cols();
    Eigen::MatrixXb mask(H, W);  //
    XYWH xywh;
    try {
      this->ProjectToMask(mask, xywh, camera_qvec, camera_tvec, camera_params);
    } catch (const std::exception& e) {
      std::cerr << "WARNING: return IoU of 0: " << e.what() << std::endl;
      return 0;
    }

    // Only look at the bounding box
    // Sub-block without copy
    Eigen::Block<Eigen::MatrixXb> mask_pos_bbox =
        mask.block(xywh.y(), xywh.x(), xywh.h(), xywh.w());

    const Eigen::Block<const Eigen::MatrixXb> semantic_bbox =
        semantic_map_bool->block(xywh.y(), xywh.x(), xywh.h(), xywh.w());

    // Compute various number of pixels
    int sem_pos_count = semantic_map_bool->count();  // inexpensive (const)
    int sem_pos_inside_count = semantic_bbox.count();
    int sem_pos_outside_count = sem_pos_count - sem_pos_inside_count;
    int sem_neg_count = H * W - sem_pos_count;
    int sem_neg_inside_count = xywh.nPixels() - sem_pos_inside_count;
    int sem_neg_outside_count = sem_neg_count - sem_neg_inside_count;

    // Compute true positives, false positives, false negatives
    int tp = semantic_bbox.array().select(mask_pos_bbox, false).count();
    int fp =
        (semantic_bbox.array() == false).select(mask_pos_bbox, false).count();
    int fn = sem_pos_outside_count +
             (xywh.nPixels() -
              semantic_bbox.array().select(mask_pos_bbox, true).count());

    // Compute Intersection over Union
    double iou = static_cast<double>(tp) / (tp * 1. + fp + fn);
    return iou;
  }

 public:
  // Getters
  double& RadiusData() { return radius_; }
  double& HeightData() { return height_; }

  const double& RadiusData() const { return radius_; }
  const double& HeightData() const { return height_; }

  // Setters
  void SetQvec(const double qvec[4]) { eig_qvec_ = Eigen::Vector4d(qvec); }
  void SetTvec(const double tvec[4]) { eig_tvec_ = Eigen::Vector3d(tvec); }
  void SetRadius(const double radius) { radius_ = radius; }
  void SetHeight(const double height) { height_ = height; }

  Eigen::Vector4d& Qvec() {
    //    std::cout << "Called varying Qvec()" << std::endl;
    return eig_qvec_;
  }
  const Eigen::Vector4d& Qvec() const {
    //   std::cout << "Called CONSTANT Qvec()" << std::endl;
    return eig_qvec_;
  }
  Eigen::Vector3d& Tvec() { return eig_tvec_; }
  const Eigen::Vector3d& Tvec() const { return eig_tvec_; }

  void GetUpperTvec(double upper_tvec[3]) const {
    double upper_circle[3] = {0, 0, this->height_};
    ceres::PoseTransformPoint(this->Qvec().data(), this->Tvec().data(),
                              upper_circle, upper_tvec);
  }

  Eigen::Vector3d GetEigUpperTvec() const {
    double upper_tvec[3];
    this->GetUpperTvec(upper_tvec);
    return Eigen::Vector3d(upper_tvec[0], upper_tvec[1], upper_tvec[2]);
  }

 private:
  Eigen::Vector4d eig_qvec_;
  Eigen::Vector3d eig_tvec_;
  double radius_;
  double height_;

 public:
  // Print (overload << operator)
  friend std::ostream& operator<<(std::ostream& os, const Cylinder& cylinder) {
    os << "Cylinder(q=[" << cylinder.eig_qvec_(0) << ", "
       << cylinder.eig_qvec_(1) << ", " << cylinder.eig_qvec_(2) << ", "
       << cylinder.eig_qvec_(3) << "], t=[" << cylinder.eig_tvec_(0) << ", "
       << cylinder.eig_tvec_(1) << ", " << cylinder.eig_tvec_(2)
       << "], radius=" << cylinder.radius_ << ", height=" << cylinder.height_
       << ")";
    return os;
  }
  void print() const {
    std::cout << "Cylinder with:\n"
              << "  - qvec: (" << eig_qvec_(0) << ", " << eig_qvec_(1) << ", "
              << eig_qvec_(2) << ", " << eig_qvec_(3) << ") \n"
              << "  - tvec: (" << eig_tvec_(0) << ", " << eig_tvec_(1) << ", "
              << eig_tvec_(2) << ") \n"
              << "  - radius: " << radius_ << "\n"
              << "  - height: " << height_ << "\n";
  }
};

// Export cylinders to text
inline bool exportCylindersToText(const std::vector<Cylinder>& cylinders,
                                  std::string output_path) {
  std::ofstream output(output_path);
  for (const Cylinder& cylinder : cylinders) {
    output << cylinder.to_string() << std::endl;
  }
  output.close();
  return true;
}

// Read cylinders from a text file and push back to a cylinder vector
inline bool pushBackCylindersReadFromText(std::string input_path,
                                          std::vector<Cylinder>& cylinders) {
  std::ifstream input(input_path);
  std::string line;
  while (std::getline(input, line)) {
    // directly in place, no copy, using constructor from string
    cylinders.emplace_back(line);
  }
  return true;
}

}  // namespace colmap

#endif  // CCOLMAP_SRC_UTIL_CYLINDER_H_