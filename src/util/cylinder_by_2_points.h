#ifndef COLMAP_SRC_UTIL_CYLINDER_BY_2_POINTS_H_
#define COLMAP_SRC_UTIL_CYLINDER_BY_2_POINTS_H_

#include <Eigen/Core>
#include <vector>

#include "util/rotation_extension.h"
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
#include "util/utils.h"

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// CylinderBy2Points
////////////////////////////////////////////////////////////////////////////////

class CylinderBy2Points {
 public:
  // Default constructor
  explicit CylinderBy2Points()
      : eig_tvec_1_(0, 0, 1), eig_tvec_2_(0, 0, 0), radius_(1) {}

  explicit CylinderBy2Points(const double tvec_1[3], const double tvec_2[3],
                             const double radius)
      : eig_tvec_1_(tvec_1[0], tvec_1[1], tvec_1[2]),
        eig_tvec_2_(tvec_2[0], tvec_2[1], tvec_2[2]),
        radius_(radius) {
    this->Check();
  }

  // Constructor from usual Cylinder
  explicit CylinderBy2Points(const Cylinder& c)
      : eig_tvec_1_(c.Tvec()),
        eig_tvec_2_(c.GetEigUpperTvec()),
        radius_(c.RadiusData()) {}

  // Check the validity of the cylinder
  bool Check() {
    if (this->radius_ <= 0) {
#ifdef _DEBUG
      std::cerr << "WARNING: Can't have a negative or zero radius cylinder."
                << std::endl;
#endif
      this->radius_ = 1e-4;
    }
    return true;
  }

  // Copy
  CylinderBy2Points(const CylinderBy2Points& c)
      : eig_tvec_1_(c.eig_tvec_1_),
        eig_tvec_2_(c.eig_tvec_2_),
        radius_(c.radius_) {}

  // Destructor
  ~CylinderBy2Points() {}

  // Height
  double Height() const { return (eig_tvec_1_ - eig_tvec_2_).norm(); }

  // Getters
  Eigen::Vector3d& Tvec1() { return eig_tvec_1_; }
  const Eigen::Vector3d& Tvec1() const { return eig_tvec_1_; }

  Eigen::Vector3d& Tvec2() { return eig_tvec_2_; }
  const Eigen::Vector3d& Tvec2() const { return eig_tvec_2_; }

  double& RadiusData() { return radius_; }
  const double& RadiusData() const { return radius_; }

  // Convert to Cylinder
  Cylinder ToCylinder() const {
    Eigen::Vector3d d = this->eig_tvec_2_ - eig_tvec_1_;
    d /= d.norm();

    Eigen::Vector3d z(0, 0, 1);

    Eigen::Vector3d axis = z.cross(d);

    if (std::fabs(axis.norm()) < 1e-10) {
      axis = Eigen::Vector3d(1, 0, 0);
    } else {
      axis /= axis.norm();
    }

    double angle = std::acos(z.dot(d));

    Eigen::Vector3d angle_axis = angle * axis;

    Eigen::Vector4d qvec;
    ceres::AngleAxisToQuaternion(angle_axis.data(), qvec.data());

    return Cylinder(qvec, this->eig_tvec_1_, this->radius_, this->Height());
  }

  template <typename T = double>
  double ComputeSemanticIoU(
      const T* camera_qvec, const T* const camera_tvec,
      const T* const camera_params,
      const Eigen::MatrixXb* const semantic_map_bool) const {
    return this->ToCylinder().ComputeSemanticIoU<T>(
        camera_qvec, camera_tvec, camera_params, semantic_map_bool);
  }

 private:
  Eigen::Vector3d eig_tvec_1_;
  Eigen::Vector3d eig_tvec_2_;
  double radius_;

 public:
  // Print (overload << operator)
  friend std::ostream& operator<<(std::ostream& os,
                                  const CylinderBy2Points& c) {
    os << "Cylinder(t_1=[" << c.eig_tvec_1_(0) << ", " << c.eig_tvec_1_(1)
       << ", " << c.eig_tvec_1_(2) << "], t_2=[" << c.eig_tvec_2_(0) << ", "
       << c.eig_tvec_2_(1) << ", " << c.eig_tvec_2_(2)
       << "], radius=" << c.radius_ << ") [height=" << c.Height() << "]";
    return os;
  }
};

// Export cylinders to text
inline bool exportCylindersToText(
    const std::vector<CylinderBy2Points>& cylinders, std::string output_path) {
  std::ofstream output(output_path);
  for (const CylinderBy2Points& cylinder_by_2_points : cylinders) {
    output << cylinder_by_2_points.ToCylinder().to_string() << std::endl;
  }
  output.close();
  return true;
}

// Read cylinders from a text file and push back to a cylinder vector
inline bool pushBackCylindersReadFromText(
    std::string input_path, std::vector<CylinderBy2Points>& cylinders) {
  std::ifstream input(input_path);
  std::string line;
  while (std::getline(input, line)) {
    Cylinder cylinder(line);
    cylinders.push_back(CylinderBy2Points(cylinder));
  }
  return true;
}

}  // namespace colmap

#endif  // CCOLMAP_SRC_UTIL_CYLINDER_BY_2_POINTS_H_