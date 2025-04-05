#ifndef COLMAP_SRC_UTIL_UTILS_H_
#define COLMAP_SRC_UTIL_UTILS_H_

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <boost/filesystem.hpp>

namespace Eigen {
// MatrixXb
typedef Matrix<bool, Dynamic, Dynamic> MatrixXb;

// MatrixX<T>
template <typename T>
using MatrixX = Matrix<T, Dynamic, Dynamic>;
}  // namespace Eigen

namespace colmap {

template <typename T = double>
inline void simplePinholeProject(const T* const camera_qvec,
                                 const T* const camera_tvec,
                                 const T* const camera_params,
                                 const T* const point3D, double* point2D) {
  // Transform point to camera coordinates
  T point_w_camera[3];
  ceres::PoseTransformPoint(camera_qvec, camera_tvec, point3D, point_w_camera);

  // Ascertain points are in front of camera
  if (point_w_camera[2] <= 0) {
    std::cerr << "ERROR: Point lies behind camera in simplePinholeProject."
              << std::endl;
    throw std::runtime_error(
        "Point lies behind camera in simplePinholeProject.");
  }

  // Projection
  point_w_camera[0] /= point_w_camera[2];
  point_w_camera[1] /= point_w_camera[2];

  // World to image, equivalent to SimplePinholeCameraModel::WorldToImage
  const T f = camera_params[0];
  const T c1 = camera_params[1];
  const T c2 = camera_params[2];

  // No distortion
  // Transform to image coordinates
  point2D[0] = f * point_w_camera[0] + c1;
  point2D[1] = f * point_w_camera[1] + c2;

  return;
}

inline void deleteFolderContent(const std::string& folder) {
  try {
    boost::filesystem::path directory(folder);

    if (boost::filesystem::exists(directory) &&
        boost::filesystem::is_directory(directory)) {
      // Collect paths before deletion
      std::vector<boost::filesystem::path> pathsToDelete;
      for (const auto& entry :
           boost::filesystem::recursive_directory_iterator(directory)) {
        pathsToDelete.push_back(entry.path());
      }

      // Delete collected paths
      for (const auto& path : pathsToDelete) {
        if (boost::filesystem::is_regular_file(path)) {
          boost::filesystem::remove(path);
          std::cout << "Deleted file: " << path << std::endl;
        } else if (boost::filesystem::is_directory(path)) {
          boost::filesystem::remove_all(path);
          std::cout << "Deleted directory: " << path << std::endl;
        }
      }
    } else {
      std::cerr << "The specified path is not a directory." << std::endl;
    }
  } catch (const boost::filesystem::filesystem_error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    throw std::runtime_error("ERROR: " + std::string(e.what()));
  }
}

inline void createFolder(const std::string& folder, const bool empty = true) {
  if (!boost::filesystem::exists(folder)) {
    try {
      if (boost::filesystem::create_directory(folder)) {
        std::cout << "Folder created successfully at: " << folder << std::endl;
      } else {
        std::cerr << "ERROR: Error creating folder at: " << folder << std::endl;
        throw std::runtime_error("ERROR: Error creating folder at: " + folder);
      }
    } catch (...) {
      std::cerr << "ERROR: Error creating folder at: " << folder << std::endl;
      throw std::runtime_error("ERROR: Error creating folder at: " + folder);
    }
  } else {
    std::cout << "Folder already exists at: " << folder << std::endl;
    if (empty) deleteFolderContent(folder);
  }
}
template <typename... Args>
inline void createFolders(const Args&... folders) {
  for (const std::string& folder : {folders...}) {
    createFolder(folder);
  }
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_UTILS_H_
