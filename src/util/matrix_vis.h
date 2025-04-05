#ifndef COLMAP_SRC_UTIL_MATRIX_VIS_H_
#define COLMAP_SRC_UTIL_MATRIX_VIS_H_

#include <Eigen/Core>

#include "util/bitmap.h"
#include <FreeImage.h>

namespace colmap {

template <typename T = double>
inline bool saveMatrixToJpg(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> matrix,
    const std::string& path) {
  if (!std::is_same<T, bool>::value) {
    // Check values range
    if (matrix.maxCoeff() > 1 || matrix.minCoeff() < 0) {
      std::cout << "[SaveMatrixToJpg error] Values are not in range [0,1]."
                << std::endl;
      return false;
    }
  }

  // Create bitmap
  Bitmap b;
  if (!CreateFromMatrix<T>(matrix, b)) {
    std::cout << "[SaveMatrixToJpg error] Fail." << std::endl;
    return false;
  }

  // Save bitmap to the provided path
  if (!b.Write(path.c_str())) {
    std::cout << "[SaveMatrixToJpg error] Fail." << std::endl;
    return false;
  }

  // Success
  return true;
}

// Draw a point on a matrix
template <typename T = float>
inline void drawPointOnMatrix(Eigen::Ref<Eigen::MatrixX<T>> matrix,
                               const Eigen::Vector2i& point,
                               const double& radius, const T& color) {
  int i = point(1);
  int j = point(0);
  for (int row = std::max<int>(0, i - radius);
       row <= std::min<int>(int(matrix.rows()) - 1, i + radius); ++row) {
    for (int col = std::max<int>(0, j - radius);
         col <= std::min<int>(int(matrix.cols()) - 1, j + radius); ++col) {
      // Check the distance from (i, j) and set the coefficient to 1 if within
      // the radius
      if (std::fabs(row - i) * std::fabs(row - i) +
              std::fabs(col - j) * std::fabs(col - j) <=
          radius * radius) {
        matrix(row, col) = color;
      }
    }
  }
}

// Function to convert the Bitmap to Eigen::MatrixXd
template <typename T = double>
inline bool ConvertToMatrix(
    const Bitmap& bitmap,
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> matrix) {
  // Set matrix to zero
  matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(
      bitmap.Height(), bitmap.Width());

  // Loop through the pixels
  for (int y = 0; y < bitmap.Height(); ++y) {
    for (int x = 0; x < bitmap.Width(); ++x) {
      // Get the pixel value
      BitmapColor<uint8_t> pixelColor;
      if (bitmap.GetPixel(x, y, &pixelColor)) {
        // For grayscale, use the red component
        matrix(y, x) = static_cast<T>(static_cast<double>(pixelColor.r) / 255);
      } else {
        // Handle error case or set default value
        std::cerr << "Error reading pixel " << x << ", " << y << "."
                  << std::endl;
        throw std::runtime_error("Error reading pixel.");
      }
    }
  }
  return matrix;
};

template <typename T = double>
inline bool CreateFromMatrix(
    Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> matrix,
    Bitmap& bitmap, const bool& as_rgb = false) {
  // Allocate the bitmap
  const int width = matrix.cols();
  const int height = matrix.rows();
  bitmap.Allocate(width, height, as_rgb);

  if (as_rgb) {
    std::cout << "RGB reading not implemented! -> Error" << std::endl;
    return false;
  }

  // Loop through the matrix
  for (int y = 0; y < bitmap.Height(); ++y) {
    for (int x = 0; x < bitmap.Width(); ++x) {
      // Get the matrix value and cast to double
      double value = static_cast<double>(matrix(y, x));
      if (value < 0 || value > 1) {
        std::cerr << "Matrix value " << value << " is not in bounds [0,1]"
                  << std::endl;
        throw std::runtime_error("Error creating bitmap from matrix.");
        return false;
      }

      // Scaling the matrix value from [0, 1] to [0, 255]
      BitmapColor<uint8_t> pixelColor;
      pixelColor.r = static_cast<uint8_t>(value * 255.0);
      pixelColor.g = static_cast<uint8_t>(value * 255.0);
      pixelColor.b = static_cast<uint8_t>(value * 255.0);

      // Set the pixel in the bitmap
      bitmap.SetPixel(x, y, pixelColor);
    }
  }
  return true;
};

inline Eigen::MatrixXf matrixFromTiff(const std::string& path) {
  // Initialise FreeImage
  FreeImage_Initialise();

  // Load the image
  FIBITMAP* image = FreeImage_Load(FIF_TIFF, path.c_str(), TIFF_DEFAULT);

  if (!image) {
    std::cerr << "Error loading depth map image: " << path << std::endl;
    throw std::runtime_error("Error loading depth map.");
  }

  // Get image information
  int width = FreeImage_GetWidth(image);
  int height = FreeImage_GetHeight(image);
  int bpp = FreeImage_GetBPP(image);
  if (bpp != 32) {
    std::cerr << "Probably not working with not float32 values." << std::endl;
    throw std::runtime_error("Error loading depth map.");
  }

  // Define the matrix
  Eigen::MatrixXf matrix(height, width);

  // Get the raw data pointer
  BYTE* rawData = FreeImage_GetBits(image);

  // Number of bytes per pixel
  int bytesPerPixel = bpp / 8;

  for (int j = 0; j < width; ++j) {
    for (int i = 0; i < height; ++i) {
      size_t offset = i * FreeImage_GetPitch(image) + j * bytesPerPixel;

      // Interpret the data as a 32-bit floating-point number
      float pixelValue;
      memcpy(&pixelValue, &rawData[offset], sizeof(float));

      // Save the values in the matrix with special ordering
      matrix(height - 1 - i, j) = pixelValue;
    }
  }
  FreeImage_Unload(image);
  FreeImage_DeInitialise();

  return matrix;
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_MATRIX_VIS_H_
