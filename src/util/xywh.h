#ifndef COLMAP_SRC_UTIL_XYWH_H_
#define COLMAP_SRC_UTIL_XYWH_H_

#include <Eigen/Core>
#include <vector>

namespace colmap {

class XYWH {
  // Bounding box class
  // XYWH(0,0,2,2) means 2x2 boundings box with pixels (0,0), (1,0), (0,1),
  // (1,1) Assuming y-axis pointing downwards
 public:
  // Default constructor
  explicit XYWH() : x_(0), y_(0), w_(0), h_(0) {}

  // Copy constructor
  XYWH(const XYWH& xywh) : x_(xywh.x_), y_(xywh.y_), w_(xywh.w_), h_(xywh.h_) {
    std::cout << "XYWH copy !" << std::endl;
  }

  // Destructor
  ~XYWH() { /*std::cout << "XYWH destructor !" << std::endl;*/
  }

  bool Check() {
    if (this->w_ < 0 || this->h_ < 0) {
      std::cerr << "ERROR: can't have a XYWH with negative width or height."
                << std::endl;
      throw std::runtime_error(
          "ERROR: can't have a XYWH with negative width or height.");
    }
  }

  explicit XYWH(const int& x, const int& y, const int& w, const int& h)
      : x_(x), y_(y), w_(w), h_(h) {
    this->Check();
  }

  explicit XYWH(const Eigen::Vector2i& top_left,
                const Eigen::Vector2i& bottom_right) {
    this->setTopLeftBottomRight(top_left, bottom_right);
  }

  void setToZero() {
    this->x_ = 0;
    this->y_ = 0;
    this->w_ = 0;
    this->h_ = 0;
  }
  // Overload the xywh_1 == xywh_2 operator
  bool operator==(const XYWH& other) const {
    return this->x_ == other.x_ && this->y_ == other.y_ &&
           this->w_ == other.w_ && this->h_ == other.h_;
  }

  // Overload xywh == value operator
  bool operator==(const int& value) const { return this->nPixels() == value; }

  // Overload the != operator
  template <typename T>
  bool operator!=(const T& other) const {
    return !(*this == other);
  }

  void setTopLeftBottomRight(const Eigen::Vector2i& top_left,
                             const Eigen::Vector2i& bottom_right) {
    this->x_ = top_left.x();
    this->y_ = top_left.y();
    this->w_ = bottom_right.x() - top_left.x() + 1;
    this->h_ = bottom_right.y() - top_left.y() + 1;
    this->Check();
  }

  template <typename... Args>
  void setToBoundPoints(const Args&... points) {
    // Loop through each vector using a traditional for loop
    std::vector<Eigen::Vector2d> points_vector;
    for (const auto& point : {points...}) {
      points_vector.push_back(point);
    }
    this->setToBoundPoints(points_vector);
  }

  void setToBoundPoints(
      const std::vector<Eigen::Vector2d>&
          points) {  //, const int & max_X, const int & max_Y) {

    double min_x = points[0].x();
    double min_y = points[0].y();
    double max_x = points[0].x();
    double max_y = points[0].y();

    for (const Eigen::Vector2d& point : points) {
      min_x = std::min<double>(min_x, point.x());
      max_x = std::max<double>(max_x, point.x());

      min_y = std::min<double>(min_y, point.y());
      max_y = std::max<double>(max_y, point.y());
    }

    Eigen::Vector2i top_left;
    top_left(0) = static_cast<int>(std::floor(min_x));
    top_left(1) = static_cast<int>(std::floor(min_y));

    Eigen::Vector2i bottom_right;
    bottom_right(0) = static_cast<int>(std::ceil(max_x));
    bottom_right(1) = static_cast<int>(std::ceil(max_y));

    this->setTopLeftBottomRight(top_left, bottom_right);
  }

  void shrinkToFitInToFitIn(const XYWH& xywh) {
    Eigen::Vector2i top_left;
    top_left(0) = std::max<int>(this->x(), xywh.x());
    top_left(1) = std::max<int>(this->y(), xywh.y());

    Eigen::Vector2i bottom_right;
    bottom_right(0) =
        std::min<int>(this->BottomRight().x(), xywh.BottomRight().x());
    bottom_right(1) =
        std::min<int>(this->BottomRight().y(), xywh.BottomRight().y());

    if (bottom_right(0) < top_left(0) || bottom_right(1) < top_left(1)) {
      this->setToZero();
    } else {
      this->setTopLeftBottomRight(top_left, bottom_right);
    }
  }

  // Corners
  Eigen::Vector2i TopLeft() const { return Eigen::Vector2i(x(), y()); }
  Eigen::Vector2i TopRight() const { return Eigen::Vector2i(x_end(), y()); }
  Eigen::Vector2i BottomLeft() const { return Eigen::Vector2i(x(), y_end()); }
  Eigen::Vector2i BottomRight() const {
    return Eigen::Vector2i(x_end(), y_end());
  }

  std::vector<Eigen::Vector2i> Corners() const {
    return {TopLeft(), TopRight(), BottomRight(), BottomLeft()};
  }

  int nPixels() const { return w_ * h_; }

  // Getters
  const int x() const { return x_; };
  const int x_end() const { return x_ + w_ - 1; };
  const int y() const { return y_; };
  const int y_end() const { return y_ + h_ - 1; };
  const int w() const { return w_; };
  const int h() const { return h_; };

  // Print (overload << operator)
  friend std::ostream& operator<<(std::ostream& os, const XYWH& xywh) {
    os << "XYWH(x=" << xywh.x_ << ", y=" << xywh.y_ << ", w=" << xywh.w_
       << ", h=" << xywh.h_ << ")";
    return os;
  }

 private:
  int x_;
  int y_;
  int w_;
  int h_;
};

}  // namespace colmap

#endif  // CCOLMAP_SRC_UTIL_XYWH_H_
