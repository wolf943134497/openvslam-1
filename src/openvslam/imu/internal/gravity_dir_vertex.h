#ifndef OPENVSLAM_IMU_INTERNAL_GRAVITY_DIR_VERTEX_H
#define OPENVSLAM_IMU_INTERNAL_GRAVITY_DIR_VERTEX_H

#include "openvslam/type.h"
#include "openvslam/util/converter.h"

#include <g2o/core/base_vertex.h>

namespace openvslam {
namespace imu {
namespace internal {

class gravity_dir_vertex final : public g2o::BaseVertex<2, Sophus::SO3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    gravity_dir_vertex();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void setToOriginImpl() override;

    void oplusImpl(const number_t* update_) override;
};

inline gravity_dir_vertex::gravity_dir_vertex()
    : g2o::BaseVertex<2, Sophus::SO3d>() {}

inline bool gravity_dir_vertex::read(std::istream& is) {
    return false;
}

inline bool gravity_dir_vertex::write(std::ostream& os) const {
    return false;
}

inline void gravity_dir_vertex::setToOriginImpl() {
    setEstimate(Sophus::SO3d());
}

inline void gravity_dir_vertex::oplusImpl(const number_t* update_) {
    setEstimate(estimate() * Sophus::SO3d::exp({update_[0], update_[1], 0.0}));
}

} // namespace internal
} // namespace imu
} // namespace openvslam

#endif // OPENVSLAM_IMU_INTERNAL_GRAVITY_DIR_VERTEX_H
