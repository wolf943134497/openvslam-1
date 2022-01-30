//
// Created by tanjunyao7 on 2022/1/28.
//

#ifndef OPENVSLAM_POSITION_PRIOR_EDGE_H
#define OPENVSLAM_POSITION_PRIOR_EDGE_H

#include "openvslam/type.h"
#include "openvslam/optimize/internal/se3/shot_vertex.h"
#include <g2o/core/base_unary_edge.h>

namespace openvslam {
namespace optimize {
namespace internal {
namespace se3 {


class pose_fix_xyzyaw_edge final : public g2o::BaseUnaryEdge<4, Vec6_t , shot_vertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    pose_fix_xyzyaw_edge(): g2o::BaseUnaryEdge<4, Vec6_t, shot_vertex>() {};

    bool read(std::istream& is) override {};

    bool write(std::ostream& os) const override {};

    void computeError() override{
        const auto shot_vertex = static_cast<const openvslam::optimize::internal::se3::shot_vertex*>(_vertices[0]);
        _error = shot_vertex->estimate().log().tail<4>();
    };

//    void linearizeOplus() override{
//        _jacobianOplusXi=-Mat66_t::Identity().bottomRows(4);
//    };
};


} // namespace se3
} // namespace internal
} // namespace optimize
} // namespace openvslam

#endif //OPENVSLAM_POSITION_PRIOR_EDGE_H
