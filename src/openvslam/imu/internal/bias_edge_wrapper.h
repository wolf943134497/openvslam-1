//
// Created by tanjunyao7 on 2022/1/11.
//

#ifndef OPENVSLAM_BIAS_EDGE_WRAPPER_H
#define OPENVSLAM_BIAS_EDGE_WRAPPER_H

#include "openvslam/imu/preintegrated.h"
#include "openvslam/imu/internal/inertial_edge_on_camera.h"

#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/base_binary_edge.h>

namespace openvslam {
namespace imu {
namespace internal {


class bias_edge final: public g2o::BaseBinaryEdge<3,Vec3_t ,bias_vertex,bias_vertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool read(std::istream& is) override{return false;}
    bool write(std::ostream& os) const override{return false;}

    void computeError() override
    {
        auto v1 = dynamic_cast<const bias_vertex*>(_vertices[0]);
        auto v2 = dynamic_cast<const bias_vertex*>(_vertices[1]);
        _error = v1->estimate()-v2->estimate();
    }
    void linearizeOplus() override
    {
        _jacobianOplusXi = Mat33_t::Identity();
        _jacobianOplusXj = - _jacobianOplusXi;
    }
};

class bias_edge_wrapper {
public:
    bias_edge_wrapper() = delete;

    bias_edge_wrapper(const std::shared_ptr<preintegrated> preintegrated,
                      bias_vertex* bias_vtx1, bias_vertex* bias_vtx2);

    virtual ~bias_edge_wrapper() = default;

    bool is_inlier() const;

    bool is_outlier() const;

    void set_as_inlier() const;

    void set_as_outlier() const;

    g2o::OptimizableGraph::Edge* edge_;
};

inline bias_edge_wrapper::bias_edge_wrapper(const std::shared_ptr<preintegrated> preintegrated,
                                            bias_vertex* bias_vtx1, bias_vertex* bias_vtx2) {

    auto edge = new bias_edge();
    assert(bias_vtx1->type==bias_vtx2->type);
    assert(bias_vtx1->type!=bias_vertex::Type::NOT_SET);

    if(bias_vtx1->type==bias_vertex::Type::Gyr)
        edge->setInformation(preintegrated->get_information().block<3, 3>(9, 9));
    else
        edge->setInformation(preintegrated->get_information().block<3, 3>(12, 12));
    edge->setVertex(0, bias_vtx1);
    edge->setVertex(1, bias_vtx2);

    edge_ = edge;
}

inline bool bias_edge_wrapper::is_inlier() const {
    return edge_->level() == 0;
}

inline bool bias_edge_wrapper::is_outlier() const {
    return edge_->level() != 0;
}

inline void bias_edge_wrapper::set_as_inlier() const {
    edge_->setLevel(0);
}

inline void bias_edge_wrapper::set_as_outlier() const {
    edge_->setLevel(1);
}

} // namespace internal
} // namespace imu
} // namespace openvslam

#endif //OPENVSLAM_BIAS_EDGE_WRAPPER_H
