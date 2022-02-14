//
// Created by tanjunyao7 on 2022/2/12.
//

#ifndef OPENVSLAM_INERTIAL_PREDICTION_EDGE_H
#define OPENVSLAM_INERTIAL_PREDICTION_EDGE_H

#include "openvslam/type.h"
#include "openvslam/util/converter.h"
#include "openvslam/imu/bias.h"
#include "openvslam/imu/constant.h"
#include "openvslam/imu/config.h"
#include "openvslam/imu/preintegrated.h"
#include "openvslam/optimize/internal/se3/shot_vertex.h"
#include "openvslam/imu/internal/velocity_vertex.h"
#include "openvslam/imu/internal/bias_vertex.h"
#include "openvslam/imu/internal/gravity_dir_vertex.h"
#include "openvslam/imu/internal/scale_vertex.h"

#include <g2o/core/base_multi_edge.h>

namespace openvslam {
namespace imu {
namespace internal {

class inertial_prediction_edge final : public g2o::BaseUnaryEdge<6, std::shared_ptr<preintegrated>,optimize::internal::se3::shot_vertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    inline inertial_prediction_edge(const Sophus::SE3d Tcw1,
                                    const Vec3_t v1_,
                                    const Sophus::SO3d Rwg,
                                    const double scale_,
                                    const imu::bias& bias_)
        : g2o::BaseUnaryEdge<6, std::shared_ptr<preintegrated>,optimize::internal::se3::shot_vertex>(),
            v1(v1_),scale(scale_), bias(bias_){

        Twg = Sophus::SE3d(Rwg,{0,0,0});

        Sophus::SE3d Twc1 = Tcw1.inverse();
        Sophus::SE3d Twc1_aligned = Twg.inverse()*Twc1;
        Sophus::SE3d Twc1_aligned_scaled = Twc1_aligned;
        Twc1_aligned_scaled.translation() *= scale_;
        Tgi1 = Twc1_aligned_scaled*Tci;
    }

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() override;

    void linearizeOplus() override;

    Sophus::SE3d Tgi1;
    const Vec3_t v1;

    const double scale;
    const imu::bias bias;

    Sophus::SE3d Twg;
    Sophus::SE3d Tci{imu::config::get_rel_pose_ci()};

};



inline bool inertial_prediction_edge::read(std::istream& is) {
    return false;
}

inline bool inertial_prediction_edge::write(std::ostream& os) const {
    return false;
}

inline void inertial_prediction_edge::computeError() {
    const auto frm_vtx2 = static_cast<const optimize::internal::se3::shot_vertex*>(_vertices[0]);

    const Sophus::SO3d delta_rotation(_measurement->get_delta_rotation_on_bias(bias));
    const Vec3_t delta_position = _measurement->get_delta_position_on_bias(bias);
    const double dt = _measurement->dt_;



    //----------------nav state 1
    Sophus::SO3d Rgi1 = Tgi1.so3();
    Vec3_t tgi1 = Tgi1.translation();


    //----------------nav state 2
    Sophus::SE3d Tcw2 = Sophus::SE3d(frm_vtx2->estimate().to_homogeneous_matrix());
    Sophus::SE3d Twc2 = Tcw2.inverse();
    Sophus::SE3d Twc2_aligned = Twg.inverse()*Twc2;
    Sophus::SE3d Twc2_aligned_scaled = Twc2_aligned;
    Twc2_aligned_scaled.translation() *= scale;
    Sophus::SE3d Tgi2 = Twc2_aligned_scaled*Tci;
    Sophus::SO3d Rgi2 = Tgi2.rotationMatrix();
    Vec3_t tgi2 = Tgi2.translation();

    Vec3_t g(0,0,-9.81);
    const Vec3_t error_rotation = (delta_rotation.inverse() * Rgi1.inverse() * Rgi2).log();        // (7)
    const Vec3_t error_position = Rgi1.inverse() * (tgi2 - tgi1 - v1 * dt - 0.5 * g * dt * dt) - delta_position; // (9)

    _error << error_rotation, error_position;
//    std::cout<<"dt: "<<_measurement->dt_<<std::endl;
//    std::cout<<"prediction residual: "<<_error.transpose()<<std::endl;
////    std::cout<<"infomation: "<<information()<<std::endl;
//    std::cout<<"prediction error: "<<this->chi2()<<std::endl;
//    std::cout<<velocity_vtx1->keyframe_id_<<" "<<velocity_vtx2->keyframe_id_<<std::endl;
//    std::cout<<"dt: "<<dt<<std::endl;
//    std::cout<<"tgi1: "<<tgi1.transpose()<<std::endl;
//    std::cout<<"tgi2: "<<tgi2.transpose()<<std::endl;
//    std::cout<<"Rgi1: "<<Rgi1.matrix()<<std::endl;
//    std::cout<<"delta_velocity: "<<delta_velocity.transpose()<<std::endl;
//    std::cout<<"delta_position: "<<delta_position.transpose()<<std::endl;
//    std::cout<<"Rgi1*delta_position: "<<(Rgi1*delta_position).transpose()<<std::endl;
//    std::cout<<"velocities[i]: "<<v1.transpose()<<std::endl;
//    std::cout<<"_error: "<<_error.transpose()<<std::endl;
//    std::cout<<"--------------------"<<std::endl;
}

inline void inertial_prediction_edge::linearizeOplus() {
    const auto frm_vtx2 = static_cast<const optimize::internal::se3::shot_vertex*>(_vertices[0]);


    Sophus::SO3d Rgi1 = Tgi1.so3();
    Vec3_t tgi1 = Tgi1.translation();


    //----------------nav state 2---------------
    Sophus::SE3d Tcw2 = Sophus::SE3d(frm_vtx2->estimate().to_homogeneous_matrix());
    //jacobian of right pertubation to left pertubation
    Mat66_t J_Tcw2r_Tcw2l = Tcw2.inverse().Adj();

    Sophus::SE3d Twc2 = Tcw2.inverse();
    //jacobian of right pertubation of Twc2 to Tcw2
    Mat66_t J_Twc2r_Tcw2r = -Tcw2.Adj();

    Sophus::SE3d Twc2_aligned = Twg.inverse()*Twc2;
    //jacobian of right pertubation of Twc2_aligned to Twc2
    Mat66_t J_Twc2_aligned_Twc2r = -Twc2_aligned.inverse().Adj();

    Sophus::SE3d Twc2_aligned_scaled = Twc2_aligned;
    Twc2_aligned_scaled.translation() *= scale;
    //jacobian of right pertubation of Twc2_aligned_scaled to Twc2_aligned
    Mat66_t J_Twc2_aligned_scaled_Twc2_aligned = Mat66_t::Identity();
    J_Twc2_aligned_scaled_Twc2_aligned.topLeftCorner<3,3>() *= scale;

    Sophus::SE3d Tgi2 = Twc2_aligned_scaled*Tci;
    //jacobian of right pertubation of Tgi2 to Twc2_aligned_scaled
    Mat66_t J_Tgi2_J_Twc2_aligned_scaled = util::converter::adjoint(imu::config::get_rel_pose_ic());

    // chain rule conclusion
    Mat66_t J_Tgi2_Tcw2 = J_Tgi2_J_Twc2_aligned_scaled
                          *J_Twc2_aligned_scaled_Twc2_aligned
                          *J_Twc2_aligned_Twc2r
                          *J_Twc2r_Tcw2r
                          *J_Tcw2r_Tcw2l;


    Sophus::SO3d Rgi2 = Tgi2.rotationMatrix();
    Vec3_t tgi2 = Tgi2.translation();


    const imu::bias& b0 = _measurement->b_;
    const Vec3_t delta_bias_gyr = bias.gyr_ - b0.gyr_;

    const Mat33_t jacob_rotation_gyr = _measurement->jacob_rotation_gyr_;
    const Mat33_t jacob_position_gyr = _measurement->jacob_position_gyr_;
    const Mat33_t jacob_position_acc = _measurement->jacob_position_acc_;


    const Sophus::SO3d delta_rotation(_measurement->get_delta_rotation_on_bias(bias));
    const Sophus::SO3d error_rotation = delta_rotation.inverse() * Rgi1.inverse() * Rgi2;
    const Mat33_t inv_right_jacobian = util::converter::inverse_right_jacobian_so3(error_rotation.log());
    const double dt = _measurement->dt_;

    Mat33_t Zero3x3;    Zero3x3.setZero();

    Mat33_t J_dR_Rgi2 = inv_right_jacobian;
    Mat33_t J_dP_tgi2 = (Rgi1.inverse() * Rgi2).matrix();
    Mat66_t J_Tgi2;
    J_Tgi2<<Zero3x3,J_dR_Rgi2,
        J_dP_tgi2,Zero3x3;

    decltype(J_Tgi2) J_Tcw2 = J_Tgi2*J_Tgi2_Tcw2;


    // Jacobians wrt Pose 2
    _jacobianOplusXi.setZero();
    //swap translation and rotation part
    _jacobianOplusXi.leftCols(3) = J_Tcw2.rightCols(3);
    _jacobianOplusXi.rightCols(3) = J_Tcw2.leftCols(3);

}

} // namespace internal
} // namespace imu
} // namespace openvslam


#endif //OPENVSLAM_INERTIAL_PREDICTION_EDGE_H
