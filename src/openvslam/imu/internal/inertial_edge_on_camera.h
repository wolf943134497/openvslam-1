#ifndef OPENVSLAM_IMU_INTERNAL_INERTIAL_EDGE_ON_CAMERA_H
#define OPENVSLAM_IMU_INTERNAL_INERTIAL_EDGE_ON_CAMERA_H

#include "openvslam/type.h"
#include "openvslam/util/converter.h"
#include "openvslam/imu/bias.h"
#include "openvslam/imu/constant.h"
#include "openvslam/optimize/internal/se3/shot_vertex.h"
#include "openvslam/imu/internal/velocity_vertex.h"
#include "openvslam/imu/internal/bias_vertex.h"
#include "openvslam/imu/config.h"

#include <g2o/core/base_multi_edge.h>

namespace openvslam {
namespace imu {
namespace internal {

class inertial_edge_on_camera final : public g2o::BaseMultiEdge<9, std::shared_ptr<preintegrated>> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    inertial_edge_on_camera();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() override;

    void linearizeOplus() override;

    Vec3_t gravity;
};

inline inertial_edge_on_camera::inertial_edge_on_camera()
    : g2o::BaseMultiEdge<9, std::shared_ptr<preintegrated>>() {
    resize(6);
    gravity << 0, 0, -imu::constant::gravity();
}

inline bool inertial_edge_on_camera::read(std::istream& is) {
    (void)is;
    return false;
}

inline bool inertial_edge_on_camera::write(std::ostream& os) const {
    (void)os;
    return false;
}

inline void inertial_edge_on_camera::computeError() {
    const auto keyfrm_vtx1 = static_cast<const optimize::internal::se3::shot_vertex*>(_vertices[0]);
    const auto velocity_vtx1 = static_cast<const velocity_vertex*>(_vertices[1]);
    const auto gyr_bias_vtx = static_cast<const bias_vertex*>(_vertices[2]);
    const auto acc_bias_vtx = static_cast<const bias_vertex*>(_vertices[3]);
    const auto keyfrm_vtx2 = static_cast<const optimize::internal::se3::shot_vertex*>(_vertices[4]);
    const auto velocity_vtx2 = static_cast<const velocity_vertex*>(_vertices[5]);

    const bias b(acc_bias_vtx->estimate(), gyr_bias_vtx->estimate());
    const Mat33_t delta_rotation = _measurement->get_delta_rotation_on_bias(b);
    const Vec3_t delta_velocity = _measurement->get_delta_velocity_on_bias(b);
    const Vec3_t delta_position = _measurement->get_delta_position_on_bias(b);
    const double dt = _measurement->dt_;

    const Mat33_t Rcw1 = keyfrm_vtx1->estimate().rotation().toRotationMatrix();
    const Mat33_t Riw1 = imu::config::get_rel_rot_ic() * Rcw1;
    const Mat33_t Rwi1 = Riw1.transpose();
    const Vec3_t tcw1 = keyfrm_vtx1->estimate().translation();
    const Vec3_t twi1 = -Rwi1 * (imu::config::get_rel_rot_ic() * tcw1 + imu::config::get_rel_trans_ic());
    const Mat33_t Rcw2 = keyfrm_vtx2->estimate().rotation().toRotationMatrix();
    const Mat33_t Riw2 = imu::config::get_rel_rot_ic() * Rcw2;
    const Mat33_t Rwi2 = Riw2.transpose();
    const Vec3_t tcw2 = keyfrm_vtx2->estimate().translation();
    const Vec3_t twi2 = -Rwi2 * (imu::config::get_rel_rot_ic() * tcw2 + imu::config::get_rel_trans_ic());

    const Vec3_t v1 = velocity_vtx1->estimate();
    const Vec3_t v2 = velocity_vtx2->estimate();

    const Vec3_t error_rotation = util::converter::log_so3(delta_rotation.transpose() * Riw1 * Rwi2);
    const Vec3_t error_velocity = Riw1 * (v2 - v1 - gravity * dt) - delta_velocity;
    const Vec3_t error_position = Riw1 * (twi2 - twi1 - v1 * dt - 0.5 * gravity * dt * dt) - delta_position;

    _error << error_rotation, error_velocity, error_position;
}

inline void inertial_edge_on_camera::linearizeOplus() {
    const auto keyfrm_vtx1 = static_cast<const optimize::internal::se3::shot_vertex*>(_vertices[0]);
    const auto velocity_vtx1 = static_cast<const velocity_vertex*>(_vertices[1]);
    const auto gyr_bias_vtx = static_cast<const bias_vertex*>(_vertices[2]);
    const auto acc_bias_vtx = static_cast<const bias_vertex*>(_vertices[3]);
    const auto keyfrm_vtx2 = static_cast<const optimize::internal::se3::shot_vertex*>(_vertices[4]);
    const auto velocity_vtx2 = static_cast<const velocity_vertex*>(_vertices[5]);

    const imu::bias b(acc_bias_vtx->estimate(), gyr_bias_vtx->estimate());
    const imu::bias& b0 = _measurement->b_;
    const Vec3_t delta_bias_gyr = b.gyr_ - b0.gyr_;

    const Mat33_t jacob_rotation_gyr = _measurement->jacob_rotation_gyr_;
    const Mat33_t jacob_velocity_gyr = _measurement->jacob_velocity_gyr_;
    const Mat33_t jacob_position_gyr = _measurement->jacob_position_gyr_;
    const Mat33_t jacob_velocity_acc = _measurement->jacob_velocity_acc_;
    const Mat33_t jacob_position_acc = _measurement->jacob_position_acc_;

    //----------------nav state 1
    Mat44_t Tcw1 = keyfrm_vtx1->estimate().to_homogeneous_matrix();
    //jacobian of right pertubation to left pertubation
    Mat66_t J_cw1r_cw1l = util::converter::adjoint(util::converter::inv(Tcw1));

    Mat44_t Twi1 = util::converter::inv(Tcw1)*imu::config::get_rel_pose_ci();
    //jacobian of right pertubation on Twi1 to right pertubation on Tcw1
    Mat66_t J_Twi1_Tcw1_r = -util::converter::adjoint(util::converter::inv(Twi1));

    Mat33_t Rwi1 = Twi1.topLeftCorner<3,3>();
    Vec3_t twi1 = Twi1.topRightCorner<3,1>();
    Vec3_t v1 = velocity_vtx1->estimate();

    //----------------nav state 2
    Mat44_t Tcw2 = keyfrm_vtx2->estimate().to_homogeneous_matrix();
    //jacobian of right pertubation to left pertubation
    Mat66_t J_cw2r_cw2l = util::converter::adjoint(util::converter::inv(Tcw2));

    Mat44_t Twi2 = util::converter::inv(Tcw2)*imu::config::get_rel_pose_ci();
    //jacobian of right pertubation on Twi2 to right pertubation on Tcw2
    Mat66_t J_Twi2_Tcw2_r = -util::converter::adjoint(util::converter::inv(Twi2));

    Mat33_t Rwi2 = Twi2.topLeftCorner<3,3>();
    Vec3_t twi2 = Twi2.topRightCorner<3,1>();
    const Vec3_t v2 = velocity_vtx2->estimate();

    const Mat33_t delta_rotation = _measurement->get_delta_rotation_on_bias(b);
    const Mat33_t error_rotation = delta_rotation.transpose() * Rwi1.transpose() * Rwi2;
    const Mat33_t inv_right_jacobian = util::converter::inverse_right_jacobian_so3(util::converter::log_so3(error_rotation));
    const double dt = _measurement->dt_;


    Mat33_t J_dR_Rwi1 = -inv_right_jacobian * Rwi2.transpose() * Rwi1;
    Mat33_t J_dV_Rwi1 = util::converter::to_skew_symmetric_mat(Rwi1.transpose() * (v2 - v1 - gravity * dt));
    Mat33_t J_dP_Rwi1 = util::converter::to_skew_symmetric_mat(Rwi1.transpose() * (twi2 - twi1 - v1 * dt - 0.5 * gravity * dt * dt));
    Mat33_t J_dP_twi1 = -Eigen::Matrix3d::Identity();
    Mat33_t Zero3x3;    Zero3x3.setZero();


    Eigen::Matrix<double,9,6> J_Twi1;
    J_Twi1<<Zero3x3,J_dR_Rwi1,
            Zero3x3,J_dV_Rwi1,
            J_dP_twi1,J_dP_Rwi1;

    //chain rule
    Eigen::Matrix<double,9,6> J_Tcw1_l = J_Twi1*J_Twi1_Tcw1_r*J_cw1r_cw1l;
    

    Mat33_t J_dR_Rwi2 = inv_right_jacobian;
    Mat33_t J_dP_twi2 = Rwi1.transpose() * Rwi2;
    Eigen::Matrix<double,9,6> J_Twi2;
    J_Twi2<<Zero3x3,J_dR_Rwi2,
        Zero3x3,Zero3x3,
        J_dP_twi2,Zero3x3;
    
    //chain rule
    Eigen::Matrix<double,9,6> J_Tcw2_l = J_Twi2*J_Twi2_Tcw2_r*J_cw2r_cw2l;

    // Jacobians wrt Pose 1
    _jacobianOplus[0].setZero();
    //swap translation and rotation part
    _jacobianOplus[0].leftCols(3) = J_Tcw1_l.rightCols(3);
    _jacobianOplus[0].rightCols(3) = J_Tcw1_l.leftCols(3);


    // Jacobians wrt Velocity 1
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3, 3>(3, 0) = -Rwi1.transpose();
    _jacobianOplus[1].block<3, 3>(6, 0) = -Rwi1.transpose() * dt;

    // Jacobians wrt Gyro 1
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3, 3>(0, 0) = -inv_right_jacobian * error_rotation.transpose()
                                          * util::converter::right_jacobian_so3(jacob_rotation_gyr * delta_bias_gyr) * jacob_rotation_gyr;
    _jacobianOplus[2].block<3, 3>(3, 0) = -jacob_velocity_gyr;
    _jacobianOplus[2].block<3, 3>(6, 0) = -jacob_position_gyr;

    // Jacobians wrt Accelerometer 1
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3, 3>(3, 0) = -jacob_velocity_acc;
    _jacobianOplus[3].block<3, 3>(6, 0) = -jacob_position_acc;

    // Jacobians wrt Pose 2
    _jacobianOplus[4].setZero();
    //swap translation and rotation part
    _jacobianOplus[4].leftCols(3) = J_Tcw2_l.rightCols(3);
    _jacobianOplus[4].rightCols(3) = J_Tcw2_l.leftCols(3);

    // Jacobians wrt Velocity 2
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3, 3>(3, 0) = Rwi1.transpose();
}

} // namespace internal
} // namespace imu
} // namespace openvslam

#endif // OPENVSLAM_IMU_INTERNAL_INERTIAL_EDGE_ON_CAMERA_H
