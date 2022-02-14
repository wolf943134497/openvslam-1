#ifndef OPENVSLAM_IMU_INTERNAL_GRAVITY_SCALE_EDGE_ON_CAMERA_H
#define OPENVSLAM_IMU_INTERNAL_GRAVITY_SCALE_EDGE_ON_CAMERA_H

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

class inertial_gravity_scale_edge_on_camera final : public g2o::BaseMultiEdge<9, std::shared_ptr<preintegrated>> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    inertial_gravity_scale_edge_on_camera();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() override;

    void linearizeOplus() override;

};

inline inertial_gravity_scale_edge_on_camera::inertial_gravity_scale_edge_on_camera()
    : g2o::BaseMultiEdge<9, std::shared_ptr<preintegrated>>() {
    resize(8);
}

inline bool inertial_gravity_scale_edge_on_camera::read(std::istream& is) {
    double dt;
    MatRC_t<15, 15> covariance;
    Vec3_t acc;
    Vec3_t gyr;
    Mat33_t delta_rotation;
    Vec3_t delta_velocity;
    Vec3_t delta_position;
    Mat33_t jacob_rotation_gyr;
    Mat33_t jacob_velocity_gyr;
    Mat33_t jacob_velocity_acc;
    Mat33_t jacob_position_gyr;
    Mat33_t jacob_position_acc;
    is >> dt;
    read_matrix(is, covariance);
    read_matrix(is, acc);
    read_matrix(is, gyr);
    read_matrix(is, delta_rotation);
    read_matrix(is, delta_velocity);
    read_matrix(is, delta_position);
    read_matrix(is, jacob_rotation_gyr);
    read_matrix(is, jacob_velocity_gyr);
    read_matrix(is, jacob_velocity_acc);
    read_matrix(is, jacob_position_gyr);
    read_matrix(is, jacob_position_acc);
    _measurement = eigen_alloc_shared<preintegrated>(dt, covariance, bias(acc, gyr), delta_rotation, delta_velocity, delta_position,
                                                     jacob_rotation_gyr, jacob_velocity_gyr, jacob_velocity_acc, jacob_position_gyr, jacob_position_acc);
    for (unsigned int i = 0; i < Dimension; ++i) {
        for (unsigned int j = i; j < Dimension; ++j) {
            is >> information()(i, j);
            if (i != j) {
                information()(j, i) = information()(i, j);
            }
        }
    }
    return true;
}

inline bool inertial_gravity_scale_edge_on_camera::write(std::ostream& os) const {
    os << _measurement->dt_ << " ";
    write_matrix(os, _measurement->covariance_);
    write_matrix(os, _measurement->b_.acc_);
    write_matrix(os, _measurement->b_.gyr_);
    write_matrix(os, _measurement->delta_rotation_);
    write_matrix(os, _measurement->delta_velocity_);
    write_matrix(os, _measurement->delta_position_);
    write_matrix(os, _measurement->jacob_rotation_gyr_);
    write_matrix(os, _measurement->jacob_velocity_gyr_);
    write_matrix(os, _measurement->jacob_velocity_acc_);
    write_matrix(os, _measurement->jacob_position_gyr_);
    write_matrix(os, _measurement->jacob_position_acc_);
    for (unsigned int i = 0; i < Dimension; ++i) {
        for (unsigned int j = i; j < Dimension; ++j) {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

inline void inertial_gravity_scale_edge_on_camera::computeError() {
    const auto keyfrm_vtx1 = static_cast<const optimize::internal::se3::shot_vertex*>(_vertices[0]);
    const auto velocity_vtx1 = static_cast<const velocity_vertex*>(_vertices[1]);
    const auto gyr_bias_vtx = static_cast<const bias_vertex*>(_vertices[2]);
    const auto acc_bias_vtx = static_cast<const bias_vertex*>(_vertices[3]);
    const auto keyfrm_vtx2 = static_cast<const optimize::internal::se3::shot_vertex*>(_vertices[4]);
    const auto velocity_vtx2 = static_cast<const velocity_vertex*>(_vertices[5]);
    const auto gravity_dir_vtx = static_cast<const gravity_dir_vertex*>(_vertices[6]);
    const auto scale_vtx = static_cast<const scale_vertex*>(_vertices[7]);

    const bias b(acc_bias_vtx->estimate(), gyr_bias_vtx->estimate());
    const Sophus::SO3d delta_rotation(_measurement->get_delta_rotation_on_bias(b));
    const Vec3_t delta_velocity = _measurement->get_delta_velocity_on_bias(b);
    const Vec3_t delta_position = _measurement->get_delta_position_on_bias(b);
    const double dt = _measurement->dt_;

    Sophus::SE3d Twg(gravity_dir_vtx->estimate(),{0,0,0});
    Sophus::SE3d Tci(imu::config::get_rel_pose_ci());
    
    //----------------nav state 1
    Sophus::SE3d Tcw1 = Sophus::SE3d(keyfrm_vtx1->estimate().to_homogeneous_matrix());
    Sophus::SE3d Twc1 = Tcw1.inverse();
    Sophus::SE3d Twc1_aligned = Twg.inverse()*Twc1;
    Sophus::SE3d Twc1_aligned_scaled = Twc1_aligned;
    Twc1_aligned_scaled.translation() *= scale_vtx->estimate();
    Sophus::SE3d Tgi1 = Twc1_aligned_scaled*Tci;
    Sophus::SO3d Rgi1 = Tgi1.so3();
    Vec3_t tgi1 = Tgi1.translation();
    Vec3_t v1 = velocity_vtx1->estimate();

    //----------------nav state 2
    Sophus::SE3d Tcw2 = Sophus::SE3d(keyfrm_vtx2->estimate().to_homogeneous_matrix());
    Sophus::SE3d Twc2 = Tcw2.inverse();
    Sophus::SE3d Twc2_aligned = Twg.inverse()*Twc2;
    Sophus::SE3d Twc2_aligned_scaled = Twc2_aligned;
    Twc2_aligned_scaled.translation() *= scale_vtx->estimate();
    Sophus::SE3d Tgi2 = Twc2_aligned_scaled*Tci;
    Sophus::SO3d Rgi2 = Tgi2.rotationMatrix();
    Vec3_t tgi2 = Tgi2.translation();
    Vec3_t v2 = velocity_vtx2->estimate();

    Vec3_t g(0,0,-9.81);
    const Vec3_t error_rotation = (delta_rotation.inverse() * Rgi1.inverse() * Rgi2).log();        // (7)
    const Vec3_t error_velocity = Rgi1.inverse() * (v2 - v1 - g * dt) - delta_velocity;                          // (8)
    const Vec3_t error_position = Rgi1.inverse() * (tgi2 - tgi1 - v1 * dt - 0.5 * g * dt * dt) - delta_position; // (9)

    _error << error_rotation, error_velocity, error_position;

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

inline void inertial_gravity_scale_edge_on_camera::linearizeOplus() {
    const auto keyfrm_vtx1 = static_cast<const optimize::internal::se3::shot_vertex*>(_vertices[0]);
    const auto velocity_vtx1 = static_cast<const velocity_vertex*>(_vertices[1]);
    const auto gyr_bias_vtx = static_cast<const bias_vertex*>(_vertices[2]);
    const auto acc_bias_vtx = static_cast<const bias_vertex*>(_vertices[3]);
    const auto keyfrm_vtx2 = static_cast<const optimize::internal::se3::shot_vertex*>(_vertices[4]);
    const auto velocity_vtx2 = static_cast<const velocity_vertex*>(_vertices[5]);
    const auto gravity_dir_vtx = static_cast<const gravity_dir_vertex*>(_vertices[6]);
    const auto scale_vtx = static_cast<const scale_vertex*>(_vertices[7]);


    Sophus::SE3d Twg(gravity_dir_vtx->estimate(),{0,0,0});
    Sophus::SE3d Tci(imu::config::get_rel_pose_ci());

    //----------------nav state 1---------------
    Sophus::SE3d Tcw1 = Sophus::SE3d(keyfrm_vtx1->estimate().to_homogeneous_matrix());
    //jacobian of right pertubation to left pertubation
    Mat66_t J_Tcw1r_Tcw1l = Tcw1.inverse().Adj();

    Sophus::SE3d Twc1 = Tcw1.inverse();
    //jacobian of right pertubation of Twc1 to Tcw1
    Mat66_t J_Twc1r_Tcw1r = -Tcw1.Adj();

    Sophus::SE3d Twc1_aligned = Twg.inverse()*Twc1;
    //jacobian of right pertubation of Twc1_aligned to Twc1
    Mat66_t J_Twc1_aligned_Twc1r = -Twc1_aligned.inverse().Adj();
    //jacobian tof right pertubation of Twc1_aligned to Twg
    Mat66_t J_Twc1_aligned_Twg = Mat66_t::Identity()-Twc1_aligned.inverse().Adj();

    Sophus::SE3d Twc1_aligned_scaled = Twc1_aligned;
    Twc1_aligned_scaled.translation() *= scale_vtx->estimate();
    //jacobian of right pertubation of Twc1_aligned_scaled to Twc1_aligned
    Mat66_t J_Twc1_aligned_scaled_Twc1_aligned = Mat66_t::Identity();
    J_Twc1_aligned_scaled_Twc1_aligned.topLeftCorner<3,3>() *= scale_vtx->estimate();
    //jacobian of right pertubation of Twc1_aligned_scaled to scale
    Vec6_t J_Twc1_aligned_scaled_s;
    J_Twc1_aligned_scaled_s<<Twc1_aligned.translation(),0,0,0;

    Sophus::SE3d Tgi1 = Twc1_aligned_scaled*Tci;
    //jacobian of right pertubation of Tgi1 to Twc1_aligned_scaled
    Mat66_t J_Tgi1_J_Twc1_aligned_scaled = util::converter::adjoint(imu::config::get_rel_pose_ic());

    // chain rule conclusion
    Mat66_t J_Tgi1_Tcw1 = J_Tgi1_J_Twc1_aligned_scaled
                          *J_Twc1_aligned_scaled_Twc1_aligned
                          *J_Twc1_aligned_Twc1r
                          *J_Twc1r_Tcw1r
                          *J_Tcw1r_Tcw1l;
    Mat63_t J_Tgi1_Rwg =  J_Tgi1_J_Twc1_aligned_scaled
                         *J_Twc1_aligned_scaled_Twc1_aligned
                         *J_Twc1_aligned_Twg.rightCols<3>();
    Vec6_t J_Tgi1_s = J_Tgi1_J_Twc1_aligned_scaled
                    * J_Twc1_aligned_scaled_s;

    Sophus::SO3d Rgi1 = Tgi1.so3();
    Vec3_t tgi1 = Tgi1.translation();
    Vec3_t v1 = velocity_vtx1->estimate();

    //----------------nav state 2---------------
    Sophus::SE3d Tcw2 = Sophus::SE3d(keyfrm_vtx2->estimate().to_homogeneous_matrix());
    //jacobian of right pertubation to left pertubation
    Mat66_t J_Tcw2r_Tcw2l = Tcw2.inverse().Adj();

    Sophus::SE3d Twc2 = Tcw2.inverse();
    //jacobian of right pertubation of Twc2 to Tcw2
    Mat66_t J_Twc2r_Tcw2r = -Tcw2.Adj();

    Sophus::SE3d Twc2_aligned = Twg.inverse()*Twc2;
    //jacobian of right pertubation of Twc2_aligned to Twc2
    Mat66_t J_Twc2_aligned_Twc2r = -Twc2_aligned.inverse().Adj();
    //jacobian tof right pertubation of Twc2_aligned to Twg
    Mat66_t J_Twc2_aligned_Twg = Mat66_t::Identity()-Twc2_aligned.inverse().Adj();

    Sophus::SE3d Twc2_aligned_scaled = Twc2_aligned;
    Twc2_aligned_scaled.translation() *= scale_vtx->estimate();
    //jacobian of right pertubation of Twc2_aligned_scaled to Twc2_aligned
    Mat66_t J_Twc2_aligned_scaled_Twc2_aligned = Mat66_t::Identity();
    J_Twc2_aligned_scaled_Twc2_aligned.topLeftCorner<3,3>() *= scale_vtx->estimate();
    //jacobian of right pertubation of Twc2_aligned_scaled to scale
    Vec6_t J_Twc2_aligned_scaled_s;
    J_Twc2_aligned_scaled_s<<Twc2_aligned.translation(),0,0,0;

    Sophus::SE3d Tgi2 = Twc2_aligned_scaled*Tci;
    //jacobian of right pertubation of Tgi2 to Twc2_aligned_scaled
    Mat66_t J_Tgi2_J_Twc2_aligned_scaled = util::converter::adjoint(imu::config::get_rel_pose_ic());

    // chain rule conclusion
    Mat66_t J_Tgi2_Tcw2 = J_Tgi2_J_Twc2_aligned_scaled
                          *J_Twc2_aligned_scaled_Twc2_aligned
                          *J_Twc2_aligned_Twc2r
                          *J_Twc2r_Tcw2r
                          *J_Tcw2r_Tcw2l;
    Mat63_t J_Tgi2_Rwg =  J_Tgi2_J_Twc2_aligned_scaled
                          *J_Twc2_aligned_scaled_Twc2_aligned
                          *J_Twc2_aligned_Twg.rightCols<3>();
    Vec6_t J_Tgi2_s = J_Tgi2_J_Twc2_aligned_scaled
                      * J_Twc2_aligned_scaled_s;

    Sophus::SO3d Rgi2 = Tgi2.rotationMatrix();
    Vec3_t tgi2 = Tgi2.translation();
    Vec3_t v2 = velocity_vtx2->estimate();



    const imu::bias b(acc_bias_vtx->estimate(), gyr_bias_vtx->estimate());
    const imu::bias& b0 = _measurement->b_;
    const Vec3_t delta_bias_gyr = b.gyr_ - b0.gyr_;

    const Mat33_t jacob_rotation_gyr = _measurement->jacob_rotation_gyr_;
    const Mat33_t jacob_velocity_gyr = _measurement->jacob_velocity_gyr_;
    const Mat33_t jacob_position_gyr = _measurement->jacob_position_gyr_;
    const Mat33_t jacob_velocity_acc = _measurement->jacob_velocity_acc_;
    const Mat33_t jacob_position_acc = _measurement->jacob_position_acc_;


    const Sophus::SO3d delta_rotation(_measurement->get_delta_rotation_on_bias(b));
    const Sophus::SO3d error_rotation = delta_rotation.inverse() * Rgi1.inverse() * Rgi2;
    const Mat33_t inv_right_jacobian = util::converter::inverse_right_jacobian_so3(error_rotation.log());
    const double dt = _measurement->dt_;

    Vec3_t g(0,0,-9.81);
    Mat33_t J_dR_Rgi1 = -inv_right_jacobian * (Rgi2.inverse() * Rgi1).matrix();
    Mat33_t J_dV_Rgi1 = Sophus::SO3d::hat(Rgi1.inverse() * (v2 - v1 - g * dt));
    Mat33_t J_dP_Rgi1 = Sophus::SO3d::hat(Rgi1.inverse() * (tgi2 - tgi1 - v1 * dt - 0.5 * g * dt * dt));
    Mat33_t J_dP_tgi1 = -Eigen::Matrix3d::Identity();
    Mat33_t Zero3x3;    Zero3x3.setZero();
    Eigen::Matrix<double,9,6> J_Tgi1;
    J_Tgi1<<Zero3x3,J_dR_Rgi1,
        Zero3x3,J_dV_Rgi1,
        J_dP_tgi1,J_dP_Rgi1;

    decltype(J_Tgi1) J_Tcw1 = J_Tgi1*J_Tgi1_Tcw1;

    Mat33_t J_dR_Rgi2 = inv_right_jacobian;
    Mat33_t J_dP_tgi2 = (Rgi1.inverse() * Rgi2).matrix();
    Eigen::Matrix<double,9,6> J_Tgi2;
    J_Tgi2<<Zero3x3,J_dR_Rgi2,
        Zero3x3,Zero3x3,
        J_dP_tgi2,Zero3x3;

    decltype(J_Tgi2) J_Tcw2 = J_Tgi2*J_Tgi2_Tcw2;

    Eigen::Matrix<double,9,3> J_Rwg = J_Tgi1*J_Tgi1_Rwg
                                    + J_Tgi2*J_Tgi2_Rwg;

    Eigen::Matrix<double,9,1> J_s = J_Tgi1*J_Tgi1_s
                                    + J_Tgi2*J_Tgi2_s;

    // Jacobians wrt Pose 1
    _jacobianOplus[0].setZero();
    //swap translation and rotation part
    _jacobianOplus[0].leftCols(3) = J_Tcw1.rightCols(3);
    _jacobianOplus[0].rightCols(3) = J_Tcw1.leftCols(3);

    // Jacobians wrt Velocity 1
    _jacobianOplus[1].setZero();
        _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3, 3>(3, 0) = -Rgi1.inverse().matrix();
    _jacobianOplus[1].block<3, 3>(6, 0) = -Rgi1.inverse().matrix() * dt;

    // Jacobians wrt Gyro 1
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3, 3>(0, 0) = -inv_right_jacobian * error_rotation.inverse().matrix()
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
    _jacobianOplus[4].leftCols(3) = J_Tcw2.rightCols(3);
    _jacobianOplus[4].rightCols(3) = J_Tcw2.leftCols(3);

    // Jacobians wrt Velocity 2
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3, 3>(3, 0) = Rgi1.inverse().matrix();

    // Jacobians wrt Gravity direction
    _jacobianOplus[6] = J_Rwg.leftCols<2>();

    // Jacobians wrt scale factor
    _jacobianOplus[7] = J_s;
}

} // namespace internal
} // namespace imu
} // namespace openvslam

#endif // OPENVSLAM_IMU_INTERNAL_GRAVITY_SCALE_EDGE_ON_CAMERA_H
