#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/imu/imu_initializer.h"
#include "openvslam/imu/preintegrator.h"
#include "openvslam/imu/preintegrated.h"
#include "openvslam/imu/internal/velocity_vertex_container.h"
#include "openvslam/imu/internal/bias_vertex_container.h"
#include "openvslam/imu/internal/prior_bias_edge_wrapper.h"
#include "openvslam/imu/internal/gravity_dir_vertex.h"
#include "openvslam/imu/internal/scale_vertex.h"
#include "openvslam/optimize/internal/se3/shot_vertex_container.h"

#include <g2o/core/block_solver.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include "sophus/sim3.hpp"
#include "sophus/se3.hpp"

#include <spdlog/spdlog.h>

using namespace Sophus;
namespace openvslam {
namespace optimize {

imu_initializer::imu_initializer(const unsigned int num_iter)
    : num_iter_(num_iter) {}

bool imu_initializer::initialize(const std::vector<data::keyframe*>& keyfrms, Sophus::SO3d & Rwg, double& scale,
                                 bool depth_is_avaliable, float info_prior_acc) const {


    std::vector<data::keyframe*> keyfrms_sorted = keyfrms;
    std::sort(keyfrms_sorted.begin(),keyfrms_sorted.end(),[](data::keyframe* kf1,data::keyframe* kf2){
      return kf1->id_<kf2->id_;
    });

    int iters = 30;
//    printf("1. estimating gyroscope bias\n");
    Vec3_t gyr_bias;gyr_bias.setZero();
    Mat33_t Hg;
    Vec3_t bg;
    double error;
    int count;
    for(int iter=0;iter<iters;iter++)
    {
        error = 0;
        count = 0;
        Hg.setZero();
        bg.setZero();
        for (size_t i = 1; i < keyfrms_sorted.size(); i++) {
            auto keyfrm = keyfrms_sorted.at(i);
            if (!keyfrm) {
                continue;
            }
            if (keyfrm->will_be_erased()) {
                continue;
            }
            if (!keyfrm->inertial_ref_keyfrm_) {
                continue;
            }

            assert(keyfrm->imu_preintegrator_from_inertial_ref_keyfrm_);
            const auto preintegrated = keyfrm->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_;
            auto ref_keyfrm = keyfrm->inertial_ref_keyfrm_;
            assert(ref_keyfrm==keyfrms_sorted[i-1]);

            assert(fabs(ref_keyfrm->timestamp_+ keyfrm->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_->dt_-keyfrm->timestamp_)<0.01);

            const Mat33_t Rcw1 = ref_keyfrm->get_rotation();
            const Mat33_t Riw1 = imu::config::get_rel_rot_ic() * Rcw1;
            const Mat33_t Rcw2 = keyfrm->get_rotation();
            const Mat33_t Riw2 = imu::config::get_rel_rot_ic() * Rcw2;
            const Mat33_t Rwi2 = Riw2.transpose();

            const Mat33_t delta_rotation = preintegrated->get_delta_rotation_on_bias({{0,0,0},gyr_bias});

            const Mat33_t error_rotation = delta_rotation.transpose() * Riw1 * Rwi2;
            Vec3_t r = util::converter::log_so3(error_rotation);
            error += r.squaredNorm();
            count ++;
            //approximate jacobian
            const Mat33_t inv_right_jacobian = util::converter::inverse_right_jacobian_so3(util::converter::log_so3(error_rotation));
            const Mat33_t jacob_rotation_gyr = preintegrated->jacob_rotation_gyr_;

            Mat33_t J = -inv_right_jacobian * error_rotation.transpose()
                        * util::converter::right_jacobian_so3(jacob_rotation_gyr * gyr_bias) * jacob_rotation_gyr;;
            Hg += J.transpose()*J;
            bg += J.transpose()*r;
        }
        Vec3_t bg_increment = - Hg.ldlt().solve(bg);
        double rmse = sqrt(error/count);
//        printf("iter: %d, rmse: %.6f inc: %.3f %.3f %.3f\n",iter,rmse,bg_increment[0],bg_increment[1],bg_increment[2]);
        if(bg_increment.norm()<1e-6)
            break;
        gyr_bias += bg_increment;
    }

//    printf("gyroscope bias: %.6f %.6f %.6f\n",gyr_bias[0],gyr_bias[1],gyr_bias[2]);

//    printf("2. estimating scale, gravity\n");
    scale = 1.0;
    const Vec3_t gI{0,0,-9.81};
    Vec3_t gW=gI;
    int N = keyfrms_sorted.size();
    Eigen::MatrixXd A(3*(N-2),4);
    Eigen::VectorXd B(3*(N-2));
    for(int i=0;i<keyfrms_sorted.size()-2;i++)
    {
        const data::keyframe* f1 = keyfrms_sorted[i];
        const data::keyframe* f2 = keyfrms_sorted[i+1];
        const data::keyframe* f3 = keyfrms_sorted[i+2];
        assert(f2->inertial_ref_keyfrm_ == f1);
        assert(f3->inertial_ref_keyfrm_ == f2);
        const Mat44_t Twc1 = f1->get_cam_pose_inv();
        const Mat33_t Rwc1 = Twc1.topLeftCorner<3,3>();
        const Mat33_t Rwi1 = Rwc1*imu::config::get_rel_rot_ci();
        const Vec3_t twc1 = Twc1.topRightCorner<3,1>();

        const Mat44_t Twc2 = f2->get_cam_pose_inv();
        const Mat33_t Rwc2 = Twc2.topLeftCorner<3,3>();
        const Mat33_t Rwi2 = Rwc2*imu::config::get_rel_rot_ci();
        const Vec3_t twc2 = Twc2.topRightCorner<3,1>();

        const Mat44_t Twc3 = f3->get_cam_pose_inv();
        const Mat33_t Rwc3 = Twc3.topLeftCorner<3,3>();
        const Mat33_t Rwi3 = Rwc3*imu::config::get_rel_rot_ci();
        const Vec3_t twc3 = Twc3.topRightCorner<3,1>();

        const auto preintegrated12 = f2->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_;
        const auto preintegrated23 = f3->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_;
        const double dt12 = preintegrated12->dt_;
        const double dt23 = preintegrated23->dt_;

        const Vec3_t delta_velocity12 = preintegrated12->get_delta_velocity_on_bias({{0,0,0},gyr_bias});
        const Vec3_t delta_position12 = preintegrated12->get_delta_position_on_bias({{0,0,0},gyr_bias});
        const Vec3_t delta_position23 = preintegrated23->get_delta_position_on_bias({{0,0,0},gyr_bias});


        /*
         * |(twc2-twc1)*dt23-(twc3-twc2)*dt12|^T   | scale|
         * |                                 |  *  |      | =  -((Rwc2-Rwc1)*dt23-(Rwc3-Rwc2)*dt12)*imu::config::get_rel_trans_ci()+Rwi1*delta_position12*dt23-Rwi2*delta_position23*dt12
         * |0.5*dt12*dt23(dt12+dt23)         |     | gW   |   - Rwi1*delta_velocity12*dt12*dt23
         */
        A.middleRows(3*i,3)<<(twc2-twc1)*dt23-(twc3-twc2)*dt12,0.5*dt12*dt23*(dt12+dt23)*Mat33_t::Identity();
        B.segment<3>(3*i) = -((Rwc2-Rwc1)*dt23-(Rwc3-Rwc2)*dt12)*imu::config::get_rel_trans_ci()
                            +Rwi1*delta_position12*dt23-Rwi2*delta_position23*dt12
                            -Rwi1*delta_velocity12*dt12*dt23;

    }
    Mat44_t Hsgw = A.transpose()*A;
    Vec4_t bsgw = A.transpose()*B;
//    if(depth_is_avaliable)
//    {
//        //in this case visual odometry has absolute scale, no need to solve scale here
//        //use schur complete
//        Mat33_t Hgw = Hsgw.bottomRightCorner<3,3>()-Hsgw.bottomLeftCorner<3,1>()*
//                        Hsgw.topRightCorner<1,3>()/Hsgw(0,0);
//        Vec3_t bgw = bsgw.tail<3>()-Hsgw.bottomLeftCorner<3,1>()*bsgw(0)/Hsgw(0,0);
//        gW = Hgw.ldlt().solve(bgw);
//    }
//    else
//    {
        Eigen::VectorXd sgW = Hsgw.ldlt().solve(bsgw);
        scale = sgW[0];
        gW = sgW.tail<3>();
//    }

//    printf("initial scale: %.3f, initial gW: %.3f %.3f %.3f \n",scale,gW[0],gW[1],gW[2]);
    std::cout<<"gW-1: "<<gW.transpose()<<std::endl;
    gW = gW.normalized()*9.81;
    std::cout<<"gW0: "<<gW.transpose()<<std::endl;
    Vec3_t rotationAxis = (gI.cross(gW)).normalized();
    double angle = atan2((gI.cross(gW)).norm(),gI.dot(gW));
    Rwg = Sophus::SO3d::exp(rotationAxis*angle);

    std::cout<<"Rwg: "<<Rwg.matrix()<<std::endl;
    Vec3_t acc_bias;acc_bias.setZero();
    std::vector<Vec3_t> velocities(N);
    //compute velocities
    for(int i=0;i<keyfrms_sorted.size()-1;i++)
    {
        data::keyframe* f1 = keyfrms_sorted[i];
        data::keyframe* f2 = keyfrms_sorted[i+1];
        assert(f2->inertial_ref_keyfrm_ == f1);

        const auto preintegrated12 = f2->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_;
        const imu::bias b(acc_bias, gyr_bias);
        const Mat33_t delta_rotation = preintegrated12->get_delta_rotation_on_bias(b);
        const Vec3_t delta_velocity = preintegrated12->get_delta_velocity_on_bias(b);
        const Vec3_t delta_position = preintegrated12->get_delta_position_on_bias(b);
        const double dt = preintegrated12->dt_;

        Mat44_t Twg;
        Twg<<Rwg.matrix(),Vec3_t::Zero(),
            0,0,0,1;

        Mat44_t Tcw1 = f1->get_cam_pose();
        Mat44_t Twc1 = util::converter::inv(Tcw1);
        Mat44_t Twc1_aligned = util::converter::inv(Twg)*Twc1;
        Mat44_t Twc1_aligned_scaled = Twc1_aligned;
        Twc1_aligned_scaled.topRightCorner<3,1>() *= scale;
        Mat44_t Tgi1 = Twc1_aligned_scaled*imu::config::get_rel_pose_ci();
        Mat33_t Rgi1 = Tgi1.topLeftCorner<3,3>();
        Vec3_t tgi1 = Tgi1.topRightCorner<3,1>();


        //----------------nav state 2
        Mat44_t Tcw2 = f2->get_cam_pose();
        Mat44_t Twc2 = util::converter::inv(Tcw2);
        Mat44_t Twc2_aligned = util::converter::inv(Twg)*Twc2;
        Mat44_t Twc2_aligned_scaled = Twc2_aligned;
        Twc2_aligned_scaled.topRightCorner<3,1>() *= scale;
        Mat44_t Tgi2 = Twc2_aligned_scaled*imu::config::get_rel_pose_ci();
        Mat33_t Rgi2 = Tgi2.topLeftCorner<3,3>();
        Vec3_t tgi2 = Tgi2.topRightCorner<3,1>();

        Vec3_t g(0,0,-9.81);

        velocities[i] = (tgi2 - tgi1 - Rgi1*delta_position - 0.5 * g * dt * dt)/dt;
        velocities[i+1] = velocities[i] +g*dt +Rgi1*delta_velocity;

//        std::cout<<f1->id_<<" "<<f2->id_<<std::endl;
//        std::cout<<"dt: "<<dt<<std::endl;
//        std::cout<<"tgi1: "<<tgi1.transpose()<<std::endl;
//        std::cout<<"tgi2: "<<tgi2.transpose()<<std::endl;
//        std::cout<<"Rgi1: "<<Rgi1<<std::endl;
//        std::cout<<"delta_velocity: "<<delta_velocity.transpose()<<std::endl;
//        std::cout<<"delta_position: "<<delta_position.transpose()<<std::endl;
//        std::cout<<"Rgi1*delta_position: "<<(Rgi1*delta_position).transpose()<<std::endl;
//        std::cout<<"velocities[i]: "<<velocities[i].transpose()<<std::endl;
//        std::cout<<"--------------------"<<std::endl;
    }

    printf("scale: %.3f\n",scale);
    std::cout<<"gW: "<<(Rwg*gI).transpose()<<std::endl;

    imu::bias bias(acc_bias,gyr_bias);
    for(int i=0;i<N;i++)
    {
        keyfrms_sorted[i]->set_velocity(velocities[i]);
        keyfrms_sorted[i]->set_bias(bias);
        std::cout<<keyfrms_sorted[i]->id_<<" "<<velocities[i].transpose()<<std::endl;
    }


    return scale>0;
}

} // namespace optimize
} // namespace openvslam