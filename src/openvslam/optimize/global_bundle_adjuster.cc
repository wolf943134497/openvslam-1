#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/optimize/global_bundle_adjuster.h"
#include "openvslam/optimize/internal/landmark_vertex_container.h"
#include "openvslam/optimize/internal/se3/shot_vertex_container.h"
#include "openvslam/optimize/internal/se3/reproj_edge_wrapper.h"
#include "openvslam/optimize/internal/se3/position_prior_edge.h"
#include "openvslam/util/converter.h"
#include "openvslam/imu/internal/velocity_vertex_container.h"
#include "openvslam/imu/internal/bias_vertex_container.h"
#include "openvslam/imu/internal/bias_edge_wrapper.h"
#include "openvslam/imu/internal/inertial_edge_wrapper.h"
#include "openvslam/imu/internal/prior_bias_edge_wrapper.h"
#include "openvslam/imu/preintegrator.h"

#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>

namespace openvslam {
namespace optimize {

global_bundle_adjuster::global_bundle_adjuster(data::map_database* map_db, const unsigned int num_iter, const bool use_huber_kernel)
    : map_db_(map_db), num_iter_(num_iter), use_huber_kernel_(use_huber_kernel) {}

void global_bundle_adjuster::optimize(const unsigned int lead_keyfrm_id_in_global_BA, bool* const force_stop_flag, double info_prior_acc, double info_prior_gyr) const {
    // 1. Collect the dataset

    const auto keyfrms = map_db_->get_all_keyframes();
    const auto lms = map_db_->get_all_landmarks();
    std::vector<bool> is_optimized_lm(lms.size(), true);

    // 2. Construct an optimizer

    std::unique_ptr<g2o::BlockSolverBase> block_solver;
    g2o::OptimizationAlgorithmWithHessian* algorithm;
    if (enable_inertial_optimization_) {
        auto linear_solver = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>>();
        block_solver = g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver));
    }
    else {
        auto linear_solver = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>>();
        block_solver = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
    }
    algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    if (force_stop_flag) {
        optimizer.setForceStopFlag(force_stop_flag);
    }

    int n_inertial_vertices = 0;
    int n_visual_vertices = 0;
    // 3. Convert each of the keyframe to the g2o vertex, then set it to the optimizer

    // Container of the shot vertices
    auto vtx_id_offset = std::make_shared<unsigned int>(0);
    internal::se3::shot_vertex_container keyfrm_vtx_container(vtx_id_offset, keyfrms.size());
    imu::internal::velocity_vertex_container velocity_vtx_container(vtx_id_offset, keyfrms.size());
    imu::internal::bias_vertex_container acc_bias_vtx_container(vtx_id_offset,imu::internal::bias_vertex::Type::ACC, keyfrms.size());
    imu::internal::bias_vertex_container gyr_bias_vtx_container(vtx_id_offset,imu::internal::bias_vertex::Type::Gyr, keyfrms.size());

    // Set the keyframes to the optimizer
    for (const auto keyfrm : keyfrms) {
        if (!keyfrm) {
            continue;
        }
        if (keyfrm->will_be_erased()) {
            continue;
        }

        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(keyfrm, false);
        optimizer.addVertex(keyfrm_vtx);


        if(keyfrm->id_==0)
        {
            auto edge = new openvslam::optimize::internal::se3::pose_fix_xyzyaw_edge();
            edge->setVertex(0,keyfrm_vtx);
            edge->setInformation(Mat44_t::Identity());
            optimizer.addEdge(edge);
        }

        if (enable_inertial_optimization_) {
            auto velocity_vtx = velocity_vtx_container.create_vertex(keyfrm, false);
            optimizer.addVertex(velocity_vtx);
        }

        if (enable_inertial_optimization_ && !use_shared_bias_) {
            auto bias = keyfrm->get_bias();
            auto acc_bias_vtx = acc_bias_vtx_container.create_vertex(keyfrm->id_, bias.acc_, false);
            optimizer.addVertex(acc_bias_vtx);
            auto gyr_bias_vtx = gyr_bias_vtx_container.create_vertex(keyfrm->id_, bias.gyr_, false);
            optimizer.addVertex(gyr_bias_vtx);

            // Add prior to shared bias
            imu::internal::prior_bias_edge_wrapper pba_edge_wrap(info_prior_acc, acc_bias_vtx);
            optimizer.addEdge(pba_edge_wrap.edge_);
            imu::internal::prior_bias_edge_wrapper pbg_edge_wrap(info_prior_gyr, gyr_bias_vtx);
            optimizer.addEdge(pbg_edge_wrap.edge_);
            n_inertial_vertices ++;
        }
    }

    if (enable_inertial_optimization_ && use_shared_bias_) {
        auto bias = keyfrms.back()->get_bias();
        auto acc_bias_vtx = acc_bias_vtx_container.create_vertex(keyfrms.back()->id_, bias.acc_, false);
        optimizer.addVertex(acc_bias_vtx);
        auto gyr_bias_vtx = gyr_bias_vtx_container.create_vertex(keyfrms.back()->id_, bias.gyr_, false);
        optimizer.addVertex(gyr_bias_vtx);


        // Add prior to shared bias
        imu::internal::prior_bias_edge_wrapper pba_edge_wrap(info_prior_acc, acc_bias_vtx);
        optimizer.addEdge(pba_edge_wrap.edge_);
        imu::internal::prior_bias_edge_wrapper pbg_edge_wrap(info_prior_gyr, gyr_bias_vtx);
        optimizer.addEdge(pbg_edge_wrap.edge_);
        n_inertial_vertices +=2;
    }

    // 4.1 Connect the vertices of the keyframe and the imu data

    if (enable_inertial_optimization_) {
        for (size_t i = 0; i < keyfrms.size(); i++) {
            auto keyfrm = keyfrms.at(i);
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
            auto ref_keyfrm = keyfrm->inertial_ref_keyfrm_;
            assert(fabs(keyfrm->timestamp_-ref_keyfrm->timestamp_
                        -keyfrm->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_->dt_)<0.01);

            imu::internal::bias_vertex* acc_bias_vtx1;
            imu::internal::bias_vertex* gyr_bias_vtx1;
            imu::internal::bias_vertex* acc_bias_vtx2;
            imu::internal::bias_vertex* gyr_bias_vtx2;
            if (use_shared_bias_) {
                acc_bias_vtx1 = acc_bias_vtx_container.get_vertex(keyfrms.back());
                gyr_bias_vtx1 = gyr_bias_vtx_container.get_vertex(keyfrms.back());
            }
            else {
                acc_bias_vtx1 = acc_bias_vtx_container.get_vertex(ref_keyfrm);
                gyr_bias_vtx1 = gyr_bias_vtx_container.get_vertex(ref_keyfrm);
                acc_bias_vtx2 = acc_bias_vtx_container.get_vertex(keyfrm);
                gyr_bias_vtx2 = gyr_bias_vtx_container.get_vertex(keyfrm);
            }
            auto keyfrm_vtx1 = keyfrm_vtx_container.get_vertex(ref_keyfrm);
            auto velocity_vtx1 = velocity_vtx_container.get_vertex(ref_keyfrm);
            auto keyfrm_vtx2 = keyfrm_vtx_container.get_vertex(keyfrm);
            auto velocity_vtx2 = velocity_vtx_container.get_vertex(keyfrm);

            const bool use_huber_kernel_inertial = true;
            // Chi-squared value with significance level of 5%
            // 9 degree-of-freedom (n=9)
            constexpr float chi_sq = 16.92;
            const float sqrt_chi_sq = std::sqrt(chi_sq);
            imu::internal::inertial_edge_wrapper inertial_edge_wrap(keyfrm->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_,
                                                                    acc_bias_vtx1, gyr_bias_vtx1,
                                                                    keyfrm_vtx1, velocity_vtx1, keyfrm_vtx2, velocity_vtx2,
                                                                    sqrt_chi_sq, use_huber_kernel_inertial);

            optimizer.addEdge(inertial_edge_wrap.edge_);
            n_inertial_vertices ++;

            if (!use_shared_bias_) {
                imu::internal::bias_edge_wrapper acc_bias_edge_wrap(keyfrm->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_,
                                                               acc_bias_vtx1,acc_bias_vtx2,
                                                               sqrt_chi_sq, use_huber_kernel_inertial);
                optimizer.addEdge(acc_bias_edge_wrap.edge_);

                imu::internal::bias_edge_wrapper gyr_bias_edge_wrap(keyfrm->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_,
                                                               gyr_bias_vtx1,gyr_bias_vtx2,
                                                               sqrt_chi_sq, use_huber_kernel_inertial);
                optimizer.addEdge(gyr_bias_edge_wrap.edge_);
                n_inertial_vertices += 2;
            }
        }
    }

    // 4.2 Connect the vertices of the keyframe and the landmark by using reprojection edge

    // Container of the landmark vertices
    internal::landmark_vertex_container lm_vtx_container(vtx_id_offset, lms.size());

    // Container of the reprojection edges
    using reproj_edge_wrapper = internal::se3::reproj_edge_wrapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> reproj_edge_wraps;
    reproj_edge_wraps.reserve(10 * lms.size());

    // Chi-squared value with significance level of 5%
    // Two degree-of-freedom (n=2)
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // Three degree-of-freedom (n=3)
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    for (unsigned int i = 0; i < lms.size(); ++i) {
        auto lm = lms.at(i);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        // Convert the landmark to the g2o vertex, then set it to the optimizer
        auto lm_vtx = lm_vtx_container.create_vertex(lm, false);
        optimizer.addVertex(lm_vtx);

        unsigned int num_edges = 0;
        const auto observations = lm->get_observations();
        for (const auto& obs : observations) {
            auto keyfrm = obs.first;
            auto idx = obs.second;
            if (!keyfrm) {
                continue;
            }
            if (keyfrm->will_be_erased()) {
                continue;
            }

            if (!keyfrm_vtx_container.contain(keyfrm)) {
                continue;
            }

            const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
            const auto& undist_keypt = keyfrm->undist_keypts_.at(idx);
            const float x_right = keyfrm->stereo_x_right_.at(idx);
            const float inv_sigma_sq = keyfrm->inv_level_sigma_sq_.at(undist_keypt.octave);
            const auto sqrt_chi_sq = (keyfrm->camera_->setup_type_ == camera::setup_type_t::Monocular)
                                         ? sqrt_chi_sq_2D
                                         : sqrt_chi_sq_3D;
            auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, lm, lm_vtx,
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq, use_huber_kernel_);
            reproj_edge_wraps.push_back(reproj_edge_wrap);
            optimizer.addEdge(reproj_edge_wrap.edge_);
            n_visual_vertices ++;
            ++num_edges;
        }

        if (num_edges == 0) {
            optimizer.removeVertex(lm_vtx);
            is_optimized_lm.at(i) = false;
        }
    }




    // 5. Perform optimization

    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(num_iter_);

    if (force_stop_flag && *force_stop_flag) {
        return;
    }

    // 6. Extract the result
    {
        std::unique_lock<std::mutex> lock(data::map_database::mtx_database_);
        for (auto keyfrm : keyfrms) {
            if (keyfrm->will_be_erased()) {
                continue;
            }
            auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
            const auto cam_pose_cw = util::converter::to_eigen_mat(keyfrm_vtx->estimate());
            if (lead_keyfrm_id_in_global_BA == 0) {
                keyfrm->set_cam_pose(cam_pose_cw);
            }
            else {
                keyfrm->cam_pose_cw_after_loop_BA_ = cam_pose_cw;
                keyfrm->loop_BA_identifier_ = lead_keyfrm_id_in_global_BA;
            }

            if (enable_inertial_optimization_) {
                if (keyfrm->inertial_ref_keyfrm_) {
                    auto velocity_vtx = static_cast<imu::internal::velocity_vertex*>(velocity_vtx_container.get_vertex(keyfrm));
                    keyfrm->set_velocity( velocity_vtx->estimate());
                    if (use_shared_bias_) {
                        auto gyr_bias_vtx = static_cast<imu::internal::bias_vertex*>(gyr_bias_vtx_container.get_vertex(keyfrms.back()));
                        auto acc_bias_vtx = static_cast<imu::internal::bias_vertex*>(acc_bias_vtx_container.get_vertex(keyfrms.back()));
                        keyfrm->set_bias( {acc_bias_vtx->estimate(), gyr_bias_vtx->estimate()});
                    }
                    else {
                        auto gyr_bias_vtx = static_cast<imu::internal::bias_vertex*>(gyr_bias_vtx_container.get_vertex(keyfrm));
                        auto acc_bias_vtx = static_cast<imu::internal::bias_vertex*>(acc_bias_vtx_container.get_vertex(keyfrm));
                        keyfrm->set_bias( {acc_bias_vtx->estimate(), gyr_bias_vtx->estimate()});                }
                }
            }
        }

        for (unsigned int i = 0; i < lms.size(); ++i) {
            if (!is_optimized_lm.at(i)) {
                continue;
            }

            auto lm = lms.at(i);
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            auto lm_vtx = lm_vtx_container.get_vertex(lm);
            const Vec3_t pos_w = lm_vtx->estimate();

            if (lead_keyfrm_id_in_global_BA == 0) {
                lm->set_pos_in_world(pos_w);
                lm->update_normal_and_depth();
            }
            else {
                lm->pos_w_after_global_BA_ = pos_w;
                lm->loop_BA_identifier_ = lead_keyfrm_id_in_global_BA;
            }
        }
    }


}

void global_bundle_adjuster::enable_inertial_optimization(bool enabled, bool use_shared_bias) {
    enable_inertial_optimization_ = enabled;
    use_shared_bias_ = use_shared_bias;
}

} // namespace optimize
} // namespace openvslam
