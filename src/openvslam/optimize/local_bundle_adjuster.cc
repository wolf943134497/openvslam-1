#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/optimize/local_bundle_adjuster.h"
#include "openvslam/optimize/internal/landmark_vertex_container.h"
#include "openvslam/optimize/internal/se3/shot_vertex_container.h"
#include "openvslam/optimize/internal/se3/reproj_edge_wrapper.h"
#include "openvslam/util/converter.h"
#include "openvslam/imu/internal/bias_edge_wrapper.h"
#include "openvslam/imu/internal/inertial_edge_on_camera.h"
#include "openvslam/imu/internal//inertial_edge_wrapper.h"
#include "openvslam/imu/internal/velocity_vertex_container.h"
#include "openvslam/imu/internal/bias_vertex_container.h"
#include "openvslam/imu/preintegrator.h"


#include <unordered_map>

#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

namespace openvslam {
namespace optimize {

local_bundle_adjuster::local_bundle_adjuster(const unsigned int num_first_iter,
                                             const unsigned int num_second_iter)
    : num_first_iter_(num_first_iter), num_second_iter_(num_second_iter) {}

void local_bundle_adjuster::set_enable_inertial_optimization(bool v) {
    enable_inertial_optimization_ = v;
}

void local_bundle_adjuster::optimize(openvslam::data::keyframe* curr_keyfrm, bool* const force_stop_flag) const {
    // 1. Aggregate the local and fixed keyframes, and local landmarks

    // Correct the local keyframes of the current keyframe
    std::unordered_map<unsigned int, data::keyframe*> local_keyfrms;

    auto curr_covisibilities = curr_keyfrm->graph_node_->get_covisibilities();
    curr_covisibilities.push_back(curr_keyfrm);

    for (auto local_keyfrm : curr_covisibilities) {
        if (!local_keyfrm) {
            continue;
        }
        if (local_keyfrm->will_be_erased()) {
            continue;
        }

        local_keyfrms[local_keyfrm->id_] = local_keyfrm;
    }

    // Correct landmarks seen in local keyframes
    std::unordered_map<unsigned int, data::landmark*> local_lms;

    for (auto local_keyfrm : local_keyfrms) {
        const auto landmarks = local_keyfrm.second->get_landmarks();
        for (auto local_lm : landmarks) {
            if (!local_lm) {
                continue;
            }
            if (local_lm->will_be_erased()) {
                continue;
            }

            // Avoid duplication
            if (local_lms.count(local_lm->id_)) {
                continue;
            }

            local_lms[local_lm->id_] = local_lm;
        }
    }

    // Fixed keyframes: keyframes which observe local landmarks but which are NOT in local keyframes
    std::unordered_map<unsigned int, data::keyframe*> fixed_keyfrms;

    for (auto local_lm : local_lms) {
        const auto observations = local_lm.second->get_observations();
        for (auto& obs : observations) {
            auto fixed_keyfrm = obs.first;
            if (!fixed_keyfrm) {
                continue;
            }
            if (fixed_keyfrm->will_be_erased()) {
                continue;
            }

            // Do not add if it's in the local keyframes
            if (local_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            // Avoid duplication
            if (fixed_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            fixed_keyfrms[fixed_keyfrm->id_] = fixed_keyfrm;
        }
    }

    // 2. Construct an optimizer

    std::unique_ptr<g2o::Solver> block_solver;
    if (enable_inertial_optimization_ && imu::config::is_tightly_coupled()) {
        auto linear_solver = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>>();
        block_solver = g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver));
    }
    else {
        auto linear_solver = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>>();
        block_solver = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
    }
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);


    if (force_stop_flag) {
        optimizer.setForceStopFlag(force_stop_flag);
    }

    // 3. Convert each of the keyframe to the g2o vertex, then set it to the optimizer

    // Container of the shot vertices
    auto vtx_id_offset = std::make_shared<unsigned int>(0);
    int N = local_keyfrms.size() + fixed_keyfrms.size();
    internal::se3::shot_vertex_container keyfrm_vtx_container(vtx_id_offset, N);
    imu::internal::velocity_vertex_container velocity_vtx_container(vtx_id_offset, N);
    imu::internal::bias_vertex_container acc_bias_vtx_container(vtx_id_offset,imu::internal::bias_vertex::Type::ACC, N);
    imu::internal::bias_vertex_container gyr_bias_vtx_container(vtx_id_offset,imu::internal::bias_vertex::Type::Gyr, N);

    auto add_keyfrms_to_optimizer = [&](const std::unordered_map<unsigned int, data::keyframe*>& kfrms, bool fixed)
    {
      for (auto& id_keyfrm_pair : kfrms) {
          auto keyfrm = id_keyfrm_pair.second;

          auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(keyfrm, fixed || keyfrm->id_ == 0);
          optimizer.addVertex(keyfrm_vtx);
          if(enable_inertial_optimization_ && imu::config::is_tightly_coupled())
          {
              auto velocity_vtx = velocity_vtx_container.create_vertex(keyfrm, false);
              optimizer.addVertex(velocity_vtx);
              auto bias = keyfrm->get_bias();
              auto acc_bias_vtx = acc_bias_vtx_container.create_vertex(keyfrm->id_, bias.acc_, false);
              optimizer.addVertex(acc_bias_vtx);
              auto gyr_bias_vtx = gyr_bias_vtx_container.create_vertex(keyfrm->id_, bias.gyr_, false);
              optimizer.addVertex(gyr_bias_vtx);
          }
      }
    };

    add_keyfrms_to_optimizer(local_keyfrms, false);
    add_keyfrms_to_optimizer(fixed_keyfrms,true);


    // 4. Connect the vertices of the keyframe and the landmark by using an edge of reprojection constraint

    // Container of the landmark vertices
    internal::landmark_vertex_container lm_vtx_container(vtx_id_offset, local_lms.size());

    // Container of the reprojection edges
    using reproj_edge_wrapper = internal::se3::reproj_edge_wrapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> reproj_edge_wraps;
    reproj_edge_wraps.reserve(N * local_lms.size());

    // Chi-squared value with significance level of 5%
    // Two degree-of-freedom (n=2)
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // Three degree-of-freedom (n=3)
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    for (auto& id_local_lm_pair : local_lms) {
        auto local_lm = id_local_lm_pair.second;

        // Convert the landmark to the g2o vertex, then set to the optimizer
        auto lm_vtx = lm_vtx_container.create_vertex(local_lm, false);
        optimizer.addVertex(lm_vtx);

        const auto observations = local_lm->get_observations();
        for (const auto& obs : observations) {
            auto keyfrm = obs.first;
            auto idx = obs.second;
            if (!keyfrm) {
                continue;
            }
            if (keyfrm->will_be_erased()) {
                continue;
            }

            const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
            const auto& undist_keypt = keyfrm->undist_keypts_.at(idx);
            const float x_right = keyfrm->stereo_x_right_.at(idx);
            const float inv_sigma_sq = keyfrm->inv_level_sigma_sq_.at(undist_keypt.octave);
            const auto sqrt_chi_sq = (keyfrm->camera_->setup_type_ == camera::setup_type_t::Monocular)
                                     ? sqrt_chi_sq_2D
                                     : sqrt_chi_sq_3D;
            auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, local_lm, lm_vtx,
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
            reproj_edge_wraps.push_back(reproj_edge_wrap);
            optimizer.addEdge(reproj_edge_wrap.edge_);
        }
    }

    if(enable_inertial_optimization_ && imu::config::is_tightly_coupled())
    {
        //add inertial vertices

        for(auto id_kfr: local_keyfrms)
        {
            auto kfr = id_kfr.second;
            auto ref_kfr = kfr->inertial_ref_keyfrm_;
            if(!ref_kfr)
                continue;
            if(local_keyfrms.count(ref_kfr->id_) || fixed_keyfrms.count(ref_kfr->id_))
            {
                auto acc_bias_vtx1 = acc_bias_vtx_container.get_vertex(ref_kfr);
                auto gyr_bias_vtx1 = gyr_bias_vtx_container.get_vertex(ref_kfr);
                auto acc_bias_vtx2 = acc_bias_vtx_container.get_vertex(kfr);
                auto gyr_bias_vtx2 = gyr_bias_vtx_container.get_vertex(kfr);

                auto keyfrm_vtx1 = keyfrm_vtx_container.get_vertex(ref_kfr);
                auto velocity_vtx1 = velocity_vtx_container.get_vertex(ref_kfr);
                auto keyfrm_vtx2 = keyfrm_vtx_container.get_vertex(kfr);
                auto velocity_vtx2 = velocity_vtx_container.get_vertex(kfr);

                const bool use_huber_kernel_inertial = true;
                // Chi-squared value with significance level of 5%
                // 9 degree-of-freedom (n=9)
                constexpr float chi_sq = 16.92;
                const float sqrt_chi_sq = std::sqrt(chi_sq);
                imu::internal::inertial_edge_wrapper inertial_edge_wrap(kfr->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_,
                                                                        acc_bias_vtx1, gyr_bias_vtx1,
                                                                        keyfrm_vtx1, velocity_vtx1, keyfrm_vtx2, velocity_vtx2,
                                                                        sqrt_chi_sq, use_huber_kernel_inertial);

                optimizer.addEdge(inertial_edge_wrap.edge_);


                imu::internal::bias_edge_wrapper acc_bias_edge_wrap(kfr->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_,
                                                                    acc_bias_vtx1,acc_bias_vtx2,
                                                                    sqrt_chi_sq, use_huber_kernel_inertial);
                optimizer.addEdge(acc_bias_edge_wrap.edge_);

                imu::internal::bias_edge_wrapper gyr_bias_edge_wrap(kfr->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_,
                                                                    gyr_bias_vtx1,gyr_bias_vtx2,
                                                                    sqrt_chi_sq, use_huber_kernel_inertial);
                optimizer.addEdge(gyr_bias_edge_wrap.edge_);
            }

        }
    }
    // 5. Perform the first optimization

    if (force_stop_flag) {
        if (*force_stop_flag) {
            return;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(num_first_iter_);

    // 6. Discard outliers, then perform the second optimization

    bool run_robust_BA = true;

    if (force_stop_flag) {
        if (*force_stop_flag) {
            run_robust_BA = false;
        }
    }

    if (run_robust_BA) {
        for (auto& reproj_edge_wrap : reproj_edge_wraps) {
            auto edge = reproj_edge_wrap.edge_;

            auto local_lm = reproj_edge_wrap.lm_;
            if (local_lm->will_be_erased()) {
                continue;
            }

            if (reproj_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }

            edge->setRobustKernel(nullptr);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(num_second_iter_);
    }

    // 7. Count the outliers

    std::vector<std::pair<data::keyframe*, data::landmark*>> outlier_observations;
    outlier_observations.reserve(reproj_edge_wraps.size());

    for (auto& reproj_edge_wrap : reproj_edge_wraps) {
        auto edge = reproj_edge_wrap.edge_;

        auto local_lm = reproj_edge_wrap.lm_;
        if (local_lm->will_be_erased()) {
            continue;
        }

        if (reproj_edge_wrap.is_monocular_) {
            if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
        else {
            if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
    }

    // 8. Update the information

    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        for (auto& outlier_obs : outlier_observations) {
            auto keyfrm = outlier_obs.first;
            auto lm = outlier_obs.second;
            keyfrm->erase_landmark(lm);
            lm->erase_observation(keyfrm);
        }

        for (auto id_local_keyfrm_pair : local_keyfrms) {
            auto local_keyfrm = id_local_keyfrm_pair.second;
            auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(local_keyfrm);
            local_keyfrm->set_cam_pose(keyfrm_vtx->estimate());
            if(enable_inertial_optimization_ && imu::config::is_tightly_coupled())
            {
                auto velocity_vtx = velocity_vtx_container.get_vertex(local_keyfrm);
                local_keyfrm->set_velocity(velocity_vtx->estimate());
                if(local_keyfrm->inertial_ref_keyfrm_)
                    std::cout<<"kfr id: "<<local_keyfrm->id_<<" ref kfm id:"<<local_keyfrm->inertial_ref_keyfrm_->id_<<" velocity: "<<velocity_vtx->estimate().transpose()<<std::endl;
                else
                    std::cout<<"kfr id: "<<local_keyfrm->id_<<" velocity: "<<velocity_vtx->estimate().transpose()<<std::endl;

                auto acc_bias_vtx = acc_bias_vtx_container.get_vertex(local_keyfrm);
                auto gyr_bias_vtx = gyr_bias_vtx_container.get_vertex(local_keyfrm);
                local_keyfrm->set_bias({acc_bias_vtx->estimate(),gyr_bias_vtx->estimate()});
            }
        }

        for (auto id_local_lm_pair : local_lms) {
            auto local_lm = id_local_lm_pair.second;

            auto lm_vtx = lm_vtx_container.get_vertex(local_lm);
            local_lm->set_pos_in_world(lm_vtx->estimate());
            local_lm->update_normal_and_depth();
        }
    }

    //if imu states not jointly optimized, do inertial only optimization
    if(enable_inertial_optimization_ && !imu::config::is_tightly_coupled())
        optimize_imu(curr_covisibilities,force_stop_flag);
}

void local_bundle_adjuster::optimize_imu(std::vector<data::keyframe*> local_keyfrms, bool* const force_stop_flag) const {

    // 1. Construct an optimizer
    std::unique_ptr<g2o::Solver> block_solver;

    auto linear_solver = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>>();
    block_solver = g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver));

    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);


    if (force_stop_flag) {
        optimizer.setForceStopFlag(force_stop_flag);
    }

    // 2. Convert each of the keyframe to the g2o vertex, then set it to the optimizer

    auto vtx_id_offset = std::make_shared<unsigned int>(0);
    int N = local_keyfrms.size();
    internal::se3::shot_vertex_container keyfrm_vtx_container(vtx_id_offset, N);
    imu::internal::velocity_vertex_container velocity_vtx_container(vtx_id_offset, N);
    imu::internal::bias_vertex_container acc_bias_vtx_container(vtx_id_offset,imu::internal::bias_vertex::Type::ACC, N);
    imu::internal::bias_vertex_container gyr_bias_vtx_container(vtx_id_offset,imu::internal::bias_vertex::Type::Gyr, N);

    for (auto keyfrm : local_keyfrms) {

        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(keyfrm, true);
        optimizer.addVertex(keyfrm_vtx);

        auto velocity_vtx = velocity_vtx_container.create_vertex(keyfrm, false);
        optimizer.addVertex(velocity_vtx);
        auto bias = keyfrm->get_bias();
        auto acc_bias_vtx = acc_bias_vtx_container.create_vertex(keyfrm->id_, bias.acc_, false);
        optimizer.addVertex(acc_bias_vtx);
        auto gyr_bias_vtx = gyr_bias_vtx_container.create_vertex(keyfrm->id_, bias.gyr_, false);
        optimizer.addVertex(gyr_bias_vtx);
    }


    //add inertial vertices

    for(auto kfr: local_keyfrms)
    {
        auto ref_kfr = kfr->inertial_ref_keyfrm_;
        if(!ref_kfr)
            continue;
        if(std::find(local_keyfrms.begin(),local_keyfrms.end(),ref_kfr)
            ==local_keyfrms.end())
            continue;

        auto acc_bias_vtx1 = acc_bias_vtx_container.get_vertex(ref_kfr);
        auto gyr_bias_vtx1 = gyr_bias_vtx_container.get_vertex(ref_kfr);
        auto acc_bias_vtx2 = acc_bias_vtx_container.get_vertex(kfr);
        auto gyr_bias_vtx2 = gyr_bias_vtx_container.get_vertex(kfr);

        auto keyfrm_vtx1 = keyfrm_vtx_container.get_vertex(ref_kfr);
        auto velocity_vtx1 = velocity_vtx_container.get_vertex(ref_kfr);
        auto keyfrm_vtx2 = keyfrm_vtx_container.get_vertex(kfr);
        auto velocity_vtx2 = velocity_vtx_container.get_vertex(kfr);

        const bool use_huber_kernel_inertial = true;
        // Chi-squared value with significance level of 5%
        // 9 degree-of-freedom (n=9)
        constexpr float chi_sq = 16.92;
        const float sqrt_chi_sq = std::sqrt(chi_sq);
        imu::internal::inertial_edge_wrapper inertial_edge_wrap(kfr->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_,
                                                                acc_bias_vtx1, gyr_bias_vtx1,
                                                                keyfrm_vtx1, velocity_vtx1, keyfrm_vtx2, velocity_vtx2,
                                                                sqrt_chi_sq, use_huber_kernel_inertial);

        optimizer.addEdge(inertial_edge_wrap.edge_);


        imu::internal::bias_edge_wrapper acc_bias_edge_wrap(kfr->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_,
                                                            acc_bias_vtx1,acc_bias_vtx2,
                                                            sqrt_chi_sq, use_huber_kernel_inertial);
        optimizer.addEdge(acc_bias_edge_wrap.edge_);

        imu::internal::bias_edge_wrapper gyr_bias_edge_wrap(kfr->imu_preintegrator_from_inertial_ref_keyfrm_->preintegrated_,
                                                            gyr_bias_vtx1,gyr_bias_vtx2,
                                                            sqrt_chi_sq, use_huber_kernel_inertial);
        optimizer.addEdge(gyr_bias_edge_wrap.edge_);


    }


    if (force_stop_flag) {
        if (*force_stop_flag) {
            return;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(num_second_iter_);



    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        for (auto local_keyfrm : local_keyfrms) {
            auto velocity_vtx = velocity_vtx_container.get_vertex(local_keyfrm);
            local_keyfrm->set_velocity(velocity_vtx->estimate());

            auto acc_bias = acc_bias_vtx_container.get_vertex(local_keyfrm)->estimate();
            auto gyr_bias = gyr_bias_vtx_container.get_vertex(local_keyfrm)->estimate();
            local_keyfrm->set_bias({acc_bias,gyr_bias});
        }
    }
}

} // namespace optimize
} // namespace openvslam
