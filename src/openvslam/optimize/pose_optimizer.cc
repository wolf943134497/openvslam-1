#include "openvslam/data/frame.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/imu/preintegrator.h"
#include "openvslam/imu/internal/inertial_prediction_edge.h"
#include "openvslam/imu/bias.h"
#include "openvslam/optimize/pose_optimizer.h"
#include "openvslam/optimize/internal/se3/pose_opt_edge_wrapper.h"
#include "openvslam/util/converter.h"

#include <vector>
#include <mutex>


#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

namespace openvslam {
namespace optimize {

pose_optimizer::pose_optimizer(const unsigned int num_trials, const unsigned int num_each_iter)
    : num_trials_(num_trials), num_each_iter_(num_each_iter) {}

unsigned int pose_optimizer::optimize(data::frame& frm,
                                      data::keyframe* ref_kfm,
                                      imu::preintegrator* preint,
                                      const Vec3_t& frm_velocity,
                                      const Sophus::SO3d& Rwg,
                                      const double scale,
                                      bool verbose) const {
    // 1. Construct an optimizer

    std::unique_ptr<g2o::Solver> block_solver;
    if (ref_kfm && preint) {
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

    unsigned int num_init_obs = 0;

    // 2. Convert the frame to the g2o vertex, then set it to the optimizer

    auto frm_vtx = new internal::se3::shot_vertex();
    frm_vtx->setId(frm.id_);
    frm_vtx->setEstimate(util::converter::to_g2o_SE3(frm.cam_pose_cw_));
    frm_vtx->setFixed(false);
    optimizer.addVertex(frm_vtx);

    if(ref_kfm && preint)
    {

        auto prediction_edge = new
            imu::internal::inertial_prediction_edge(Sophus::SE3d(ref_kfm->get_cam_pose()),
                                                    ref_kfm->get_velocity(),
                                                    Rwg,
                                                    scale,
                                                    ref_kfm->get_bias());
        Eigen::Matrix<double,9,9> Info = preint->preintegrated_->covariance_.block<9,9>(0,0).cast<double>().inverse();
        Info = (Info+Info.transpose())/2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(Info);
        Eigen::Matrix<double,9,1> eigs = es.eigenvalues();
        for(int i=0;i<9;i++)
            if(eigs[i]<1e-12)
                eigs[i]=0;
        Info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
//        prediction_edge->setInformation(Info);

//        MatRC_t<15, 15> info = preint->preintegrated_->get_information();
        Mat66_t info_Rt;
        info_Rt<<Info.block<3,3>(0,0),Info.block<3,3>(0,6),
            Info.block<3,3>(6,0),Info.block<3,3>(6,6);
        prediction_edge->setInformation(info_Rt);

        prediction_edge->setMeasurement(preint->preintegrated_);

        prediction_edge->setVertex(0, frm_vtx);

        optimizer.addEdge(prediction_edge);
    }

    const unsigned int num_keypts = frm.num_keypts_;

    // 3. Connect the landmark vertices by using projection edges

    // Container of the reprojection edges
    using pose_opt_edge_wrapper = internal::se3::pose_opt_edge_wrapper<data::frame>;
    std::vector<pose_opt_edge_wrapper> pose_opt_edge_wraps;
    pose_opt_edge_wraps.reserve(num_keypts);

    // Chi-squared value with significance level of 5%
    // Two degree-of-freedom (n=2)
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // Three degree-of-freedom (n=3)
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    for (unsigned int idx = 0; idx < num_keypts; ++idx) {
        auto lm = frm.landmarks_.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        ++num_init_obs;
        frm.outlier_flags_.at(idx) = false;

        // Connect the frame and the landmark vertices using the projection edges
        const auto& undist_keypt = frm.undist_keypts_.at(idx);
        const float x_right = frm.stereo_x_right_.at(idx);
        const float inv_sigma_sq = frm.inv_level_sigma_sq_.at(undist_keypt.octave);
        const auto sqrt_chi_sq = (frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                     ? sqrt_chi_sq_2D
                                     : sqrt_chi_sq_3D;
        auto pose_opt_edge_wrap = pose_opt_edge_wrapper(&frm, frm_vtx, lm->get_pos_in_world(),
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
        pose_opt_edge_wraps.push_back(pose_opt_edge_wrap);
        optimizer.addEdge(pose_opt_edge_wrap.edge_);
    }

    if (num_init_obs < 5) {
        return 0;
    }

    // 4. Perform robust Bundle Adjustment (BA)

    unsigned int num_bad_obs = 0;
    for (unsigned int trial = 0; trial < num_trials_; ++trial) {
        optimizer.initializeOptimization();
        optimizer.setVerbose(verbose);
        optimizer.optimize(num_each_iter_);

        num_bad_obs = 0;

        for (auto& pose_opt_edge_wrap : pose_opt_edge_wraps) {
            auto edge = pose_opt_edge_wrap.edge_;

            if (frm.outlier_flags_.at(pose_opt_edge_wrap.idx_)) {
                edge->computeError();
            }

            if (pose_opt_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2()) {
                    frm.outlier_flags_.at(pose_opt_edge_wrap.idx_) = true;
                    pose_opt_edge_wrap.set_as_outlier();
                    ++num_bad_obs;
                }
                else {
                    frm.outlier_flags_.at(pose_opt_edge_wrap.idx_) = false;
                    pose_opt_edge_wrap.set_as_inlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2()) {
                    frm.outlier_flags_.at(pose_opt_edge_wrap.idx_) = true;
                    pose_opt_edge_wrap.set_as_outlier();
                    ++num_bad_obs;
                }
                else {
                    frm.outlier_flags_.at(pose_opt_edge_wrap.idx_) = false;
                    pose_opt_edge_wrap.set_as_inlier();
                }
            }

            if (trial == num_trials_ - 2) {
                edge->setRobustKernel(nullptr);
            }
        }

        if (num_init_obs - num_bad_obs < 5) {
            break;
        }
    }

    // 5. Update the information

    frm.set_cam_pose(frm_vtx->estimate());

    return num_init_obs - num_bad_obs;
}

} // namespace optimize
} // namespace openvslam
