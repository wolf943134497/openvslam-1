#include "openvslam/data/landmark.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/map_database.h"
#include "openvslam/publish/map_publisher.h"

#include <spdlog/spdlog.h>

namespace openvslam {
namespace publish {

map_publisher::map_publisher(const std::shared_ptr<config>& cfg, data::map_database* map_db)
    : cfg_(cfg), map_db_(map_db) {
    spdlog::debug("CONSTRUCT: publish::map_publisher");
}

map_publisher::~map_publisher() {
    spdlog::debug("DESTRUCT: publish::map_publisher");
}

void map_publisher::set_current_cam_pose(const Mat44_t& cam_pose_cw) {
    std::lock_guard<std::mutex> lock(mtx_cam_pose_);
    cam_pose_cw_ = cam_pose_cw;
}

Mat44_t map_publisher::get_current_cam_pose() {
    std::lock_guard<std::mutex> lock(mtx_cam_pose_);
    return cam_pose_cw_;
}

void map_publisher::set_current_cam_pose_wc(const Mat44_t& cam_pose_wc) {
    std::lock_guard<std::mutex> lock(mtx_cam_pose_);
    cam_pose_wc_ = cam_pose_wc;
}

Mat44_t map_publisher::get_current_cam_pose_wc() {
    std::lock_guard<std::mutex> lock(mtx_cam_pose_);
    return cam_pose_wc_;
}

unsigned int map_publisher::get_keyframes(std::vector<data::keyframe*>& all_keyfrms) {
    all_keyfrms = map_db_->get_all_keyframes();
    return map_db_->get_num_keyframes();
}

unsigned int map_publisher::get_landmarks(std::vector<data::landmark*>& all_landmarks,
                                          std::set<data::landmark*>& local_landmarks) {
    all_landmarks = map_db_->get_all_landmarks();
    const auto _local_landmarks = map_db_->get_local_landmarks();
    local_landmarks = std::set<data::landmark*>(_local_landmarks.begin(), _local_landmarks.end());
    return map_db_->get_num_landmarks();
}

void map_publisher::clear_cam_pose_predicts() {
    std::lock_guard<std::mutex> lock(mtx_cam_pose_pred);
    cam_pose_wc_predicts_.clear();
}

void map_publisher::add_cam_pose_predicts(const Mat44_t& cam_pose_cw_pred) {
    Mat44_t T_wc = Mat44_t::Identity();
    T_wc.topLeftCorner<3,3>() = cam_pose_cw_pred.topLeftCorner<3,3>().transpose();
    T_wc.topRightCorner<3,1>() = -T_wc.topLeftCorner<3,3>()*cam_pose_cw_pred.topRightCorner<3,1>();
    std::lock_guard<std::mutex> lock(mtx_cam_pose_pred);
    cam_pose_wc_predicts_.push_back(T_wc);
}

std::vector<Mat44_t> map_publisher::get_cam_pose_predicts() {
    std::lock_guard<std::mutex> lock(mtx_cam_pose_pred);
    return cam_pose_wc_predicts_;
}

void map_publisher::add_cam_poses(const Mat44_t& cam_pose_wc) {
    std::lock_guard<std::mutex> lock(mtx_cam_poses_);
    cam_poses_wc_.push_back(cam_pose_wc);
}

void map_publisher::clear_cam_poses() {
    std::lock_guard<std::mutex> lock(mtx_cam_poses_);
    cam_poses_wc_.clear();
}

std::vector<Mat44_t> map_publisher::get_cam_poses() {
    std::lock_guard<std::mutex> lock(mtx_cam_poses_);
    return cam_poses_wc_;
}

} // namespace publish
} // namespace openvslam
