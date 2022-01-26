#include "openvslam/imu/config.h"
#include "openvslam/data/common.h"
#include <nlohmann/json.hpp>

namespace openvslam {
namespace imu {

config config::instance_;

void config::fromYaml(const YAML::Node& yaml_node) {
    instance_.name_ = yaml_node["name"].as<std::string>();
    instance_.rate_hz_ = yaml_node["rate_hz"].as<unsigned int>();
    instance_.rate_dt_ = 1.0/instance_.rate_hz_;
    instance_.rel_pose_ic_ = Mat44_t(yaml_node["rel_pose_ic"].as<std::vector<double>>().data()).transpose();
    instance_.ns_acc_ = yaml_node["ns_acc"].as<double>();
    instance_.ns_gyr_ = yaml_node["ns_gyr"].as<double>();
    instance_.rw_acc_bias_ = yaml_node["rw_acc_bias"].as<double>();
    instance_.rw_gyr_bias_ = yaml_node["rw_gyr_bias"].as<double>();
    instance_.tightly_coupled_ = yaml_node["tightly_coupled"].as<bool>();
    instance_.update_pose();
    instance_.update_covariance();

    instance_.available_ = true;
}

void config::fromJson(const nlohmann::json& json_imu) {
    instance_.name_ = json_imu.at("name").get<std::string>();
    instance_.rate_hz_ = json_imu.at("rate_hz").get<double>();
    instance_.rate_dt_ = 1.0/instance_.rate_hz_;
    instance_.rel_pose_ic_ = data::convert_json_to_matrix<Mat44_t>(json_imu.at("rel_pose_ic"));
    instance_.ns_acc_ = json_imu.at("ns_acc").get<double>();
    instance_.ns_gyr_ = json_imu.at("ns_gyr").get<double>();
    instance_.rw_acc_bias_ = json_imu.at("rw_acc_bias").get<double>();
    instance_.rw_gyr_bias_ = json_imu.at("rw_gyr_bias").get<double>();
    instance_.tightly_coupled_ = json_imu.at("tightly_coupled").get<bool>();
    instance_.update_pose();
    instance_.update_covariance();

    instance_.available_ = true;
}

void config::fromPara(std::string name, unsigned int rate_hz, const std::vector<double>& rel_pose_ic,
                      double ns_acc, double ns_gyr, double rw_acc_bias, double rw_gyr_bias) {
    instance_.name_ = name;
    instance_.rate_hz_ = rate_hz;
    instance_.rate_dt_ = 1.0/instance_.rate_hz_;
    instance_.rel_pose_ic_ = Mat44_t(rel_pose_ic.data());
    instance_.ns_acc_ = ns_acc;
    instance_.ns_gyr_ = ns_gyr;
    instance_.rw_acc_bias_ = rw_acc_bias;
    instance_.rw_gyr_bias_ = rw_gyr_bias;
    instance_.update_pose();
    instance_.update_covariance();

    instance_.available_ = true;
}

nlohmann::json config::to_json() {
    nlohmann::json json_imu_config;
    json_imu_config["name"] = instance_.name_;
    json_imu_config["rate_hz"] = instance_.rate_hz_;
    json_imu_config["rel_pose_ic"] = data::convert_matrix_to_json(instance_.rel_pose_ic_);
    json_imu_config["ns_acc"] = instance_.ns_acc_;
    json_imu_config["ns_gyr"] = instance_.ns_gyr_;
    json_imu_config["rw_acc_bias"] = instance_.rw_acc_bias_;
    json_imu_config["rw_gyr_bias"] = instance_.rw_gyr_bias_;
    return json_imu_config;
}

std::string config::get_name()  {
    return instance_.name_;
}

double config::get_rate_hz()  {
    return instance_.rate_hz_;
}

double config::get_rate_dt()  {
    return instance_.rate_dt_;
}

Mat44_t config::get_rel_pose_ic()  {
    return instance_.rel_pose_ic_;
}

Mat33_t config::get_rel_rot_ic()  {
    return instance_.rel_pose_ic_.block<3, 3>(0, 0);
}

Vec3_t config::get_rel_trans_ic()  {
    return instance_.rel_pose_ic_.block<3, 1>(0, 3);
}

Mat44_t config::get_rel_pose_ci()  {
    return instance_.rel_pose_ci_;
}

Mat33_t config::get_rel_rot_ci()  {
    return instance_.rel_pose_ci_.block<3, 3>(0, 0);
}

Vec3_t config::get_rel_trans_ci()  {
    return instance_.rel_pose_ci_.block<3, 1>(0, 3);
}

void config::set_acc_noise_density(const double ns_acc) {
    instance_.ns_acc_ = ns_acc;
    instance_.update_covariance();
}

void config::set_gyr_noise_density(const double ns_gyr) {
    instance_.ns_gyr_ = ns_gyr;
    instance_.update_covariance();
}

void config::set_acc_bias_random_walk(const double rw_acc_bias) {
    instance_.rw_acc_bias_ = rw_acc_bias;
    instance_.update_covariance();
}

void config::set_gyr_bias_random_walk(const double rw_gyr_bias) {
    instance_.rw_gyr_bias_ = rw_gyr_bias;
    instance_.update_covariance();
}

Mat33_t config::get_acc_covariance()  {
    return instance_.cov_acc_;
}

Mat33_t config::get_gyr_covariance()  {
    return instance_.cov_gyr_;
}

Mat33_t config::get_acc_bias_covariance() {
    return instance_.cov_acc_bias_;
}

Mat33_t config::get_gyr_bias_covariance()  {
    return instance_.cov_gyr_bias_;
}

void config::update_pose() {
    const Mat33_t rel_rot_ic = rel_pose_ic_.block<3, 3>(0, 0);
    const Vec3_t rel_trans_ic = rel_pose_ic_.block<3, 1>(0, 3);
    rel_pose_ci_ = Mat44_t::Identity();
    rel_pose_ci_.block<3, 3>(0, 0) = rel_rot_ic.transpose();
    rel_pose_ci_.block<3, 1>(0, 3) = -rel_rot_ic.transpose() * rel_trans_ic;
}

void config::update_covariance() {
    cov_acc_ = Mat33_t::Identity() * ns_acc_ * ns_acc_ * rate_hz_;
    cov_gyr_ = Mat33_t::Identity() * ns_gyr_ * ns_gyr_ * rate_hz_;
    cov_acc_bias_ = Mat33_t::Identity() * rw_acc_bias_ * rw_acc_bias_ * rate_hz_;
    cov_gyr_bias_ = Mat33_t::Identity() * rw_gyr_bias_ * rw_gyr_bias_ * rate_hz_;
}

} // namespace imu
} // namespace openvslam
