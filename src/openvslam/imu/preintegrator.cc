#include "openvslam/imu/preintegrator.h"
#include "openvslam/imu/preintegrated.h"
#include "openvslam/data/common.h"
#include <nlohmann/json.hpp>
#include "iostream"

namespace openvslam {
namespace imu {

measurement::measurement(const double acc_x, const double acc_y, const double acc_z,
                         const double gyr_x, const double gyr_y, const double gyr_z,
                         const double dt)
    : acc_(acc_x, acc_y, acc_z), gyr_(gyr_x, gyr_y, gyr_z), dt_(dt) {}

measurement::measurement(const Vec3_t& acc, const Vec3_t& gyr, const double dt)
    : acc_(acc), gyr_(gyr), dt_(dt) {}

preintegrator::preintegrator(const bias& b) {
    initial_covariance_ << imu::config::get_gyr_covariance(), Mat33_t::Zero(), Mat33_t::Zero(), imu::config::get_acc_covariance();
    bias_covariance_ << imu::config::get_gyr_bias_covariance(), Mat33_t::Zero(), Mat33_t::Zero(), imu::config::get_acc_bias_covariance();
    preintegrated_ = eigen_alloc_shared<preintegrated>(b);
    preintegrated_->initialize();
}

preintegrator::preintegrator(const bias& b, const Mat66_t& initial_covariance, const Mat66_t& bias_covariance) {
    initial_covariance_ = initial_covariance;
    bias_covariance_ = bias_covariance;
    preintegrated_ = eigen_alloc_shared<preintegrated>(b);
    preintegrated_->initialize();
}

preintegrator::preintegrator(const nlohmann::json& json_preintegrator) {
    initial_covariance_ = data::convert_json_to_matrix<Mat66_t>(json_preintegrator.at("initial_covariance"));
    bias_covariance_ = data::convert_json_to_matrix<Mat66_t>(json_preintegrator.at("bias_covariance"));
    preintegrated_ = eigen_alloc_shared<preintegrated>(json_preintegrator.at("preintegrated"));
    for (const auto& json_mesurement : json_preintegrator.at("measurements")) {
        measurements_.emplace_back(
            data::convert_json_to_matrix<Vec3_t>(json_mesurement.at("acc")),
            data::convert_json_to_matrix<Vec3_t>(json_mesurement.at("gyr")),
            json_mesurement.at("dt").get<double>());
    }
}

void preintegrator::reintegrate(const imu::bias& b) {
    preintegrated_->b_ = b;
    preintegrated_->initialize();
    for (const auto& m : measurements_) {
        preintegrated_->integrate(m.acc_, m.gyr_, m.dt_, initial_covariance_, bias_covariance_);
    }
}

void preintegrator::merge_previous(const preintegrator& prev) {
    const auto tmp = measurements_;
    measurements_.clear();
    preintegrated_->initialize();
    for (const auto& m : prev.measurements_) {
        integrate_new_measurement(m);
    }
    for (const auto& m : tmp) {
        integrate_new_measurement(m);
    }
}

void preintegrator::integrate_new_measurement(const measurement& m) {
    measurements_.push_back(m);
    preintegrated_->integrate(m.acc_, m.gyr_, m.dt_, initial_covariance_, bias_covariance_);
}

void preintegrator::integrate_new_measurement(const Vec3_t& acc, const Vec3_t& gyr, const double dt) {
    measurements_.emplace_back(acc, gyr, dt);
    preintegrated_->integrate(acc, gyr, dt, initial_covariance_, bias_covariance_);
}

Mat44_t preintegrator::predict_pose(const Mat44_t& Twi, const Vec3_t& v, const bias& b) {

    Vec3_t gravity(0,0,-9.81);
    const Mat33_t delta_rotation = preintegrated_->get_delta_rotation_on_bias(b);
    const Vec3_t delta_position = preintegrated_->get_delta_position_on_bias(b);
    const double dt = preintegrated_->dt_;


    const Mat33_t Rwi = Twi.topLeftCorner<3,3>();
    const Vec3_t twi = Twi.topRightCorner<3,1>();
    Mat33_t Rwi2 = Rwi*delta_rotation;
    Vec3_t twi2 = twi +v*dt+0.5*gravity*dt*dt+Rwi*delta_position;

    Mat44_t Twi2;
    Twi2<<Rwi2,twi2,
        0,0,0,1;

    return Twi2;

}

Vec3_t preintegrator::predict_velo(const Mat44_t& Twi, const Vec3_t& vi, const bias& b) {
    Vec3_t gravity(0,0,-9.81);
    const Vec3_t delta_velocity = preintegrated_->get_delta_velocity_on_bias(b);
    const double dt = preintegrated_->dt_;

    const Mat33_t Rwi = Twi.topLeftCorner<3,3>();

    Vec3_t vj = Rwi*delta_velocity+vi+gravity*dt;
    return vj;
}

nlohmann::json preintegrator::to_json() const {
    nlohmann::json json_preintegrator;
    json_preintegrator["initial_covariance"] = data::convert_matrix_to_json(initial_covariance_);
    json_preintegrator["bias_covariance"] = data::convert_matrix_to_json(bias_covariance_);
    json_preintegrator["preintegrated"] = preintegrated_->to_json();
    for (const auto& m : measurements_) {
        nlohmann::json json_mesurement;
        json_mesurement["acc"] = data::convert_matrix_to_json(m.acc_);
        json_mesurement["gyr"] = data::convert_matrix_to_json(m.gyr_);
        json_mesurement["dt"] = m.dt_;
        json_preintegrator["measurements"].push_back(json_mesurement);
    }
    return json_preintegrator;
}
} // namespace imu
} // namespace openvslam
