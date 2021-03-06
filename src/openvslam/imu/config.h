#ifndef OPENVSLAM_IMU_CONFIG_H
#define OPENVSLAM_IMU_CONFIG_H

#include "openvslam/type.h"
#include <yaml-cpp/yaml.h>
#include <nlohmann/json_fwd.hpp>

namespace openvslam {
namespace imu {

class config {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    config(config& another) = delete;
    config& operator==(config& another)=delete;

    void static fromYaml(const YAML::Node& yaml_node);

    void static fromJson(const nlohmann::json& json_imu);


    void static fromPara(std::string name,unsigned int rate_hz,const std::vector<double>& rel_pose_ic,
                         double ns_acc, double ns_gyr,double rw_acc_bias, double rw_gyr_bias);
    //! Create json from IMU config
    static nlohmann::json to_json();


    inline static bool available(){
        return instance_.available_;
    }

    inline static bool is_tightly_coupled(){
        return instance_.tightly_coupled_;
    }
    //---------------------------
    // Setters and Getters

    //! Get IMU model name
    static std::string get_name();
    //! Get IMU rate [Hz]
    static double get_rate_hz();
    //! Get IMU rate [s]
    static double get_rate_dt();



    //! Get IMU's relative pose w.r.t. the camera
    static Mat44_t get_rel_pose_ic() ;
    //! Get IMU's relative rotation w.r.t. the camera
    static Mat33_t get_rel_rot_ic() ;
    //! Get IMU's relative translation w.r.t. the camera
    static Vec3_t get_rel_trans_ic() ;
    //! Get camera's relative pose w.r.t. the IMU
    static Mat44_t get_rel_pose_ci() ;
    //! Get camera's relative rotation w.r.t. the IMU
    static Mat33_t get_rel_rot_ci() ;
    //! Get camera's relative translation w.r.t. the IMU
    static Vec3_t get_rel_trans_ci() ;

    //! Set acceleration noise density [m/s^2/sqrt(Hz)]
    static void set_acc_noise_density(const double ns_acc);
    //! Set gyroscope noise density [rad/s/sqrt(Hz)]
    static void set_gyr_noise_density(const double ns_gyr);
    //! Set random walk of acceleration sensor bias [m/s^3/sqrt(Hz)]
    static void set_acc_bias_random_walk(const double rw_acc_bias);
    //! Set random walk of gyroscope sensor bias [rad/s^2/sqrt(Hz)]
    static void set_gyr_bias_random_walk(const double rw_gyr_bias);

    //! Get acceleration covariance [(m/s^2)^2]
    static Mat33_t get_acc_covariance() ;
    //! Get gyroscope covariance [(rad/s)^2]
    static Mat33_t get_gyr_covariance() ;
    //! Get acceleration bias covariance [(m/s^3)^2]
    static Mat33_t get_acc_bias_covariance() ;
    //! Get gyroscope bias covariance [(rad/s^2)^2]
    static Mat33_t get_gyr_bias_covariance() ;

private:
    config(){}
    ~config(){}

    static config instance_;

    bool available_{false};

    //! Update rel_pose_ci_ using rel_pose_ic_
    void update_pose();
    //! Update covariances using the currently assigned variables
    void update_covariance();

    //! IMU model name
    std::string name_;
    //! IMU rate [Hz]
    double rate_hz_;
    //! IMU rate [s]
    double rate_dt_;

    //! IMU's relative pose w.r.t the camera
    Mat44_t rel_pose_ic_;
    //! camera's relative pose w.r.t the IMU
    Mat44_t rel_pose_ci_;

    //! covariance of acceleration sensor [(m/s^2)^2] = [m/s^2/sqrt(Hz)] * [m/s^2/sqrt(Hz)] * [Hz]
    Mat33_t cov_acc_ = Mat33_t::Identity();
    //! covariance of gyroscope sensor [(rad/s)^2] = [rad/s/sqrt(Hz)] * [rad/s/sqrt(Hz)] * [Hz]
    Mat33_t cov_gyr_ = Mat33_t::Identity();
    //! covariance gy acceleration sensor bias [(m/s^3)^2] = [m/s^3/sqrt(Hz)] * [m/s^3/sqrt(Hz)] * [Hz]
    Mat33_t cov_acc_bias_ = Mat33_t::Identity();
    //! covariance of gyroscope sensor bias [(rad/s^2)^2] = [rad/s^2/sqrt(Hz)] * [rad/s^2/sqrt(Hz)] * [Hz]
    Mat33_t cov_gyr_bias_ = Mat33_t::Identity();

    //! noise density of acceleration sensor [m/s^2/sqrt(Hz)]
    double ns_acc_;
    //! noise density of gyroscope sensor [rad/s/sqrt(Hz)]
    double ns_gyr_;
    //! random walk of acceleration sensor bias [m/s^3/sqrt(Hz)]
    double rw_acc_bias_;
    //! random walk of gyroscope sensor bias [rad/s^2/sqrt(Hz)]
    double rw_gyr_bias_;
    //! whether jointly optimize
    bool tightly_coupled_;
};

} // namespace imu
} // namespace openvslam

#endif // OPENVSLAM_IMU_CONFIG_H
