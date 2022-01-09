#ifndef OPENVSLAM_IMU_IMU_DATABASE_H
#define OPENVSLAM_IMU_IMU_DATABASE_H

#include <mutex>
#include <unordered_map>

#include <nlohmann/json_fwd.hpp>

namespace openvslam {

namespace imu {

class imu_database {
public:
    void from_json(const nlohmann::json& json_cameras);

    nlohmann::json to_json() const;


};

} // namespace imu
} // namespace openvslam

#endif // OPENVSLAM_IMU_IMU_DATABASE_H
