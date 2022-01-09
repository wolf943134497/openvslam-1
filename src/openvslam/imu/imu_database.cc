#include "openvslam/imu/config.h"
#include "openvslam/imu/imu_database.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace openvslam {
namespace imu {



void imu_database::from_json(const nlohmann::json& json_imus) {

    for (const auto& json_id_camera : json_imus.items()) {
        const auto& imu_name = json_id_camera.key();
        const auto& json_imu = json_id_camera.value();

        spdlog::info("load a imu \"{}\" from JSON", imu_name);
//        assert(!database_.count(imu_name));
//        database_[imu_name] = eigen_alloc_shared<config>(json_imu);
        imu::config::fromJson(json_imu);
    }
}

nlohmann::json imu_database::to_json() const {
    nlohmann::json json_imus;
//    for (const auto& name_imu : database_) {
//        const auto& imu_name = name_imu.first;
//        const auto imu = name_imu.second;
        json_imus[imu::config::get_name()] = imu::config::to_json();
//    }
    return json_imus;
}

} // namespace imu
} // namespace openvslam
