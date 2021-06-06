#include <openvslam/imu/internal/inertial_gravity_scale_edge_on_camera.h>
#include <openvslam/imu/internal/inertial_gravity_scale_edge_on_imu.h>
#include <openvslam/imu/internal/prior_bias_edge.h>

#include <gtest/gtest.h>

#include <sstream>
#include <iomanip>
#include <limits>

using namespace openvslam;

TEST(imu_edges, io_prior_bias_edge) {
    imu::internal::prior_bias_edge edge;
    edge.setInformation(Mat33_t::Identity());
    edge.setMeasurement(Vec3_t(0.1, 0.2, 0.3));

    std::stringstream ss;
    ss << std::setprecision(std::numeric_limits<double>::max_digits10) << std::scientific;
    edge.write(ss);

    imu::internal::prior_bias_edge edge2;
    edge2.read(ss);

    EXPECT_TRUE(edge.information().isApprox(edge2.information()));
    EXPECT_TRUE(edge.measurement().isApprox(edge2.measurement()));
}
