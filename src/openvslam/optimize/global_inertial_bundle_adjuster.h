//
// Created by tanjunyao7 on 2022/1/29.
//

#ifndef OPENVSLAM_GLOBAL_INERTIAL_BUNDLE_ADJUSTER_H
#define OPENVSLAM_GLOBAL_INERTIAL_BUNDLE_ADJUSTER_H

namespace openvslam {

namespace data {
class map_database;
} // namespace data

namespace optimize {

class global_inertial_bundle_adjuster {
public:
    /**
     * Constructor
     * @param map_db
     * @param num_iter
     * @param use_huber_kernel
     */
    explicit global_inertial_bundle_adjuster(data::map_database* map_db, const unsigned int num_iter = 10, const bool use_huber_kernel = true);

    /**
     * Destructor
     */
    virtual ~global_inertial_bundle_adjuster() = default;

    /**
     * Perform optimization
     * @param lead_keyfrm_id_in_global_BA
     * @param force_stop_flag
     * @param info_prior_acc
     * @param info_prior_gyr
     */
    void optimize(const unsigned int lead_keyfrm_id_in_global_BA = 0, bool* const force_stop_flag = nullptr,
                  double info_prior_acc = 0.0, double info_prior_gyr = 0.0) const;

private:
    //! map database
    const data::map_database* map_db_;

    //! number of iterations of optimization
    unsigned int num_iter_;

    //! use Huber loss or not
    const bool use_huber_kernel_;

    //! use shared bias or not
    bool use_shared_bias_;
};

} // namespace optimize
} // namespace openvslam

#endif //OPENVSLAM_GLOBAL_INERTIAL_BUNDLE_ADJUSTER_H
