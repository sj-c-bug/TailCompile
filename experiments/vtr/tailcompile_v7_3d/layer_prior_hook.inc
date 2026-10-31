// Isolated 007 layer prior. Native XY, fixed nodes and router costs are unchanged.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

[[maybe_unused]] static bool tailcompile_layer_enabled() {
    const char* value = std::getenv("TC_GP_LAYER_PRIOR");
    if (!value || std::string(value) == "0") return false;
    if (std::string(value) != "1" || !std::getenv("TC_GP_INIT"))
        VPR_FATAL_ERROR(VPR_ERROR_AP, "TC_GP_LAYER_PRIOR requires 0/1 and TC_GP_INIT for 1");
    return true;
}

[[maybe_unused]] static std::vector<double> tailcompile_layer_targets;
[[maybe_unused]] static unsigned tailcompile_layer_logged_iteration = std::numeric_limits<unsigned>::max();
[[maybe_unused]] static double tailcompile_layer_beta = 0.;

template<class Placement>
[[maybe_unused]] static void tailcompile_layer_capture(const Placement& placement) {
    tailcompile_layer_targets.clear();
    tailcompile_layer_logged_iteration = std::numeric_limits<unsigned>::max();
    if (!tailcompile_layer_enabled()) return;
    for (double z : placement.block_layer_nums) tailcompile_layer_targets.push_back(z);
}

template<class Matrix, class Vector, class Rows>
[[maybe_unused]] static void tailcompile_layer_apply(Matrix& matrix, Vector& rhs,
                                                     unsigned iteration, const Rows& row_to_block) {
    if (!tailcompile_layer_enabled()) return;
    const double factor = iteration == 1 ? 1. : iteration == 2 ? .5 : 0.;
    if (tailcompile_layer_logged_iteration != iteration) {
        std::vector<double> diagonal;
        for (int i = 0; i < matrix.rows(); ++i) {
            const double value = matrix.coeff(i, i);
            if (std::isfinite(value) && value > 0.) diagonal.push_back(value);
        }
        std::sort(diagonal.begin(), diagonal.end());
        double median = 0.;
        if (!diagonal.empty()) {
            const auto n = diagonal.size();
            median = n % 2 ? diagonal[n / 2] : .5 * (diagonal[n / 2 - 1] + diagonal[n / 2]);
        }
        tailcompile_layer_beta = factor * .1 * median;
        tailcompile_layer_logged_iteration = iteration;
        VTR_LOG("TAILCOMPILE_V7_LAYER_PRIOR iteration=%u factor=%.17g beta=%.17g rows=%d\n",
                iteration, factor, tailcompile_layer_beta, int(matrix.rows()));
    }
    if (tailcompile_layer_beta == 0.) return;
    for (int i = 0; i < matrix.rows(); ++i) {
        const size_t block = size_t(row_to_block(i));
        if (block >= tailcompile_layer_targets.size())
            VPR_FATAL_ERROR(VPR_ERROR_AP, "Layer prior target missing for movable solver row");
        matrix.coeffRef(i, i) += tailcompile_layer_beta;
        rhs[i] += tailcompile_layer_beta * tailcompile_layer_targets[block];
    }
}
