#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" void build_signals_batch(
    const double* ema_values,   // shape: [n_periods, n_rows], row-major
    int n_periods,
    int n_rows,
    const int32_t* idx1,        // shape: [n_combos]
    const int32_t* idx2,        // shape: [n_combos]
    const int32_t* idx3,        // shape: [n_combos]
    int n_combos,
    uint8_t* out_entries,       // shape: [n_rows, n_combos], row-major
    uint8_t* out_exits          // shape: [n_rows, n_combos], row-major
) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n_combos; ++j) {
        const int p1 = idx1[j];
        const int p2 = idx2[j];
        const int p3 = idx3[j];

        if (p1 < 0 || p2 < 0 || p3 < 0 || p1 >= n_periods || p2 >= n_periods || p3 >= n_periods) {
            for (int t = 0; t < n_rows; ++t) {
                out_entries[t * n_combos + j] = 0;
                out_exits[t * n_combos + j] = 0;
            }
            continue;
        }

        bool prev_entry_raw = false;
        bool prev_exit_raw = false;

        out_entries[0 * n_combos + j] = 0;
        out_exits[0 * n_combos + j] = 0;

        for (int t = 1; t < n_rows; ++t) {
            const double a_prev = ema_values[p1 * n_rows + (t - 1)];
            const double a_cur = ema_values[p1 * n_rows + t];
            const double b_prev = ema_values[p2 * n_rows + (t - 1)];
            const double b_cur = ema_values[p2 * n_rows + t];
            const double c_prev = ema_values[p3 * n_rows + (t - 1)];
            const double c_cur = ema_values[p3 * n_rows + t];

            bool entry_raw = false;
            bool exit_raw = false;

            const bool valid_ab = std::isfinite(a_prev) && std::isfinite(a_cur) &&
                                  std::isfinite(b_prev) && std::isfinite(b_cur);
            const bool valid_ac = std::isfinite(a_prev) && std::isfinite(a_cur) &&
                                  std::isfinite(c_prev) && std::isfinite(c_cur);
            const bool valid_bc = std::isfinite(b_prev) && std::isfinite(b_cur) &&
                                  std::isfinite(c_prev) && std::isfinite(c_cur);

            const bool above_ab = valid_ab && (a_prev <= b_prev) && (a_cur > b_cur);
            const bool above_ac = valid_ac && (a_prev <= c_prev) && (a_cur > c_cur);
            const bool above_bc = valid_bc && (b_prev <= c_prev) && (b_cur > c_cur);

            const bool below_ab = valid_ab && (a_prev >= b_prev) && (a_cur < b_cur);
            const bool below_ac = valid_ac && (a_prev >= c_prev) && (a_cur < c_cur);
            const bool below_bc = valid_bc && (b_prev >= c_prev) && (b_cur < c_cur);

            entry_raw = above_ab || above_ac || above_bc;
            exit_raw = below_ab || below_ac || below_bc;

            out_entries[t * n_combos + j] = prev_entry_raw ? 1 : 0;
            out_exits[t * n_combos + j] = prev_exit_raw ? 1 : 0;

            prev_entry_raw = entry_raw;
            prev_exit_raw = exit_raw;
        }
    }
}
