#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr double kTiny = 1e-300;

double log_sum_exp(const std::vector<double>& values) {
    double m = -std::numeric_limits<double>::infinity();
    for (double v : values) {
        if (v > m) {
            m = v;
        }
    }
    if (!std::isfinite(m)) {
        return m;
    }
    double s = 0.0;
    for (double v : values) {
        s += std::exp(v - m);
    }
    return m + std::log(std::max(s, kTiny));
}

double gaussian_logpdf(double x, double mu, double var) {
    double v = std::max(var, 1e-12);
    double z = x - mu;
    return -0.5 * (std::log(2.0 * M_PI * v) + (z * z) / v);
}

void normalize_row(std::vector<double>& row) {
    double s = std::accumulate(row.begin(), row.end(), 0.0);
    if (s <= 0.0 || !std::isfinite(s)) {
        double w = 1.0 / static_cast<double>(row.size());
        for (double& v : row) {
            v = w;
        }
        return;
    }
    for (double& v : row) {
        v /= s;
    }
}

void viterbi_decode(
    const std::vector<double>& obs,
    int n_states,
    const std::vector<double>& pi,
    const std::vector<double>& trans,
    const std::vector<double>& means,
    const std::vector<double>& vars,
    std::vector<int32_t>& states_out
) {
    const int T = static_cast<int>(obs.size());
    states_out.assign(T, 0);
    if (T <= 0 || n_states <= 0) {
        return;
    }

    std::vector<double> delta(T * n_states, -std::numeric_limits<double>::infinity());
    std::vector<int32_t> psi(T * n_states, 0);

    for (int i = 0; i < n_states; ++i) {
        delta[i] = std::log(std::max(pi[i], kTiny)) + gaussian_logpdf(obs[0], means[i], vars[i]);
    }

    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < n_states; ++j) {
            double best = -std::numeric_limits<double>::infinity();
            int32_t best_i = 0;
            for (int i = 0; i < n_states; ++i) {
                double cand = delta[(t - 1) * n_states + i] + std::log(std::max(trans[i * n_states + j], kTiny));
                if (cand > best) {
                    best = cand;
                    best_i = i;
                }
            }
            delta[t * n_states + j] = best + gaussian_logpdf(obs[t], means[j], vars[j]);
            psi[t * n_states + j] = best_i;
        }
    }

    double best_last = -std::numeric_limits<double>::infinity();
    int32_t best_state = 0;
    for (int i = 0; i < n_states; ++i) {
        double v = delta[(T - 1) * n_states + i];
        if (v > best_last) {
            best_last = v;
            best_state = i;
        }
    }
    states_out[T - 1] = best_state;

    for (int t = T - 2; t >= 0; --t) {
        states_out[t] = psi[(t + 1) * n_states + states_out[t + 1]];
    }
}

void fit_hmm_em(
    const std::vector<double>& train_obs,
    int n_states,
    int n_iter,
    double var_floor,
    double trans_sticky,
    std::vector<double>& pi,
    std::vector<double>& trans,
    std::vector<double>& means,
    std::vector<double>& vars
) {
    const int T_train = static_cast<int>(train_obs.size());
    if (T_train <= 2 || n_states <= 1) {
        pi.assign(static_cast<size_t>(std::max(n_states, 1)), 1.0);
        trans.assign(static_cast<size_t>(std::max(n_states * n_states, 1)), 1.0);
        means.assign(static_cast<size_t>(std::max(n_states, 1)), 0.0);
        vars.assign(static_cast<size_t>(std::max(n_states, 1)), std::max(var_floor, 1e-12));
        return;
    }

    pi.assign(static_cast<size_t>(n_states), 1.0 / static_cast<double>(n_states));
    trans.assign(static_cast<size_t>(n_states * n_states), 0.0);
    for (int i = 0; i < n_states; ++i) {
        for (int j = 0; j < n_states; ++j) {
            double base = (1.0 - trans_sticky) / static_cast<double>(n_states - 1);
            trans[i * n_states + j] = (i == j) ? trans_sticky : base;
        }
    }

    double mn = train_obs[0];
    double mx = train_obs[0];
    double mean_all = 0.0;
    for (int t = 0; t < T_train; ++t) {
        mn = std::min(mn, train_obs[t]);
        mx = std::max(mx, train_obs[t]);
        mean_all += train_obs[t];
    }
    mean_all /= static_cast<double>(T_train);

    double var_all = 0.0;
    for (int t = 0; t < T_train; ++t) {
        double z = train_obs[t] - mean_all;
        var_all += z * z;
    }
    var_all = std::max(var_all / static_cast<double>(T_train), var_floor);

    means.assign(static_cast<size_t>(n_states), mean_all);
    vars.assign(static_cast<size_t>(n_states), var_all);
    if (mx > mn) {
        for (int i = 0; i < n_states; ++i) {
            double q = (static_cast<double>(i) + 0.5) / static_cast<double>(n_states);
            means[i] = mn + q * (mx - mn);
        }
    }

    std::vector<double> alpha(static_cast<size_t>(T_train * n_states), 0.0);
    std::vector<double> beta(static_cast<size_t>(T_train * n_states), 0.0);
    std::vector<double> gamma(static_cast<size_t>(T_train * n_states), 0.0);
    std::vector<double> c(static_cast<size_t>(T_train), 1.0);

    for (int iter = 0; iter < n_iter; ++iter) {
        for (int i = 0; i < n_states; ++i) {
            alpha[i] = pi[i] * std::exp(gaussian_logpdf(train_obs[0], means[i], vars[i]));
        }
        c[0] = std::accumulate(alpha.begin(), alpha.begin() + n_states, 0.0);
        c[0] = std::max(c[0], kTiny);
        for (int i = 0; i < n_states; ++i) {
            alpha[i] /= c[0];
        }

        for (int t = 1; t < T_train; ++t) {
            for (int j = 0; j < n_states; ++j) {
                double s = 0.0;
                for (int i = 0; i < n_states; ++i) {
                    s += alpha[(t - 1) * n_states + i] * trans[i * n_states + j];
                }
                alpha[t * n_states + j] = s * std::exp(gaussian_logpdf(train_obs[t], means[j], vars[j]));
            }
            c[t] = 0.0;
            for (int j = 0; j < n_states; ++j) {
                c[t] += alpha[t * n_states + j];
            }
            c[t] = std::max(c[t], kTiny);
            for (int j = 0; j < n_states; ++j) {
                alpha[t * n_states + j] /= c[t];
            }
        }

        for (int i = 0; i < n_states; ++i) {
            beta[(T_train - 1) * n_states + i] = 1.0;
        }

        for (int t = T_train - 2; t >= 0; --t) {
            for (int i = 0; i < n_states; ++i) {
                double s = 0.0;
                for (int j = 0; j < n_states; ++j) {
                    s += trans[i * n_states + j]
                        * std::exp(gaussian_logpdf(train_obs[t + 1], means[j], vars[j]))
                        * beta[(t + 1) * n_states + j];
                }
                beta[t * n_states + i] = s / std::max(c[t + 1], kTiny);
            }
        }

        for (int t = 0; t < T_train; ++t) {
            double denom = 0.0;
            for (int i = 0; i < n_states; ++i) {
                gamma[t * n_states + i] = alpha[t * n_states + i] * beta[t * n_states + i];
                denom += gamma[t * n_states + i];
            }
            denom = std::max(denom, kTiny);
            for (int i = 0; i < n_states; ++i) {
                gamma[t * n_states + i] /= denom;
            }
        }

        for (int i = 0; i < n_states; ++i) {
            pi[i] = gamma[i];
        }

        std::vector<double> xi_sum(static_cast<size_t>(n_states * n_states), 0.0);
        std::vector<double> gamma_sum(static_cast<size_t>(n_states), 0.0);

        for (int t = 0; t < T_train - 1; ++t) {
            double denom = 0.0;
            for (int i = 0; i < n_states; ++i) {
                for (int j = 0; j < n_states; ++j) {
                    denom += alpha[t * n_states + i]
                        * trans[i * n_states + j]
                        * std::exp(gaussian_logpdf(train_obs[t + 1], means[j], vars[j]))
                        * beta[(t + 1) * n_states + j];
                }
            }
            denom = std::max(denom, kTiny);

            for (int i = 0; i < n_states; ++i) {
                gamma_sum[i] += gamma[t * n_states + i];
                for (int j = 0; j < n_states; ++j) {
                    double num = alpha[t * n_states + i]
                        * trans[i * n_states + j]
                        * std::exp(gaussian_logpdf(train_obs[t + 1], means[j], vars[j]))
                        * beta[(t + 1) * n_states + j];
                    xi_sum[i * n_states + j] += num / denom;
                }
            }
        }

        for (int i = 0; i < n_states; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < n_states; ++j) {
                double val = xi_sum[i * n_states + j] / std::max(gamma_sum[i], kTiny);
                trans[i * n_states + j] = std::max(val, 1e-12);
                row_sum += trans[i * n_states + j];
            }
            if (row_sum <= 0.0 || !std::isfinite(row_sum)) {
                for (int j = 0; j < n_states; ++j) {
                    trans[i * n_states + j] = 1.0 / static_cast<double>(n_states);
                }
            } else {
                for (int j = 0; j < n_states; ++j) {
                    trans[i * n_states + j] /= row_sum;
                }
            }
        }

        for (int i = 0; i < n_states; ++i) {
            double gsum = 0.0;
            double num_mean = 0.0;
            for (int t = 0; t < T_train; ++t) {
                double g = gamma[t * n_states + i];
                gsum += g;
                num_mean += g * train_obs[t];
            }
            gsum = std::max(gsum, kTiny);
            means[i] = num_mean / gsum;

            double num_var = 0.0;
            for (int t = 0; t < T_train; ++t) {
                double d = train_obs[t] - means[i];
                num_var += gamma[t * n_states + i] * d * d;
            }
            vars[i] = std::max(num_var / gsum, var_floor);
        }
    }
}

void forward_filter_decode(
    const std::vector<double>& obs,
    int n_states,
    const std::vector<double>& pi,
    const std::vector<double>& trans,
    const std::vector<double>& means,
    const std::vector<double>& vars,
    std::vector<int32_t>& states_out
) {
    const int T = static_cast<int>(obs.size());
    states_out.assign(T, 0);
    if (T <= 0 || n_states <= 0) {
        return;
    }

    std::vector<double> alpha(static_cast<size_t>(n_states), 0.0);
    std::vector<double> alpha_next(static_cast<size_t>(n_states), 0.0);

    for (int i = 0; i < n_states; ++i) {
        alpha[i] = pi[i] * std::exp(gaussian_logpdf(obs[0], means[i], vars[i]));
    }
    normalize_row(alpha);
    states_out[0] = static_cast<int32_t>(std::distance(alpha.begin(), std::max_element(alpha.begin(), alpha.end())));

    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < n_states; ++j) {
            double s = 0.0;
            for (int i = 0; i < n_states; ++i) {
                s += alpha[i] * trans[i * n_states + j];
            }
            alpha_next[j] = s * std::exp(gaussian_logpdf(obs[t], means[j], vars[j]));
        }
        normalize_row(alpha_next);
        states_out[t] = static_cast<int32_t>(std::distance(alpha_next.begin(), std::max_element(alpha_next.begin(), alpha_next.end())));
        alpha.swap(alpha_next);
    }
}

void forward_filter_decode_with_init(
    const std::vector<double>& obs,
    int n_states,
    const std::vector<double>& init_probs,
    const std::vector<double>& trans,
    const std::vector<double>& means,
    const std::vector<double>& vars,
    std::vector<int32_t>& states_out
) {
    const int T = static_cast<int>(obs.size());
    states_out.assign(T, 0);
    if (T <= 0 || n_states <= 0) {
        return;
    }

    std::vector<double> alpha(static_cast<size_t>(n_states), 0.0);
    std::vector<double> alpha_next(static_cast<size_t>(n_states), 0.0);

    for (int i = 0; i < n_states; ++i) {
        alpha[i] = init_probs[i] * std::exp(gaussian_logpdf(obs[0], means[i], vars[i]));
    }
    normalize_row(alpha);
    states_out[0] = static_cast<int32_t>(std::distance(alpha.begin(), std::max_element(alpha.begin(), alpha.end())));

    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < n_states; ++j) {
            double s = 0.0;
            for (int i = 0; i < n_states; ++i) {
                s += alpha[i] * trans[i * n_states + j];
            }
            alpha_next[j] = s * std::exp(gaussian_logpdf(obs[t], means[j], vars[j]));
        }
        normalize_row(alpha_next);
        states_out[t] = static_cast<int32_t>(std::distance(alpha_next.begin(), std::max_element(alpha_next.begin(), alpha_next.end())));
        alpha.swap(alpha_next);
    }
}

void forward_filter_probs(
    const std::vector<double>& obs,
    int n_states,
    const std::vector<double>& init_probs,
    const std::vector<double>& trans,
    const std::vector<double>& means,
    const std::vector<double>& vars,
    std::vector<double>& probs_out
) {
    const int T = static_cast<int>(obs.size());
    probs_out.assign(static_cast<size_t>(std::max(T * n_states, 0)), 0.0);
    if (T <= 0 || n_states <= 0) {
        return;
    }

    std::vector<double> alpha(static_cast<size_t>(n_states), 0.0);
    std::vector<double> alpha_next(static_cast<size_t>(n_states), 0.0);

    for (int i = 0; i < n_states; ++i) {
        alpha[i] = init_probs[i] * std::exp(gaussian_logpdf(obs[0], means[i], vars[i]));
    }
    normalize_row(alpha);
    for (int i = 0; i < n_states; ++i) {
        probs_out[i] = alpha[i];
    }

    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < n_states; ++j) {
            double s = 0.0;
            for (int i = 0; i < n_states; ++i) {
                s += alpha[i] * trans[i * n_states + j];
            }
            alpha_next[j] = s * std::exp(gaussian_logpdf(obs[t], means[j], vars[j]));
        }
        normalize_row(alpha_next);
        for (int i = 0; i < n_states; ++i) {
            probs_out[t * n_states + i] = alpha_next[i];
        }
        alpha.swap(alpha_next);
    }
}

std::vector<double> forward_last_posterior(
    const std::vector<double>& obs,
    int n_states,
    const std::vector<double>& pi,
    const std::vector<double>& trans,
    const std::vector<double>& means,
    const std::vector<double>& vars
) {
    std::vector<double> alpha(static_cast<size_t>(n_states), 0.0);
    std::vector<double> alpha_next(static_cast<size_t>(n_states), 0.0);
    if (obs.empty() || n_states <= 0) {
        return alpha;
    }

    for (int i = 0; i < n_states; ++i) {
        alpha[i] = pi[i] * std::exp(gaussian_logpdf(obs[0], means[i], vars[i]));
    }
    normalize_row(alpha);

    for (size_t t = 1; t < obs.size(); ++t) {
        for (int j = 0; j < n_states; ++j) {
            double s = 0.0;
            for (int i = 0; i < n_states; ++i) {
                s += alpha[i] * trans[i * n_states + j];
            }
            alpha_next[j] = s * std::exp(gaussian_logpdf(obs[t], means[j], vars[j]));
        }
        normalize_row(alpha_next);
        alpha.swap(alpha_next);
    }
    return alpha;
}

}  // namespace

extern "C" int fit_predict_hmm_1d(
    const double* train_obs,
    int n_train,
    const double* test_obs,
    int n_test,
    int n_states,
    int n_iter,
    double var_floor,
    double trans_sticky,
    int32_t* train_states_out,
    int32_t* test_states_out,
    double* state_means_out,
    double* state_vars_out
) {
    if (!train_obs || !train_states_out || !test_states_out || !state_means_out || !state_vars_out) {
        return 1;
    }
    if (n_train <= 2 || n_states <= 1) {
        return 2;
    }

    std::vector<double> train_vec(static_cast<size_t>(n_train));
    std::vector<double> test_vec(static_cast<size_t>(n_test));
    for (int t = 0; t < n_train; ++t) {
        train_vec[t] = train_obs[t];
    }
    for (int t = 0; t < n_test; ++t) {
        test_vec[t] = test_obs[t];
    }

    std::vector<double> pi;
    std::vector<double> trans;
    std::vector<double> means;
    std::vector<double> vars;
    fit_hmm_em(train_vec, n_states, n_iter, var_floor, trans_sticky, pi, trans, means, vars);

    std::vector<int32_t> train_states;
    std::vector<int32_t> test_states;
    viterbi_decode(train_vec, n_states, pi, trans, means, vars, train_states);

    std::vector<double> init_test = forward_last_posterior(train_vec, n_states, pi, trans, means, vars);
    forward_filter_decode_with_init(test_vec, n_states, init_test, trans, means, vars, test_states);

    for (int t = 0; t < n_train; ++t) {
        train_states_out[t] = train_states[t];
    }
    for (int t = 0; t < n_test; ++t) {
        test_states_out[t] = test_states[t];
    }
    for (int i = 0; i < n_states; ++i) {
        state_means_out[i] = means[i];
        state_vars_out[i] = vars[i];
    }

    return 0;
}

extern "C" int fit_hmm_forward_probs_1d(
    const double* train_obs,
    int n_train,
    const double* test_obs,
    int n_test,
    int n_states,
    int n_iter,
    double var_floor,
    double trans_sticky,
    double* train_probs_out,
    double* test_probs_out,
    double* state_means_out,
    double* state_vars_out
) {
    if (!train_obs || !test_obs || !train_probs_out || !test_probs_out || !state_means_out || !state_vars_out) {
        return 1;
    }
    if (n_train <= 2 || n_states <= 1 || n_test <= 0) {
        return 2;
    }

    std::vector<double> train_vec(static_cast<size_t>(n_train));
    std::vector<double> test_vec(static_cast<size_t>(n_test));
    for (int t = 0; t < n_train; ++t) {
        train_vec[t] = train_obs[t];
    }
    for (int t = 0; t < n_test; ++t) {
        test_vec[t] = test_obs[t];
    }

    std::vector<double> pi;
    std::vector<double> trans;
    std::vector<double> means;
    std::vector<double> vars;
    fit_hmm_em(train_vec, n_states, n_iter, var_floor, trans_sticky, pi, trans, means, vars);

    std::vector<double> train_probs;
    std::vector<double> test_probs;
    forward_filter_probs(train_vec, n_states, pi, trans, means, vars, train_probs);

    std::vector<double> init_test = forward_last_posterior(train_vec, n_states, pi, trans, means, vars);
    forward_filter_probs(test_vec, n_states, init_test, trans, means, vars, test_probs);

    for (int t = 0; t < n_train; ++t) {
        for (int s = 0; s < n_states; ++s) {
            train_probs_out[t * n_states + s] = train_probs[t * n_states + s];
        }
    }
    for (int t = 0; t < n_test; ++t) {
        for (int s = 0; s < n_states; ++s) {
            test_probs_out[t * n_states + s] = test_probs[t * n_states + s];
        }
    }
    for (int i = 0; i < n_states; ++i) {
        state_means_out[i] = means[i];
        state_vars_out[i] = vars[i];
    }

    return 0;
}
