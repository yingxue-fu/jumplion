/*#########################################################################

 MaxLFQ core algorithm, extracted from tvpham/iq (src/iq.cpp, the
 `ion_table` class used inside `iq_MaxLFQ()`).

 This file is a straight port: the per-protein grouping logic and the
 `ion_table::maxLFQ()` least-squares solve are byte-for-byte the same
 algorithm as the R package. The only thing removed is the R marshaling
 layer (SEXP unpacking/packing, Rprintf progress messages, R interrupt
 checks) -- none of that touched the numerics.

 Original author: Thang V. Pham, t.pham@amsterdamumc.nl
 Source: https://github.com/tvpham/iq  (BSD 3-Clause)
 Citation: Pham TV, Henneman AA, Jimenez CR. iq: an R package to estimate
 relative protein abundances from ion quantification in DIA-MS-based
 proteomics, Bioinformatics 2020 Apr 15;36(8):2611-2613.

#########################################################################*/

#ifndef __IQ_MAXLFQ_CORE_H
#define __IQ_MAXLFQ_CORE_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils.h"

namespace iq_native {

using Eigen::FullPivHouseholderQR;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// One protein's fragment/ion x sample table, and the MaxLFQ solve for it.
// Identical logic to `ion_table` in iq.cpp, minus anything R-specific.
class IonTable {
public:
    explicit IonTable(int ncol) : ncol_(ncol) {}

    void add(int ion, int col, double quant) {
        auto r = map_.emplace(ion, std::vector<double>());
        if (r.second) {
            r.first->second.resize(ncol_, NAN);
        }
        r.first->second[col] = quant;
    }

    // buffer[j]: MaxLFQ estimate for sample j (NAN if not estimable)
    // group[j]:  connected-component id for sample j
    void solve(double *buffer, int *group,
               int full_connection, const FullPivHouseholderQR<MatrixXd> &full_qr) {
        if (map_.empty()) {
            std::fill_n(buffer, ncol_, NAN);
            std::fill_n(group, ncol_, 0);
            return;
        }

        if (map_.size() == 1) {
            const auto &only = map_.begin()->second;
            std::copy(only.begin(), only.end(), buffer);
            std::fill_n(group, ncol_, 0);
            return;
        }

        std::vector<double> median_buffer(map_.size());
        int median_size = 0;

        std::fill_n(group, ncol_, -1);

        int val = 0;
        for (int i = 0; i < ncol_; i++) {
            if (group[i] < 0) {
                spread(group, i, val++);
            }
        }

        std::fill_n(buffer, ncol_, NAN);

        for (int i = 0; i < val; i++) {
            std::vector<int> ind;
            for (int j = 0; j < ncol_; j++) {
                if (group[j] == i) {
                    ind.push_back(j);
                }
            }

            if (ind.size() == 1) {
                median_size = 0;
                for (const auto &m : map_) {
                    if (!std::isnan(m.second[ind[0]])) {
                        median_buffer[median_size++] = m.second[ind[0]];
                    }
                }
                buffer[ind[0]] = median_size == 0
                                      ? NAN
                                      : quick_median(median_buffer.data(), median_size);
            } else {
                int N = static_cast<int>(ind.size());

                MatrixXd AtA = MatrixXd::Zero(N + 1, N + 1);
                VectorXd Atb = VectorXd::Zero(N + 1);

                int n_connection = 0;

                for (int j = 0; j < (N - 1); j++) {
                    for (int k = j + 1; k < N; k++) {
                        median_size = 0;
                        for (const auto &m : map_) {
                            if (!std::isnan(m.second[ind[j]]) && !std::isnan(m.second[ind[k]])) {
                                median_buffer[median_size++] = m.second[ind[k]] - m.second[ind[j]];
                            }
                        }
                        if (median_size > 0) {
                            n_connection++;
                            double r_i_j = quick_median(median_buffer.data(), median_size);
                            AtA(j, k) = -1.0;
                            AtA(k, j) = -1.0;
                            AtA(j, j) += 1.0;
                            AtA(k, k) += 1.0;
                            Atb(j) -= r_i_j;
                            Atb(k) += r_i_j;
                        }
                    }
                }

                AtA *= 2.0;
                Atb *= 2.0;

                for (int j = 0; j < N; j++) {
                    AtA(j, N) = 1.0;
                    AtA(N, j) = 1.0;
                }
                AtA(N, N) = 0.0;

                double sum = 0.0;
                int count = 0;
                for (const auto &m : map_) {
                    for (const auto &s : ind) {
                        if (!std::isnan(m.second[s])) {
                            sum += m.second[s];
                            count++;
                        }
                    }
                }
                Atb(N) = sum * static_cast<double>(N) / static_cast<double>(count);

                VectorXd x;
                if (n_connection == full_connection) {
                    x = full_qr.solve(Atb);
                } else {
                    FullPivHouseholderQR<MatrixXd> qr(AtA);
                    x = qr.solve(Atb);
                }
                for (int j = 0; j < N; j++) {
                    buffer[ind[j]] = x(j);
                }
            }
        }
    }

    bool empty() const { return map_.empty(); }

private:
    void spread(int *g, int i, int val) {
        g[i] = val;
        for (const auto &m : map_) {
            if (!std::isnan(m.second[i])) {
                for (int j = 0; j < ncol_; j++) {
                    if (g[j] < 0 && !std::isnan(m.second[j])) {
                        spread(g, j, val);
                    }
                }
            }
        }
    }

    std::unordered_map<int, std::vector<double>> map_;
    int ncol_;
};

struct MaxLFQResult {
    MatrixXd estimate;                  // n_proteins x n_samples, row-major semantics handled by caller
    std::vector<int> row_protein_index; // original (0-based) protein codes, in output row order
    std::vector<int> col_sample_index;  // original (0-based) sample codes, in output col order
    std::vector<std::string> annotation;
};

// Direct port of the grouping + solve loop that used to live inside
// iq_MaxLFQ() in iq.cpp, with the SEXP unpacking/packing stripped out.
//
// protein_index, ion_index, sample_index, quant are parallel arrays,
// one entry per observed (protein, ion, sample) quantity -- exactly the
// `norm_data` columns R's fast_MaxLFQ() passes through .Call().
// protein_index and sample_index must be 0-based dense-ish codes
// (as produced by e.g. pandas.factorize / numpy.unique(..., return_inverse=True)).
inline MaxLFQResult run_maxlfq(const std::vector<int> &protein_index,
                                const std::vector<int> &ion_index,
                                const std::vector<int> &sample_index,
                                const std::vector<double> &quant) {
    const std::size_t nrow = protein_index.size();

    int n_proteins = *std::max_element(protein_index.begin(), protein_index.end()) + 1;
    int n_samples = *std::max_element(sample_index.begin(), sample_index.end()) + 1;

    // group observation row indices by protein
    std::vector<std::vector<std::size_t>> by_protein(n_proteins);
    for (std::size_t i = 0; i < nrow; i++) {
        by_protein[protein_index[i]].push_back(i);
    }

    // full-connection QR, reused whenever a protein's samples are all
    // pairwise connected (the common case) -- same trick as ion_table::init()
    MatrixXd AtA = MatrixXd::Zero(n_samples + 1, n_samples + 1);
    for (int j = 0; j < n_samples - 1; j++) {
        for (int k = j + 1; k < n_samples; k++) {
            AtA(j, k) = -1.0;
            AtA(k, j) = -1.0;
            AtA(j, j) += 1.0;
            AtA(k, k) += 1.0;
        }
    }
    AtA *= 2.0;
    for (int j = 0; j < n_samples; j++) {
        AtA(j, n_samples) = 1.0;
        AtA(n_samples, j) = 1.0;
    }
    AtA(n_samples, n_samples) = 0.0;
    FullPivHouseholderQR<MatrixXd> full_qr(AtA);
    int full_connection = n_samples * (n_samples - 1) / 2;

    MaxLFQResult result;
    result.estimate = MatrixXd::Constant(n_proteins, n_samples, NAN);
    result.annotation.assign(n_proteins, "");
    result.row_protein_index.resize(n_proteins);
    result.col_sample_index.resize(n_samples);
    for (int j = 0; j < n_samples; j++) result.col_sample_index[j] = j;
    for (int i = 0; i < n_proteins; i++) result.row_protein_index[i] = i;

    #pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < n_proteins; p++) {
        if (by_protein[p].empty()) {
            continue;
        }

        IonTable tab(n_samples);
        for (auto row : by_protein[p]) {
            tab.add(ion_index[row], sample_index[row], quant[row]);
        }

        std::vector<double> w(n_samples);
        std::vector<int> gr(n_samples);
        tab.solve(w.data(), gr.data(), full_connection, full_qr);

        // annotation: mark disconnected components, same rule as iq_MaxLFQ
        bool single_component = true;
        int component = -1;
        for (int j = 0; j < n_samples; j++) {
            if (!std::isnan(w[j])) {
                if (component == -1) {
                    component = gr[j];
                } else if (component != gr[j]) {
                    single_component = false;
                    break;
                }
            }
        }
        if (component == -1) {
            result.annotation[p] = "NA";
        } else if (!single_component) {
            std::string s;
            for (int j = 0; j < n_samples; j++) {
                if (j > 0) s.push_back(';');
                s += std::isnan(w[j]) ? std::string("NA") : std::to_string(gr[j] + 1);
            }
            result.annotation[p] = s;
        }

        for (int j = 0; j < n_samples; j++) {
            result.estimate(p, j) = w[j];
        }
    }

    return result;
}

}  // namespace iq_native

#endif  // __IQ_MAXLFQ_CORE_H
