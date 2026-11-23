#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "maxlfq_core.h"

namespace py = pybind11;

static py::dict py_run_maxlfq(const std::vector<int> &protein_index,
                               const std::vector<int> &ion_index,
                               const std::vector<int> &sample_index,
                               const std::vector<double> &quant) {
    if (protein_index.size() != ion_index.size() ||
        protein_index.size() != sample_index.size() ||
        protein_index.size() != quant.size()) {
        throw std::invalid_argument(
            "protein_index, ion_index, sample_index, quant must have equal length");
    }

    auto result = iq_native::run_maxlfq(protein_index, ion_index, sample_index, quant);

    py::dict out;
    out["estimate"] = result.estimate;             // -> numpy 2D array (n_proteins x n_samples)
    out["row_protein_index"] = result.row_protein_index;
    out["col_sample_index"] = result.col_sample_index;
    out["annotation"] = result.annotation;
    return out;
}

PYBIND11_MODULE(_iq_native, m) {
    m.doc() = "R-free C++ MaxLFQ core, ported from tvpham/iq (BSD-3-Clause)";
    m.def("run_maxlfq", &py_run_maxlfq,
          py::arg("protein_index"), py::arg("ion_index"),
          py::arg("sample_index"), py::arg("quant"),
          "Run the MaxLFQ solve. protein_index/sample_index must be 0-based "
          "dense codes (e.g. from pandas.factorize).");
}
