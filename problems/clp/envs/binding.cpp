#include <pybind11/pybind11.h>
#include "bsm_gm.h"
#include "bsm_vcs.h"
#include "greedy_model.h"
#include "double_effort.h"
#include "env.h"

namespace py = pybind11;

void register_bsm_gm(py::module &m);
void register_bsm_vcs(py::module &m);
void register_greedy_model(py::module &m);

PYBIND11_MODULE(bsm_engine, m) {
    py::class_<ENV>(m, "ENV");
    py::class_<BSM_ENV, ENV>(m, "BSM_ENV");

    register_bsm_gm(m);
    register_bsm_vcs(m);
    register_greedy_model(m);
    register_double_effort<BSM_GM>(m, "DoubleEffort_BSM_GM");
    register_double_effort<BSM_VCS>(m, "DoubleEffort_BSM_VCS");
}