#include <pybind11/pybind11.h>
#include "bsm_gm.h"
#include "bsm_vcs.h"
#include "greedy_model.h"
#include "value_predictor.h"
#include "env.h"

namespace py = pybind11;

void register_bsm_gm(py::module &m);
void register_bsm_vcs(py::module &m);
void register_greedy_model(py::module &m);

PYBIND11_MODULE(bsm_engine_inf, m) {
    py::class_<ENV>(m, "ENV", py::module_local());

    register_bsm_gm(m);
    register_bsm_vcs(m);
    register_greedy_model(m);
}