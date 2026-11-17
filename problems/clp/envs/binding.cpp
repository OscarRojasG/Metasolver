#include <pybind11/pybind11.h>
#include "bsg_hybrid.h"
#include "greedy_model.h"
#include "env.h"

namespace py = pybind11;

void register_bsg_hybrid(py::module &m);
void register_greedy_model(py::module &m);

PYBIND11_MODULE(clp_engine, m) {
    py::class_<ENV>(m, "ENV", py::module_local());

    register_bsg_hybrid(m);
    register_greedy_model(m);
}
