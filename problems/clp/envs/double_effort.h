#ifndef DOUBLE_EFFORT_H
#define DOUBLE_EFFORT_H

#include "envs/env.h"

template <typename T>
class DoubleEffort : public ENV {
public:
    int w = 1;
    T* bsm; // T será BSM_GM o BSM_VCS
    double best_volume = 0;

    DoubleEffort(std::string filename, int instance_number, double timelimit) 
            : ENV(new_state(filename, instance_number, 0.98, 10000, clpState::BR), timelimit) {}

    void update() {
        w = w > 1 ? w * sqrt(2) + 0.5 : 2;
        best_volume = max(best_volume, bsm->best_volume);
    }

    T* get_env() {
        bsm = new T(s0, w, timelimit, start_time);
        return bsm;
    }

    bool is_finished() {
        return (get_elapsed_time() > timelimit);
    }
};

template <typename T>
void register_double_effort(py::module &m, const std::string &name) {
    py::class_<DoubleEffort<T>, ENV>(m, name.c_str())
        .def(py::init<std::string, int, double>(), 
             py::arg("filename"), 
             py::arg("instance_number"), 
             py::arg("timelimit"))
        .def_readwrite("best_volume", &DoubleEffort<T>::best_volume)
        .def_readwrite("w", &DoubleEffort<T>::w)

        .def("update", &DoubleEffort<T>::update)
        .def("get_env", &DoubleEffort<T>::get_env)
        .def("is_finished", &DoubleEffort<T>::is_finished);
}

#endif