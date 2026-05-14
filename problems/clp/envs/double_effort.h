#ifndef DOUBLE_EFFORT_H
#define DOUBLE_EFFORT_H

#include "envs/env.h"

template <typename T>
class DoubleEffort : public ENV {
public:
    int w = 4;
    int max_w = 4;

    T* bsm; // T será BSM_GM o BSM_VCS

    DoubleEffort(std::string filename, int instance_number, double min_fr, double timelimit, int max_w=999999) 
            : ENV(new_state(filename, instance_number, min_fr, 10000, clpState::BR), timelimit) {
        this->w = min(4, max_w);
        this->max_w = max_w;
    }

    void update() {
        w = w > 1 ? w * sqrt(2) + 0.5 : 2;
        if (bsm->best_volume > best_volume) {
            best_volume = bsm->best_volume;
            best_state = bsm->best_state;
        }
    }

    T* get_env() {
        bsm = new T(s0, w, timelimit, start_time);
        return bsm;
    }

    bool is_finished() {
        bool finished = (w > max_w) || (get_elapsed_time() > timelimit);
        if (finished) final_time = get_elapsed_time();
        return finished;    
    }
};

template <typename T>
void register_double_effort(py::module &m, const std::string &name) {
    py::class_<DoubleEffort<T>, ENV>(m, name.c_str())
        .def(py::init<std::string, int, double, double, int>(), 
             py::arg("filename"), 
             py::arg("instance_number"), 
             py::arg("min_fr"),
             py::arg("timelimit"),
             py::arg("max_w"))
        .def_readwrite("best_volume", &DoubleEffort<T>::best_volume)
        .def_readwrite("final_time", &DoubleEffort<T>::final_time)
        .def_readwrite("w", &DoubleEffort<T>::w)

        .def("update", &DoubleEffort<T>::update)
        .def("get_env", &DoubleEffort<T>::get_env)
        .def("is_finished", &DoubleEffort<T>::is_finished)
        .def("get_path_length", &DoubleEffort<T>::get_path_length);
}

#endif