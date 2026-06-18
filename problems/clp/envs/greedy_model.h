#ifndef GREEDY_MODEL_H
#define GREEDY_MODEL_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <map>
#include "clpState.h"
#include "envs/env.h"
#include "data/tensor_encoder.h" // Tu nuevo orquestador de tensores

namespace py = pybind11;

class GreedyModel : public ENV {
public:
    int w;
    double volume;
    
    // Agregamos los límites para los tensores
    int max_blocks;
    int max_actions;
    int max_pblocks;

    // Constructores actualizados para recibir los límites
    GreedyModel(std::string filename, int instance_number, int w, 
                int max_blocks, int max_actions, int max_pblocks, double min_fr=1.0);

    GreedyModel(clpState* s0, int w, int max_blocks, int max_actions, int max_pblocks);

    void transition(int selected_index);

    const bool is_finished() const { return completed; }

    // --- NUEVA INTERFAZ DIRECTA ---
    py::tuple get_enc_data();
    py::tuple get_dec_data();

private:
    clpState* current_node;
    bool completed = false;

    // Vital para conectar el codificador de bloques con el de acciones
    std::map<int, int> id_to_idx; 
};

#endif