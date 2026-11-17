cmake_minimum_required(VERSION 3.12) # Aumentado para mejor soporte de Python
project(CLP_Project)

# 1. Configuración de C++
set(CMAKE_CXX_STANDARD 14) # Recomendado 14 para pybind11
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # CRÍTICO para crear módulos de Python (.so)

# 2. Localizar Python y pybind11
add_subdirectory(extern/pybind11)

INCLUDE_DIRECTORIES(
lib/bullet-2.80/src
metasolver
metasolver/strategies/
problems/clp/containers
problems/clp/objects2
problems/clp/evaluators
problems/clp
problems/clp/plugins/kdtree
problems/clp/plugins/ga
)

LINK_DIRECTORIES(
lib/bullet-2.80/src/BulletCollision
lib/bullet-2.80/src/BulletDynamics
lib/bullet-2.80/src/LinearMath
)

LINK_LIBRARIES(
 BulletDynamics BulletCollision LinearMath
)
    if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
    endif(COMMAND cmake_policy)


 SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# ADD_DEFINITION(-g -pg)

ADD_LIBRARY(
bsg
metasolver/SearchStrategy.cpp
metasolver/GlobalVariables.cpp
metasolver/strategies/BSG.cpp
metasolver/strategies/Greedy.cpp
metasolver/strategies/GRASP.cpp
metasolver/State.cpp
)

ADD_LIBRARY(
clp
problems/clp/containers/AABBContainer.cpp
problems/clp/containers/AABBTree.cpp
problems/clp/evaluators/VCS_Function.cpp
problems/clp/evaluators/VLossFunction.cpp
problems/clp/objects2/AABB.cpp
problems/clp/objects2/Block.cpp
problems/clp/objects2/BoxShape.cpp
problems/clp/objects2/Space.cpp
problems/clp/objects2/SpaceSet.cpp
problems/clp/objects2/Block_fsb.cpp
)

ADD_LIBRARY(
clp_state
problems/clp/clpState.cpp
)

ADD_LIBRARY(
clp_kdtree
problems/clp/plugins/kdtree/clpStatekd.cpp
problems/clp/plugins/kdtree/kdNode.cpp
problems/clp/plugins/kdtree/BlockSet.cpp
)

pybind11_add_module(clp_engine
    problems/clp/envs/bsg_hybrid.cpp
    problems/clp/envs/greedy_model.cpp
    problems/clp/envs/binding.cpp
)

target_link_libraries(clp_engine PRIVATE
    -Wl,--whole-archive bsg clp clp_state clp_kdtree -Wl,--no-whole-archive
    BulletDynamics BulletCollision LinearMath
)

set_target_properties(clp_engine PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(clp_engine PROPERTIES VISIBILITY_INLINES_HIDDEN ON)


ADD_EXECUTABLE(BSG_VCS problems/clp/main_clp.cpp)
ADD_EXECUTABLE(VCS problems/clp/vcs.cpp)

# BSG-VCS con w hijos fijos por nodo (en lugar de w*w / |S|), para la ablación de la regla de expansión
ADD_EXECUTABLE(BSG_VCS_FIXED_W problems/clp/main_clp.cpp)
target_compile_definitions(BSG_VCS_FIXED_W PRIVATE BSG_FIXED_W)

target_link_libraries(BSG_VCS clp_state clp bsg)
target_link_libraries(BSG_VCS_FIXED_W clp_state clp bsg)
target_link_libraries(VCS clp_state clp bsg)
