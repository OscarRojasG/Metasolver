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
problems/mclp
metasolver/plugins/mop/strategies/
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
metasolver/strategies/MCTS_RolloutStrategy.cpp
metasolver/strategies/GuidedRolloutStrategy.cpp
metasolver/State.cpp
)

ADD_LIBRARY(
bsg_mop
 metasolver/plugins/mop/strategies/BSGMOP.cpp
)

ADD_LIBRARY(
nsga2
 metasolver/plugins/mop/strategies/Chromosome.cpp
 metasolver/plugins/mop/strategies/NSGA2.cpp
 problems/clp/plugins/ga/clp_chromosome.cpp
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
mclp_state
problems/mclp/mclp-state.cpp
)

ADD_LIBRARY(
clp_kdtree
problems/clp/plugins/kdtree/clpStatekd.cpp
problems/clp/plugins/kdtree/kdNode.cpp
problems/clp/plugins/kdtree/BlockSet.cpp
)

pybind11_add_module(bsm_engine 
    #problems/clp/envs/bsm_gm.cpp
    problems/clp/envs/bsm_vcs.cpp
    problems/clp/envs/greedy_model.cpp
    problems/clp/envs/value_predictor.cpp
    problems/clp/envs/binding.cpp
)

target_link_libraries(bsm_engine PRIVATE 
    -Wl,--whole-archive bsg clp clp_state clp_kdtree -Wl,--no-whole-archive
    BulletDynamics BulletCollision LinearMath
)


ADD_EXECUTABLE(MCLP problems/mclp/main/mclp-solver.cpp)
ADD_EXECUTABLE(BSG_CLP problems/clp/main_clp.cpp)
ADD_EXECUTABLE(BSG problems/clp/bsg.cpp)
ADD_EXECUTABLE(VCS problems/clp/vcs.cpp)
ADD_EXECUTABLE(BSG_CLP_ANN problems/clp/plugins/ann/main_clp_ann.cpp)
ADD_EXECUTABLE(CLP_GENSOLV problems/clp/plugins/ann/gen_ran_inst_solve.cpp)
ADD_EXECUTABLE(test_BSG_CLP problems/clp/test_clp.cpp)
ADD_EXECUTABLE(BSG_B problems/clp/clp_bsg-mop.cpp)
ADD_EXECUTABLE(test_BSGMOP_CLP problems/clp/test_clp_mop.cpp)
#ADD_EXECUTABLE(testEvalMOP problems/clp/test_evalMOP.cpp)
ADD_EXECUTABLE(CLP_gen problems/clp/ClpInstanceGenerator.cpp)
ADD_EXECUTABLE(NSGA_CLP problems/clp/plugins/ga/main.cpp)

target_link_libraries(MCLP mclp_state  clp bsg)
target_link_libraries(BSG clp_state clp bsg)
target_link_libraries(BSG_CLP clp_state clp bsg)
target_link_libraries(VCS clp_state clp bsg)
target_link_libraries(BSG_CLP_ANN clp_state clp clp_kdtree bsg)
target_link_libraries(test_BSG_CLP clp_state clp clp_kdtree bsg)
target_link_libraries(CLP_GENSOLV clp_state clp clp_kdtree bsg)
target_link_libraries(NSGA_CLP clp_state clp nsga2 bsg)

target_link_libraries(BSG_B bsg_mop clp_state clp bsg)
target_link_libraries(test_BSGMOP_CLP bsg_mop clp_state clp bsg)
#target_link_libraries(testEvalMOP bsg_mop clp_state clp bsg)

enable_testing()

add_test( NAME o-search
   COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/BSG_B problems/clp/benchs/BRwp-1.0-0.5/BR4.txt -i 1 --min_fr=0.98 -t 1 -s o-search --fp=R -f BRwp )

 add_test( NAME bsg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/BSG_B problems/clp/benchs/BRwp-1.0-0.5/BR4.txt -i 1 --min_fr=0.98 -t 1 -s bsg --fp=R -f BRwp )

  add_test( NAME bsg_p
     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/BSG_B problems/clp/benchs/BRwp-1.0-0.5/BR4.txt -i 1 --min_fr=0.98 -t 1 -s bsg_p --fp=R -f BRwp )

 add_test( NAME bsg_vp
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/BSG_B problems/clp/benchs/BRwp-1.0-0.5/BR4.txt -i 1 --min_fr=0.98 -t 2 -s bsg_vp --fp=R -f BRwp )

IF (INTERNAL_ADD_POSTFIX_EXECUTABLE_NAMES)
			SET_TARGET_PROPERTIES(AppHelloWorld PROPERTIES  DEBUG_POSTFIX "_Debug")
			SET_TARGET_PROPERTIES(AppHelloWorld PROPERTIES  MINSIZEREL_POSTFIX "_MinsizeRel")
			SET_TARGET_PROPERTIES(AppHelloWorld PROPERTIES  RELWITHDEBINFO_POSTFIX "_RelWithDebugInfo")
ENDIF(INTERNAL_ADD_POSTFIX_EXECUTABLE_NAMES)
