#ifndef INTERACT_HPP
#define INTERACT_HPP

#include "params.hpp"
#include "state.hpp"

void compute_density(sim_state_t* s, sim_param_t* params);
void compute_accel(sim_state_t* state, sim_param_t* params);

#endif /* INTERACT_HPP */
