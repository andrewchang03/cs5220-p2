#ifndef LEAPFROG_HPP
#define LEAPFROG_HPP

#include "state.hpp"

void leapfrog_start(sim_state_t* s, double dt);
void leapfrog_step(sim_state_t* s, double dt);

#endif /* LEAPFROG_HPP */
