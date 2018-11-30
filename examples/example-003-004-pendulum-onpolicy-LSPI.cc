/*   This file is part of rl-lib
 *
 *   Copyright (C) 2017,  Supelec
 *
 *   Author : Herve Frezza-Buet and Matthieu Geist
 *
 *   Contributor : Jeremy Fix
 *
 *   This library is free software; you can redistribute it and/or
 *   modify it under the terms of the GNU General Public
 *   License (GPL) as published by the Free Software Foundation; either
 *   version 3 of the License, or any later version.
 *   
 *   This library is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 *   General Public License for more details.
 *   
 *   You should have received a copy of the GNU General Public
 *   License along with this library; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 *   Contact : Herve.Frezza-Buet@supelec.fr Matthieu.Geist@supelec.fr
 *
 */

/*
   This example shows how to use a parametric representation of the
   Q-function and apply recursive LSTD-Q to estimate the Q function. The inverted pendulum problem is solved here.
   LSTD-Q gathers some transitions before updating the parameter vector. Then, the parameter vector is continuously updated.
   */

#include <rl.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <random>

using namespace std::placeholders;

// This is our simulator.
using Simulator = rl::problem::inverted_pendulum::Simulator<rl::problem::inverted_pendulum::DefaultParam, std::mt19937>;

// Definition of Reward, S, A, Transition and TransitionSet.
#include "example-defs-transition.hpp"

// Features and a RBF architecture.
#include "example-defs-pendulum-architecture.hpp"


// Let us define the parameters.
#define paramGAMMA                          0.95
#define paramREG                            10.0

// For LSTD-Q, we need to collect some statistics (from transitions)
// before beginning to update the parameter vector of the greedy policy
// Here, we collect 5000 transitions which correspond approximately to
// 300 episodes because initially, the episode length of a random policy
// is around 15-20 balancing steps
#define NB_OF_TRANSITIONS_WARMUP  5000
#define NB_OF_EPISODES             500
#define NB_OF_TESTING_EPISODES      50
#define MAX_EPISODE_LENGTH        3000

int main(int argc, char* argv[]) {

    std::random_device rd;
    std::mt19937 gen(rd());

    int             episode;

    Simulator       simulator(gen);

    gsl_vector* theta = gsl_vector_calloc(PHI_RBF_DIMENSION);
    gsl_vector* tmp = gsl_vector_calloc(PHI_RBF_DIMENSION);

    auto q_parametrized = [tmp](const gsl_vector* th,S s, A a) -> Reward {double res;
        phi_rbf(tmp,s,a);           // phi_sa = phi(s,a)
        gsl_blas_ddot(th,tmp,&res); // res    = th^T  . phi_sa
        return res;};


    auto q = std::bind(q_parametrized,theta,_1,_2);

    // We instantiate our LSTD-Q
    //auto critic = rl::gsl::LSTDQ_Lambda<S, A>(theta, paramGAMMA, paramREG, .4, NB_OF_TRANSITIONS_WARMUP, phi_rbf);
    auto critic = rl::gsl::LSTDQ<S, A>(theta, paramGAMMA, paramREG, NB_OF_TRANSITIONS_WARMUP, phi_rbf);

    rl::enumerator<A> a_begin(rl::problem::inverted_pendulum::Action::actionNone);
    rl::enumerator<A> a_end = a_begin+rl::problem::inverted_pendulum::actionSize;
    auto greedy_policy  = rl::policy::greedy(q, a_begin,a_end);

    Simulator::phase_type start_phase;
    try {
        for(episode = 0 ; episode < NB_OF_EPISODES; ++episode) {
            start_phase.random(gen);
            simulator.setPhase(start_phase);
            rl::episode::learn(simulator,
                    greedy_policy,critic,
                    MAX_EPISODE_LENGTH);
            
            // After each episode, we test our policy for NB_OF_TESTING_EPISODES
            // episodes
            double cumul_episode_length = 0.0;
            for(unsigned int tepi = 0 ; tepi < NB_OF_TESTING_EPISODES; ++tepi) {
                start_phase.random(gen);
                simulator.setPhase(start_phase);
                cumul_episode_length += rl::episode::run(simulator, greedy_policy, MAX_EPISODE_LENGTH);
            }
            std::cout << "\r Episode " << episode << " : mean length over " << NB_OF_TESTING_EPISODES << " episodes is " << cumul_episode_length/double(NB_OF_TESTING_EPISODES) << std::string(10, ' ') << std::flush;

        }
    }
    catch(rl::exception::Any& e) {
        std::cerr << "Exception caught : " << e.what() << std::endl; 
    }
    std::cout << std::endl;
}
