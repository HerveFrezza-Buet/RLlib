/*   This file is part of rl-lib
 *
 *   Copyright (C) 2010,  Supelec
 *
 *   Author : Herve Frezza-Buet and Matthieu Geist
 *
 *   Contributor :
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
   Q-function and apply LSPI. The inverted pendulum problem is solved
   here.
   */

#include <rl.hpp>
#include <iostream>
#include <iomanip>
#include <gsl/gsl_vector.h>
#include <cmath>
#include <fstream>
#include <functional>

using namespace std::placeholders;

// This is our simulator.
using Simulator = rl::problem::inverted_pendulum::Simulator<rl::problem::inverted_pendulum::DefaultParam, std::mt19937>;

// Definition of Reward, S, A, Transition and TransitionSet.
#include "example-defs-transition.hpp"

// Features and a RBF architecture.
#include "example-defs-pendulum-architecture.hpp"


#define paramREG       0
#define paramGAMMA    .95

#define NB_OF_EPISODES         1000
#define NB_ITERATION_STEPS       10
#define MAX_EPISODE_LENGTH     3000
#define NB_LENGTH_SAMPLES        20

#include "example-defs-test-iteration.hpp"

int main(int argc, char* argv[]) {

    std::random_device rd;
    std::mt19937 gen(rd());

    int             episode,step;
    std::ofstream   ofile;

    Simulator       simulator(gen);
    TransitionSet   transitions;

    gsl_vector* theta = gsl_vector_alloc(PHI_RBF_DIMENSION);
    gsl_vector_set_zero(theta);
    gsl_vector* tmp = gsl_vector_alloc(PHI_RBF_DIMENSION);
    gsl_vector_set_zero(tmp);

    auto q_parametrized = [tmp](const gsl_vector* th,S s, A a) -> Reward {double res;
        phi_rbf(tmp,s,a);           // phi_sa = phi(s,a)
        gsl_blas_ddot(th,tmp,&res); // res    = th^T  . phi_sa
        return res;};
    auto grad_q_parametrized = [tmp](const gsl_vector* th,   
            gsl_vector* grad_th_s,
            S s, A a) -> void {phi_rbf(tmp,s,a);   // phi_sa    = phi(s,a)
        gsl_vector_memcpy(grad_th_s,tmp);};        // grad_th_s = phi_sa

    auto q = std::bind(q_parametrized,theta,_1,_2);

    rl::enumerator<A> a_begin(rl::problem::inverted_pendulum::Action::actionNone);
    rl::enumerator<A> a_end = a_begin+3;

    auto random_policy = rl::policy::random(a_begin,a_end,gen);
    auto greedy_policy = rl::policy::greedy(q,a_begin,a_end);

    try {
        // Let us fill a set of transitions from successive episodes,
        // using a random policy.
        for(episode=0;episode<NB_OF_EPISODES;++episode) {
            Simulator::phase_type start_phase;
            start_phase.random(gen);
            simulator.setPhase(start_phase);
            rl::episode::run(simulator,
                    random_policy,
                    std::back_inserter(transitions),
                    make_transition,
                    make_terminal_transition,
                    0);
        }

        // Let us try the random policy
        test_iteration(random_policy,0, gen);

        // Let us used LSTD as a batch critic. LSTD considers transitions
        // as (Z,r,Z'), but Z is a pair (s,a) here. See the definitions of
        // current_of,next_of,reward_of, and note that
        // gradvparam_of_gradqparam transforms Q(s,a) into V(z).
        auto critic = [theta,grad_q_parametrized](const TransitionSet::iterator& t_begin,
                const TransitionSet::iterator& t_end) -> void {
            rl::lstd(theta,paramGAMMA,paramREG,
                    t_begin,t_end,
                    rl::sa::gsl::gradvparam_of_gradqparam<S,A,Reward>(grad_q_parametrized),
                    current_of,next_of,reward_of,is_terminal);
        };

        // Now, let us improve the policy and measure its performance at each step.
        for(step = 1 ; step <= NB_ITERATION_STEPS ; ++step) {
            rl::batch_policy_iteration_step(critic,q,
                    transitions.begin(),transitions.end(),
                    a_begin,a_end,
                    is_terminal,next_state_of,set_next_action);
            test_iteration(greedy_policy,step, gen);
        }

        // Now, we can save the q_theta parameter
        std::cout << "Writing lspi.data" << std::endl;
        ofile.open("lspi.data");
        if(!ofile)
            std::cerr << "cannot open file for writing" << std::endl;
        else {
            ofile << theta << std::endl;
            ofile.close();
        }

    }
    catch(rl::exception::Any& e) {
        std::cerr << "Exception caught : " << e.what() << std::endl;
    }

    return 0;
}
