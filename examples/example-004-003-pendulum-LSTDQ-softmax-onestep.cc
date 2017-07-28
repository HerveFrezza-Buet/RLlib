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
#include <unistd.h>
#include <functional>

using namespace std::placeholders;

// This is our simulator.
typedef rl::problem::inverted_pendulum::Simulator<rl::problem::inverted_pendulum::DefaultParam> Simulator;

// Definition of Reward, S, A, Transition and TransitionSet.
#include "example-defs-transition.hpp"

// Features and a RBF architecture.
#include "example-defs-pendulum-architecture.hpp"


// Let us define the parameters.
#define paramGAMMA                          0.95
#define paramREG                            10.0
#define paramALPHA_P                         0.01

// For LSTD-Q, we need to collect some statistics (from transitions)
// before beginning to update the parameter vector of the greedy policy
// Here, we collect 5000 transitions which correspond approximately to
// 300 episodes because initially, the episode length of a random policy
// is around 15-20 balancing steps
#define NB_OF_TRANSITIONS_WARMUP  5000
#define NB_OF_EPISODES             500
#define NB_OF_TESTING_EPISODES      50
#define MAX_EPISODE_LENGTH        3000


double fct_p(gsl_vector* tmp, const gsl_vector* theta, const S& s, const A& a) {
  double res;
  phi_rbf(tmp,s,a);              // phi_sa = phi(s,a)
  gsl_blas_ddot(theta,tmp,&res); // res    = th^T  . phi_sa
  return res;
}

template<typename AITER>
std::vector<double> get_action_probabilities(AITER action_begin,
					     AITER action_end,
					     unsigned int nb_actions,
					     gsl_vector* tmp,
					     const gsl_vector* theta_p,
					     const S& s) {
    std::vector<double> probaActions(nb_actions);
    auto aiter = action_begin;
    double psum = 0.0;
    auto piter = probaActions.begin();
    while(aiter != action_end) {
      *piter = exp(fct_p(tmp, theta_p, s, *aiter));
      psum += *piter;
      ++aiter;
      ++piter;
    }
    for(auto& p: probaActions)
      p /= psum;
    return probaActions;
  };

template<typename AITER>
void fct_grad_log_p(AITER action_begin, AITER action_end,
		    unsigned int nb_actions,
		    gsl_vector* tmp,
		    const gsl_vector* theta_p,
		    gsl_vector* grad_log_p,
		    const S& s, const A& a) {
  gsl_vector_set_zero(grad_log_p);
    
  // We need to get the probabilities of the actions
  auto probaActions = get_action_probabilities(action_begin, action_end, nb_actions, tmp, theta_p, s);

  // TO BE DONE !!!!!
  
  // // And we can then compute the gradient of ln(Pi)
  // auto piter = probaActions.begin();
  // auto aiter = action_begin;
  // while(aiter != action_end) {
  //   gsl_vector_set(grad_log_p, nb_states * std::distance(action_begin, aiter) + s, ((*aiter)==a) - (*piter));
  //   ++aiter;
  //   ++piter;
  // }
}




int main(int argc, char* argv[]) {
   
  // Let us initialize the random seed.
  rl::random::seed(getpid());

  // 1) Instantiate the simulator
  int             episode,episode_length;
  Simulator       simulator;
  unsigned int nb_actions = 3;
  
  ////////////// Actor/Critic
  // 2a) Instantiate the Critic, we here use LSTD-Q
  gsl_vector* theta_q = gsl_vector_calloc(PHI_RBF_DIMENSION);
  
  auto critic = rl::gsl::LSTDQ<S, A>(theta_q,
				     paramGAMMA, paramREG,
				     NB_OF_TRANSITIONS_WARMUP,
				     phi_rbf);

  // 2b) Instantiate the Actor
  gsl_vector* theta_p = gsl_vector_calloc(PHI_RBF_DIMENSION);
  gsl_vector* tmp = gsl_vector_calloc(PHI_RBF_DIMENSION);
  auto scores = std::bind(fct_p, tmp, theta_p, std::placeholders::_1, std::placeholders::_2);
  rl::enumerator<A> action_begin(rl::problem::inverted_pendulum::actionNone);
  rl::enumerator<A> action_end = action_begin+nb_actions;
  auto policy  = rl::policy::softmax(scores, 1.0,
				     action_begin,action_end);

  // 2c) And finally the actor/critic
  /////// TO BE DONE , especially grad(log(p))
  auto grad = std::bind(fct_grad_log_p<decltype(action_begin)>, action_begin, action_end, nb_actions, tmp, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
  auto actor_critic = rl::gsl::ActorCritic::one_step<S, A>(critic,
							   theta_p,
							   paramALPHA_P,
							   grad);
  
  try {
 
    
    for(episode = 0 ; episode < NB_OF_EPISODES; ++episode) {
      simulator.setPhase(Simulator::phase_type());
      episode_length = rl::episode::learn(simulator,
					  policy,actor_critic,
					  MAX_EPISODE_LENGTH);
      
      // After each episode, we test our policy for NB_OF_TESTING_EPISODES
      // episodes
      double cumul_episode_length = 0.0;
      for(unsigned int tepi = 0 ; tepi < NB_OF_TESTING_EPISODES; ++tepi) {
	simulator.setPhase(Simulator::phase_type());
	cumul_episode_length += rl::episode::run(simulator, policy, MAX_EPISODE_LENGTH);
      }
      std::cout << "\r Episode " << episode << " : mean length over " << NB_OF_TESTING_EPISODES << " episodes is " << cumul_episode_length/double(NB_OF_TESTING_EPISODES) << std::string(10, ' ') << std::flush;
      
    }
  }
  catch(rl::exception::Any& e) {
    std::cerr << "Exception caught : " << e.what() << std::endl; 
  }
  std::cout << std::endl;

  gsl_vector_free(theta_q);
  gsl_vector_free(theta_p);
}
