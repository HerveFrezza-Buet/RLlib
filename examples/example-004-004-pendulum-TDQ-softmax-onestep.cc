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

// Experiment : pendulum
// Architecture : RBF features phi(s,a) with continuity in s.
// Critic : TD-Q with a linear Q functon over phi(s,a)
// Actor : Softmax with scores computes as a linear combinations of phi(s,a)
// Learner : One-step Actor Critic

// This script is not able to learn a good controller
// To check : it might not be so easy to learn the inverted pendulum with TD-Q actor critic. It might be more efficient to learn it with something like generalized advantage estimation : https://arxiv.org/pdf/1506.02438.pdf


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
//typedef rl::problem::inverted_pendulum::Simulator<rl::problem::inverted_pendulum::DefaultParam> Simulator;

class DefaultParam {
      public:
	// This is the amplitude of the noise (relative) applied to the action.
	inline static double actionNoise(void)        {return 0.0;}
	// This is the noise of angle perturbation from the equilibrium state at initialization.
	inline static double angleInitNoise(void)     {return 1e-3;}
	// This is the noise of speed perturbation from the equilibrium state at initialization.
	inline static double speedInitNoise(void)    {return 1e-3;}
	
};

typedef rl::problem::inverted_pendulum::Simulator<DefaultParam> Simulator;

// Definition of Reward, S, A, Transition and TransitionSet.
#include "example-defs-transition.hpp"

// Features and a RBF architecture.
#include "example-defs-pendulum-architecture.hpp"


// Let us define the parameters.
#define paramGAMMA                           0.9
#define paramALPHA_V                         0.1
#define paramALPHA_P                         0.01
#define paramTEMP                            1.0

#define NB_OF_EPISODES            5000
#define NB_OF_TESTING_EPISODES      50
#define MAX_EPISODE_LENGTH        3000


double fct_q(gsl_vector* tmp, const gsl_vector* theta, const S& s, const A& a) {
  double res;
  phi_rbf(tmp, s, a);              // phi_sa = phi(s,a)
  gsl_blas_ddot(theta, tmp, &res); // res    = th^T  . phi_sa
  return res;
}

void fct_grad_q(const gsl_vector* theta, gsl_vector* grad, const S& s, const A& a) {
  phi_rbf(grad, s, a);
}


template<typename AITER>
std::vector<double> get_action_probabilities(AITER action_begin,
					     AITER action_end,
					     unsigned int nb_actions,
					     double temperature,
					     gsl_vector* tmp,
					     const gsl_vector* theta_p,
					     const S& s) {
    std::vector<double> probaActions(nb_actions);
    auto aiter = action_begin;
    double psum = 0.0;
    auto piter = probaActions.begin();
    while(aiter != action_end) {
      *piter = exp(fct_q(tmp, theta_p, s, *aiter)/temperature);
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
		    double temperature, 
		    gsl_vector* tmp,
		    const gsl_vector* theta_p,
		    gsl_vector* grad_log_p,
		    const S& s, const A& a) {
  gsl_vector_set_zero(grad_log_p);
    
  // We need to get the probabilities of the actions
  auto probaActions = get_action_probabilities(action_begin, action_end, nb_actions, temperature, tmp, theta_p, s);
  
  // And we can then compute the gradient of ln(Pi)
  // which is : dlog(phi(a/s))/dtheta = phi(s, a) - sum_b pi(b/s) phi(s, b)
  auto piter = probaActions.begin();
  auto aiter = action_begin;

  phi_rbf(grad_log_p, s, a);
  while(aiter != action_end) {
    phi_rbf(tmp, s, *aiter);
    gsl_blas_daxpy(-(*piter), tmp, grad_log_p);
    ++aiter;
    ++piter;
  }
  gsl_vector_scale(grad_log_p, 1. /temperature);
}

void save_qfunction(std::string filename,
		    int Ns,
		    gsl_vector* tmp,
		    const gsl_vector* theta) {
  std::ofstream outfile(filename);
  for(unsigned int i = 0 ; i < Ns; ++i){
    auto s = S(-M_PI_2 + M_PI * i / (Ns-1.), 0.);
    outfile << s.angle << " " << s.speed << " -1 " << fct_q(tmp, theta, s, rl::problem::inverted_pendulum::actionLeft) << std::endl;
    outfile << s.angle << " " << s.speed << " 0 " << fct_q(tmp, theta, s, rl::problem::inverted_pendulum::actionNone) << std::endl;
    outfile << s.angle << " " << s.speed <<  " 1 " << fct_q(tmp, theta, s, rl::problem::inverted_pendulum::actionRight) << std::endl;
  }
  outfile.close();
}

template<typename AITER>
void save_policy(std::string filename,
		 int Ns,
		 AITER action_begin, AITER action_end,
		 unsigned int nb_actions,
		 double temperature, 
		 gsl_vector* tmp,
		 const gsl_vector* theta_p) {
  std::ofstream outfile(filename);
  for(unsigned int i = 0 ; i < Ns; ++i){
    auto s = S(-M_PI_2 + M_PI * i / (Ns-1.), 0.);
    auto probaActions = get_action_probabilities(action_begin, action_end, nb_actions, temperature, tmp, theta_p, s);
    
    outfile << s.angle << " " << s.speed << " -1 " << probaActions[1] << std::endl;
    outfile << s.angle << " " << s.speed << " 0 " << probaActions[0] << std::endl;
    outfile << s.angle << " " << s.speed <<  " 1 " << probaActions[2] << std::endl;
  }
  outfile.close();
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
  gsl_vector* tmp = gsl_vector_calloc(PHI_RBF_DIMENSION);
  
  auto critic = rl::gsl::td<S, A>(theta_q,
				  paramGAMMA, paramALPHA_V,
				  std::bind(fct_q, tmp, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
				  fct_grad_q);
  

  // 2b) Instantiate the Actor
  gsl_vector* theta_p = gsl_vector_calloc(PHI_RBF_DIMENSION);
  auto scores = std::bind(fct_q, tmp, theta_p, std::placeholders::_1, std::placeholders::_2);
  rl::enumerator<A> action_begin(rl::problem::inverted_pendulum::actionNone);
  rl::enumerator<A> action_end = action_begin+nb_actions;
  double temperature = paramTEMP;
  
  auto policy  = rl::policy::softmax(scores, temperature,
				     action_begin,action_end);
  auto greedy_policy  = rl::policy::greedy(scores, action_begin,action_end);

  // 2c) And finally the actor/critic
  auto grad = std::bind(fct_grad_log_p<decltype(action_begin)>, action_begin, action_end, nb_actions, temperature, tmp, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
  auto actor_critic = rl::gsl::ActorCritic::one_step<S, A>(critic,
  							   theta_p,
  							   paramALPHA_P,
  							   grad);

  auto save_value_function = [tmp, theta_q](std::string filename) { save_qfunction(filename, 20, tmp, theta_q);
  };
  auto save_p = [action_begin, action_end, nb_actions, temperature, tmp, theta_p](std::string filename) {
    save_policy(filename, 20, action_begin, action_end, nb_actions, temperature, tmp, theta_p);
  };

  save_value_function("q0.data");
  save_p("p0.data");
  
  try {
    
    for(episode = 0 ; episode < NB_OF_EPISODES; ++episode) {
      simulator.setPhase(Simulator::phase_type());
      rl::episode::learn(simulator,
			 policy,actor_critic,
			 MAX_EPISODE_LENGTH);
      
      // After each episode, we test our policy for NB_OF_TESTING_EPISODES
      // episodes
      double cumul_episode_length = 0.0;
      for(unsigned int tepi = 0 ; tepi < NB_OF_TESTING_EPISODES; ++tepi) {
	simulator.setPhase(Simulator::phase_type());
	cumul_episode_length += rl::episode::run(simulator, greedy_policy, MAX_EPISODE_LENGTH);
      }
      std::cout << "\r Episode " << episode << " : mean length over " << NB_OF_TESTING_EPISODES << " episodes is " << cumul_episode_length/double(NB_OF_TESTING_EPISODES) << std::string(10, ' ') << std::flush;
      
    }
  }
  catch(rl::exception::Any& e) {
    std::cerr << "Exception caught : " << e.what() << std::endl; 
  }
  std::cout << std::endl;
  
  save_value_function("q1.data");
  save_p("p1.data");
  
  gsl_vector_free(theta_q);
  gsl_vector_free(theta_p);
}
