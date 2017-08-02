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

// Experiment : Cliff-walking
// Architecture : Tabular coding of the state space with linear value functions and policy
// Critic : TD-V
// Policy : Softmax
// Learner : One-step Actor-Critic

#include <rl.hpp>

#define NB_EPISODES    3000

#define paramGAMMA     .99
#define paramALPHA_V   .05
#define paramALPHA_P   .01

// The problem on which to train a controller
using Cliff     = rl::problem::cliff_walking::Cliff<20,6>;
using Param     = rl::problem::cliff_walking::Param;
using Simulator = rl::problem::cliff_walking::Simulator<Cliff,Param>;

using S = Simulator::observation_type;
using A = Simulator::action_type;

double fct_v(const gsl_vector* theta, const S& s) {
  return gsl_vector_get(theta, s);
}
void fct_grad_v(const gsl_vector* theta, gsl_vector* grad_v, const S& s) {
  gsl_vector_set_basis(grad_v, s);
}

double fct_p(const gsl_vector* theta_p, unsigned int nb_states, const S& s, const A& a) {
  return gsl_vector_get(theta_p, a * nb_states + s);
}

template<typename AITER>
std::map<A, double> get_action_probabilities(AITER action_begin,
					     AITER action_end,
					     unsigned int nb_states,
					     unsigned int nb_actions,
					     const gsl_vector* theta_p,
					     const S& s) {
  std::map<A, double> probaActions;
  auto aiter = action_begin;
  double psum = 0.0;
  while(aiter != action_end) {
    probaActions[*aiter] = exp(fct_p(theta_p, nb_states, s, *aiter));
    psum += probaActions[*aiter];
    ++aiter;
  }
  for(auto& p: probaActions)
    p.second /= psum;
  return probaActions;
};

template<typename AITER>
void fct_grad_log_p(AITER action_begin, AITER action_end,
		    unsigned int nb_states,
		    unsigned int nb_actions,
		    const gsl_vector* theta_p,
		    gsl_vector* grad_log_p,
		    const S& s, const A& a) {
  gsl_vector_set_zero(grad_log_p);
    
  // We need to get the probabilities of the actions
  auto probaActions = get_action_probabilities(action_begin, action_end, nb_states, nb_actions, theta_p, s);

  // And we can then compute the gradient of ln(Pi)
  auto aiter = action_begin;
  while(aiter != action_end) {
    gsl_vector_set(grad_log_p, nb_states * std::distance(action_begin, aiter) + s, ((*aiter)==a) - probaActions[*aiter]);
    ++aiter;
  }
}

std::string action_to_string(const A& a) {
  if(a == rl::problem::cliff_walking::actionNorth)
    return "↑";
  else if(a == rl::problem::cliff_walking::actionEast)
    return "→";
  else if(a == rl::problem::cliff_walking::actionSouth)
    return "↓" ;
  else
    return "←";
}

template<typename AITER, typename SCORES>
void print_greedy_policy(AITER action_begin, AITER action_end,
			 unsigned int nb_states, unsigned int nb_actions,
			 const SCORES& scores, const gsl_vector* theta_p) {
  std::cout << "Greedy policy : " << std::endl;
  auto policy = rl::policy::greedy(scores, action_begin, action_end);
  for(int i = Cliff::width ; i > 0; --i) {
    for(int j = 0 ; j < Cliff::length ; ++j) {
      int state_idx = 1 + (i-1) * Cliff::length + j;
      auto a = policy(state_idx);
      std::cout << "   " << action_to_string(a) << "   ";
    }
    std::cout << std::endl;
    for(int j = 0 ; j < Cliff::length ; ++j) {
      int state_idx = 1 + (i-1) * Cliff::length + j;
      auto probaActions = get_action_probabilities(action_begin, action_end, nb_states, nb_actions, theta_p, state_idx);
      auto a = policy(state_idx);
      std::cout << " " << std::setfill(' ') << std::setw(5) << std::setprecision(3) << probaActions[a] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "   " << action_to_string(policy(0)) << "   ";
  std::cout << std::string(7 * (Cliff::length-2), ' ');
  std::cout << "   " << action_to_string(policy(Cliff::width*Cliff::length+1)) << "   " << std::endl;
  auto probaActions = get_action_probabilities(action_begin, action_end, nb_states, nb_actions, theta_p, 0);
  std::cout << " " << std::setfill(' ') << std::setw(5) << std::setprecision(3) << probaActions[policy(0)] << " ";
  std::cout << std::string(7 * (Cliff::length-2), ' ');
  std::cout << " " << std::setfill(' ') << std::setw(5) << std::setprecision(3) << probaActions[policy(Cliff::width*Cliff::length+1)] << " ";
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  rl::random::seed(time(0));
  
  // 1) Instantiate the simulator
  Param param;
  Simulator simulator(param);

  unsigned int nb_states = Cliff::size;
  unsigned int nb_actions = rl::problem::cliff_walking::actionSize;
  
  ////////////// Actor/Critic
  // 2a) Instantiate the Critic, we here use TD-V
  gsl_vector* theta_v = gsl_vector_calloc(nb_states);  
  auto critic = rl::gsl::td<S>(theta_v,
			       paramGAMMA, paramALPHA_V,
			       fct_v, fct_grad_v);

  // 2b) Instantiate the Actor
  gsl_vector* theta_p = gsl_vector_calloc(nb_states*nb_actions);
  auto scores = std::bind(fct_p, theta_p, nb_states, std::placeholders::_1, std::placeholders::_2);
  auto action_begin = rl::enumerator<A>(rl::problem::cliff_walking::actionNorth);
  auto action_end = action_begin + rl::problem::cliff_walking::actionSize;
  auto policy = rl::policy::softmax(scores, 1.0,
				    action_begin, action_end);

  // 2c) And finally the actor/critic 
  auto grad = std::bind(fct_grad_log_p<decltype(action_begin)>, action_begin, action_end, nb_states, nb_actions, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
  auto actor_critic = rl::gsl::ActorCritic::one_step<S, A>(critic,
							   theta_p,
							   paramALPHA_P,
							   grad);

  // 3) run NB_EPISODES episodes
  unsigned int episode;
  std::cout << "Learning " << std::endl;
  for(episode = 0 ;episode < NB_EPISODES; ++episode) {
    std::cout << '\r' << "Episode " << episode << std::flush;
    simulator.restart();
    rl::episode::learn(simulator, policy, actor_critic, 0);
  }
  std::cout << std::endl;

  std::cout << "Testing the learned policy" << std::endl;
  // After this training phase, we test the policy
  unsigned int nb_test_episodes = 1000;
  double cum_length = 0.0;
  for(unsigned int i = 0 ; i < nb_test_episodes; ++i) {
    simulator.restart();
    cum_length += rl::episode::run(simulator, policy, 0);
  }
  std::cout << "The mean length of "<< nb_test_episodes
	    <<" testing episodes is " << cum_length / double(nb_test_episodes) << std::endl;

  // And let us display the action probabilities for the first state :
  std::cout << "The probabilities of the actions of the learned controller, in the start state are :" << std::endl;

  
  auto proba = get_action_probabilities(action_begin, action_end, nb_states, nb_actions, theta_p, 0);
  for(auto ait = action_begin; ait != action_end; ++ait)
    std::cout << "P(" << action_to_string(*ait) << "/s=start) = " << proba[*ait] << std::endl;

  print_greedy_policy(action_begin, action_end, nb_states, nb_actions, scores, theta_p);  

  gsl_vector_free(theta_v);
  gsl_vector_free(theta_p);
}
