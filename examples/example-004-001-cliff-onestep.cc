// Experiment : Cliff-walking
// Architecture : Tabular coding of the state space with linear value functions and policy
// Learner : One-step Actor-Critic

#include <rl.hpp>

#define NB_EPISODES    3000

#define paramGAMMA     .95
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
std::vector<double> get_action_probabilities(AITER action_begin,
					     AITER action_end,
					     unsigned int nb_states,
					     unsigned int nb_actions,
					     const gsl_vector* theta_p,
					     const S& s) {
    std::vector<double> probaActions(nb_actions);
    auto aiter = action_begin;
    double psum = 0.0;
    auto piter = probaActions.begin();
    while(aiter != action_end) {
      *piter = exp(fct_p(theta_p, nb_states, s, *aiter));
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
		    unsigned int nb_states,
		    unsigned int nb_actions,
		    const gsl_vector* theta_p,
		    gsl_vector* grad_log_p,
		    const S& s, const A& a) {
  gsl_vector_set_zero(grad_log_p);
    
  // We need to get the probabilities of the actions
  auto probaActions = get_action_probabilities(action_begin, action_end, nb_states, nb_actions, theta_p, s);

  // And we can then compute the gradient of ln(Pi)
  auto piter = probaActions.begin();
  auto aiter = action_begin;
  while(aiter != action_end) {
    gsl_vector_set(grad_log_p, nb_states * std::distance(action_begin, aiter) + s, ((*aiter)==a) - (*piter));
    ++aiter;
    ++piter;
  }
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
  unsigned int step;
  A action;
  S state, next;
  double rew;

  std::cout << "Learning " << std::endl;
  for(episode = 0 ;episode < NB_EPISODES; ++episode) {
    simulator.restart();

    step = 0;
    std::cout << '\r' << "Episode " << episode << std::flush;
    
    state = simulator.sense();

    while(true) {
      action = policy(state);
      try {
	// The following may raise a Terminal state exception
	simulator.timeStep(action);
	
	rew = simulator.reward();
	next = simulator.sense();

	actor_critic.learn(state, action, rew, next);
	
	state = next;
	++step;
      }
      catch(rl::exception::Terminal& e) { 
	actor_critic.learn(state, action, simulator.reward());
	break;
      }
    }
  }
  std::cout << std::endl;

  std::cout << "Testing the learned policy" << std::endl;
  // After this training phase, we test the policy
  unsigned int nb_test_episodes = 1000;
  double cum_length = 0.0;
  for(unsigned int i = 0 ; i < nb_test_episodes; ++i) {
    simulator.restart();
    step = 0;
    state = simulator.sense();
    while(true) {
      action = policy(state);
      try {
	// The following may raise a Terminal state exception
	simulator.timeStep(action);
	state = simulator.sense();
	++step;
      }
      catch(rl::exception::Terminal& e) { 
	break;
      }
    }
    cum_length += step;
  }
  std::cout << "The mean length of "<< nb_test_episodes
	    <<" testing episodes is " << cum_length / double(nb_test_episodes) << std::endl;

  // And let us display the action probabilities for the first state :
  std::cout << "The probabilities of the actions of the learned controller, in the start state are :" << std::endl;

  
  auto proba = get_action_probabilities(action_begin, action_end, nb_states, nb_actions, theta_p, 0);
  std::cout << "P(North/s=start) = " << proba[rl::problem::cliff_walking::actionNorth] << std::endl;
  std::cout << "P(East/s=start) = " << proba[rl::problem::cliff_walking::actionEast] << std::endl;
  std::cout << "P(South/s=start) = " << proba[rl::problem::cliff_walking::actionSouth] << std::endl;
  std::cout << "P(West/s=start) = " << proba[rl::problem::cliff_walking::actionWest] << std::endl;
  

  gsl_vector_free(theta_v);
  gsl_vector_free(theta_p);
}
