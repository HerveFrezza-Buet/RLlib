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


#include <functional>
#include <array>
#include <algorithm>  
#include <cstdlib>  

#define NB_EPISODES            10000
#define MAX_EPISODE_DURATION     100
#define FRAME_PERIOD              25
#define MIN_V                    -50


// This is an output iterator that notifies the visited states.
// The use within the run function is like
//
// VisitNotifier v;
//
// *(v++) = transition(s,a,r,s');
//
// Here, we will use transition(s,a,r,s') = s, see the lambda
// functions given to the run function.
class VisitNotifier {
public:

  std::array<bool,Cliff::size>& visited;

  VisitNotifier(std::array<bool,Cliff::size>& v) 
    : visited(v) {
    std::fill(visited.begin(),visited.end(),false);
  }

  VisitNotifier(const VisitNotifier& cp) : visited(cp.visited){}

  VisitNotifier& operator*()     {return *this;}
  VisitNotifier& operator++(int) {return *this;}        

  void operator=(S s) {
    visited[s] = true;
  }
};



std::string action_to_string(const A& a) {
  if(a == rl::problem::cliff_walking::Action::actionNorth)
    return "↑";
  else if(a == rl::problem::cliff_walking::Action::actionEast)
    return "→";
  else if(a == rl::problem::cliff_walking::Action::actionSouth)
    return "↓" ;
  else
    return "←";
}

template<typename AITER, typename SCORES>
double normalized_score(const S& s, const A& a,
			AITER action_begin, AITER action_end,
			const SCORES& scores) {
  double score = exp(scores(s, a));
  double Z = 0.0;
  for(auto ai = action_begin; ai != action_end; ++ai)
    Z += exp(scores(s, *ai));
  return score / Z;
}

template<typename AITER, typename SCORES>
void print_greedy_policy(AITER action_begin, AITER action_end,
			 const SCORES& scores) {
  std::cout << "The greedy policy is depicted below. For each state, the greedy action        " << std::endl
	    << "is displayed with a normalized score : exp(Q(s,a_greedy)) / sum_a exp(Q(s, a))" << std::endl
	    << std::endl;

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
      auto a = policy(state_idx);
      std::cout << " " << std::setfill(' ') << std::setw(5) << std::setprecision(3) << normalized_score(state_idx, a, action_begin, action_end, scores) << " ";
    }
    std::cout << std::endl;
  }

  int s_start = 0;
  int s_end = Cliff::width*Cliff::length+1;
  auto a_start = policy(s_start);
  auto a_end   = policy(s_end);
  
  std::cout << "   " << action_to_string(a_start) << "   ";
  std::cout << std::string(7 * (Cliff::length-2), ' ');
  std::cout << "   " << action_to_string(a_end) << "   " << std::endl;

  std::cout << " " << std::setfill(' ') << std::setw(5) << std::setprecision(3) << normalized_score(s_start, a_start, action_begin, action_end, scores) << " ";
  std::cout << std::string(7 * (Cliff::length-2), ' ');
  std::cout << " " << std::setfill(' ') << std::setw(5) << std::setprecision(3) << normalized_score(s_end, a_end, action_begin, action_end, scores) << " ";
  std::cout << std::endl;
}


void execute_command(const std::string& command) {
    int status = std::system(command.c_str());
    if(status != EXIT_SUCCESS) 
        throw std::runtime_error(std::string("Errors raised when executing '" + command + "'"));
}

using namespace std::placeholders;

template<typename CRITIC,typename Q, typename RANDOM_GENERATOR>
void make_experiment(CRITIC& critic,
		     const Q& q,
             RANDOM_GENERATOR& gen) {
  Param         param;
  Simulator     simulator(param);
  auto          action_begin     = rl::enumerator<A>(rl::problem::cliff_walking::Action::actionNorth);
  auto          action_end       = action_begin + rl::problem::cliff_walking::actionSize;
  //auto          state_begin      = rl::enumerator<S>(Cliff::start);
  //auto          state_end        = state_begin + Cliff::size;
  double        epsilon          = paramEPSILON;
  auto          learning_policy  = rl::policy::epsilon_greedy(q,epsilon,
							      action_begin,action_end, gen);
  auto          test_policy      = rl::policy::greedy(q,action_begin,action_end);
  int           episode,frame;

  std::array<bool,Cliff::size> visited;
  
  std::cout << std::endl << std::endl;
  for(episode = 0, frame = 0;episode < NB_EPISODES; ++episode) {

    std::cout << "running episode " << std::setw(6) << episode+1
	      << "/" << NB_EPISODES
	      << "    \r" << std::flush;

    simulator.restart();
    rl::episode::learn(simulator, learning_policy,critic, MAX_EPISODE_DURATION);

    if(episode % FRAME_PERIOD == 0) {

      // Let us run an episode with a greedy policy and mark the
      // states as visited.
      VisitNotifier visit_notifier(visited);
      simulator.restart();
      rl::episode::run(simulator,
		       test_policy,
		       visit_notifier,
		       [](S s, A a, Reward r, S s_) -> S {return s;}, 
		       [](S s, A a, Reward r)       -> S {return s;}, 
		       MAX_EPISODE_DURATION);

      Cliff::draw_visited("rllib",frame++,
			  [&action_begin,&action_end,&q](S s) -> double {return rl::max(std::bind(q,s,_1),
											action_begin,
											action_end);}, // V(s) = max_a q(s,q)
			  [&visit_notifier](S s) -> bool {return visit_notifier.visited[s];},
			  MIN_V,0);
    }
  }

  std::cout << std::endl
	    << std::endl;

  
  print_greedy_policy(action_begin, action_end, q);
  
  std::string command;
  
  command = "find . -name \"rllib-*.ppm\" -exec convert \\{} -filter Box -resize 192x64 -quality 100 \\{}.jpg \\;";
  std::cout << "Executing : " << command << std::endl;
  execute_command(command.c_str());

  command = "ffmpeg -i rllib-%06d.ppm.jpg -r 5 rllib.avi";
  std::cout << "Executing : " << command << std::endl;
  execute_command(command.c_str());

  command = "find . -name \"rllib-*.ppm\" -exec rm \\{} \\;";
  std::cout << "Executing : " << command << std::endl;
  execute_command(command.c_str());

  command = "find . -name \"rllib-*.ppm.jpg\" -exec rm \\{} \\;";
  std::cout << "Executing : " << command << std::endl;
  execute_command(command.c_str());
}


