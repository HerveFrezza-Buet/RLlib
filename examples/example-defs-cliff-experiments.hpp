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

using namespace std::placeholders;

template<typename CRITIC,typename Q>
void make_experiment(CRITIC& critic,
		     const Q& q) {
  Param         param;
  Simulator     simulator(param);
  auto          action_begin     = rl::enumerator<A>(rl::problem::cliff_walking::actionNorth);
  auto          action_end       = action_begin + rl::problem::cliff_walking::actionSize;
  auto          state_begin      = rl::enumerator<S>(Cliff::start);
  auto          state_end        = state_begin + Cliff::size;
  auto          learning_policy  = rl::policy::epsilon_greedy(q,paramEPSILON,
							      action_begin,action_end);
  auto          test_policy      = rl::policy::greedy(q,action_begin,action_end);
  int           episode,frame;
  int           episode_length;

  std::array<bool,Cliff::size> visited;
  
  std::cout << std::endl << std::endl;
  for(episode = 0, frame = 0;episode < NB_EPISODES; ++episode) {

    std::cout << "running episode " << std::setw(6) << episode+1
	      << "/" << NB_EPISODES
	      << "    \r" << std::flush;

    simulator.restart();
    auto actual_episode_length = rl::episode::learn(simulator,
						    learning_policy,critic,
						    0);

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


  std::string command;
  int command_res;
  
  command = "find . -name \"rllib-*.ppm\" -exec convert \\{} -filter Box -resize 192x64 -quality 100 \\{}.jpg \\;";
  std::cout << "Executing : " << command << std::endl;
  command_res = system(command.c_str());

  command = "ffmpeg -i rllib-%06d.ppm.jpg -r 5 rllib.avi";
  std::cout << "Executing : " << command << std::endl;
  command_res = system(command.c_str());

  command = "find . -name \"rllib-*.ppm\" -exec rm \\{} \\;";
  std::cout << "Executing : " << command << std::endl;
  command_res = system(command.c_str());

  command = "find . -name \"rllib-*.ppm.jpg\" -exec rm \\{} \\;";
  std::cout << "Executing : " << command << std::endl;
  command_res = system(command.c_str());
}
