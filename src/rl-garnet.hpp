/*   This file is part of rl-lib
 *
 *   Copyright (C) 2014,  Supelec
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

#pragma once

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <iterator>
#include <utility>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <list>
#include <rlAlgo.hpp>
#include <rlEpisode.hpp>
#include <rlException.hpp>

namespace rl {
  namespace problem {
    namespace garnet {

      class BadAction : public rl::exception::Any {
      public:
	BadAction(std::string comment) 
	  : Any(std::string("Bad action performed : ")+comment) {} 
      };
      
      // Garnet parameters
      class DefaultParam {
      public:
	inline static int num_states  (void)                  { return    10;}
	inline static int num_actions (void)                  { return     1;}
	inline static int branching   (void)                  { return     1;}
      };


      /**
       * Garnet simulator  
       * @author <a href="mailto:Jeremy.Fix@supelec.fr">Jeremy.Fix@supelec.fr</a>
       */
      template<typename GARNET_PARAM>
      class Simulator {

      public:

	typedef unsigned int       phase_type;
	typedef unsigned int       observation_type;
	typedef unsigned int       action_type;
	typedef double             reward_type;

      private:
	phase_type current_phase;
	reward_type* rewards;
	std::list< std::pair<unsigned int, double> >* transition_probabilities;

	unsigned int ns, na, nb;
      public:
	
	Simulator(void) {
	  ns = GARNET_PARAM::num_states();
	  na = GARNET_PARAM::num_actions();
	  nb = GARNET_PARAM::branching();

	  current_phase = ns * (std::rand() / (RAND_MAX-1.));

	  rewards = new reward_type[ns];
	  memset(rewards, 0, ns*sizeof(double));

	  transition_probabilities = new std::list< std::pair<unsigned int, double> >[ns*na];

	  std::vector<phase_type> next_states;
	  next_states.clear();
	  for(unsigned int i = 0 ; i < ns ; ++i)
	    next_states.push_back(i);
	  double* trans_prob = new double[nb];

	  // The insertion in the transition probabilites will be sorted
	  // we create the comparison function
	  auto compare_elements = [](const std::pair<unsigned int, double>& first,
				     const std::pair<unsigned int, double>& second) -> bool {
	    return first.first < second.first;
	  };

	  for(unsigned int s = 0 ; s < ns ; ++s) {
	    for(unsigned int a = 0 ; a  < na ; ++a) {
	      // Shuffle the arrival states (we will take the first nb elements)
	      std::random_shuffle(next_states.begin(), next_states.end());
	      // Generate the transition probabilities
	      double sum = 0.0;
	      for(unsigned int k = 0 ; k < nb ; ++k) {
		trans_prob[k] = std::rand()/(RAND_MAX-1.);
		sum += trans_prob[k];
	      }
	      for(unsigned int k = 0 ; k < nb ; ++k)
		trans_prob[k] /= sum;

	      // We now fill in the transition probabilities
	      // the elements are sorted by increasing arrival state number
	      for(unsigned int k = 0 ; k < nb ; ++k)
		transition_probabilities[s*na + a].push_back(std::make_pair(next_states[k], trans_prob[k]));
	      transition_probabilities[s*na + a].sort(compare_elements);
	    }
	  }

	  // Generation of the reward
	  for(unsigned int s = 0 ; s < ns ; ++s)
	    rewards[s] = std::rand()/(RAND_MAX-1.);
	}

	~Simulator(void) {
	  delete[] rewards;
	  delete[] transition_probabilities;
	}

	const observation_type&	sense (void) const {
	  return current_phase;
	}

	void timeStep (const action_type &a) {
	  if(a < 0 || a >= na) {
	    std::ostringstream ostr;
	    ostr.str("");
	    ostr << "Action " << a << " not in [0; " << (na-1) << "]";
	    throw BadAction(ostr.str());
	  }

	  auto transition_proba = transition_probabilities[current_phase*na + a];
	  // We have a list of arrival states with their probabilities which sum to 1
	  double p = std::rand() / (RAND_MAX-1.);
	  auto piter = transition_proba.begin();
	  auto piter_end = transition_proba.end();
	  double sum = 0.0;
	  for(; piter != piter_end; ++piter) {
	    sum += piter->second;
	    if(p <= sum) 
	      break;
	  }
	  current_phase = piter->first;
	}

	reward_type reward (void) const {
	  return rewards[current_phase];
	}

	void draw(bool verbose=true) const {

	  std::ofstream outfile("graph.gv");
	  outfile << "digraph garnet {" << std::endl
		  << "node [shape = doublecircle] ; S" << current_phase << ";" << std::endl
		  << "node [shape = circle] ; " << std::endl;
	  for(unsigned int s = 0 ; s < ns ; ++s) {
	    for(unsigned int a = 0 ; a < na ; ++a) {
	      auto transition_proba = transition_probabilities[s*na+a];
	      for(auto& tpi: transition_proba) 
		outfile << "S" << s << " -> S" << tpi.first << " [ label = \"" << tpi.second << "\" , colorscheme=paired12, color=" << a+1 << " ];" << std::endl;
	    }
	  }
	  outfile <<"}" << std::endl;
	  if(verbose) std::cout << "graph.gv generated, " << std::endl
				<< "process it using the GraphViz tools, e.g. " << std::endl
				<< "dot -Tpng graph.gv > graph.png " << std::endl;
	}


	void print(void) const {
	  std::cout << "Generated garnet with " << std::endl
		    << "ns = " << ns << "; na = " << na << "; nb = " << nb << std::endl;

	  std::cout << "Rewards : " << std::endl;
	  for(unsigned int k = 0 ; k < ns ; ++k)
	    std::cout << "State " << k << " : " << rewards[k] << std::endl;

	  std::cout << "Transitions : " << std::endl;
	  unsigned int width = 7; // set it to an odd value
	  for(unsigned int a = 0 ; a < na ; ++a) {
	    std::cout << "   Action " << a << std::endl;
	    std::cout << std::string(width+1, ' ') ;
	    for(unsigned int s = 0 ; s < ns ; ++s) {
	      std::ostringstream ostr;
	      ostr.str("");
	      ostr << "" << s;
	      std::string sstr = ostr.str();
	      std::cout << std::string((width-sstr.size())/2, ' ') 
			<< sstr 
			<< std::string((width-sstr.size())/2, ' ') << " ";
	    }
	    std::cout << std::endl;
	    
	    for(unsigned int s = 0 ; s < ns ; ++s) {
	      std::cout << std::setw(width) << std::setfill(' ') << s << " ";
	      auto probas = transition_probabilities[s*na + a];
	      auto probas_iter = probas.begin();
	      auto probas_end = probas.end();
	      for(unsigned int k = 0 ; k < ns ; ++k) {
		if(probas_iter->first == k) {
		  std::cout.precision(3);
		  std::cout << std::setw(width) << std::setfill(' ') << probas_iter->second << " ";
		  ++probas_iter;
		}
		else 
		  std::cout << std::setw(width) << std::setfill(' ') << "0" << " ";
	      }
	      std::cout << std::endl;
             }
	    std::cout << std::endl;
	  }
	}

      };

    }
  }
}
