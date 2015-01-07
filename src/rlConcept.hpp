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

namespace rl {


  namespace concept {
    /**
     * @short The simulator itself
     *
     * A simulator is the external world, or the dynamical process
     * that is controlled by the agent through successive actions. It
     * also includes the reward. Its current state is calle the
     * current phase, to avoid ambiguity with the state in
     * reinforcement learning. Indeed, the state space is a model use
     * by the agent to represent the phase of the simulator. From each
     * phase, an observation can be provided, as with sensors for a
     * real robotic system. In most simulated cases, phase and state
     * are the same, as well as observation if observable markovian
     * processes are used.
     */
    template <typename ACTION,typename OBSERVATION, typename REWARD>
    class Simulator {
    public:

      Simulator(void);
      ~Simulator(void);
      
      /**
       * This gives the observation corresponding to current phase.
       */
      const OBSERVATION& sense(void) const;

      /**
       * This triggers a transition to a new phase, consecutivly to action a. This call may raise a rl::exception::Terminal if some terminal state is reached.
       */
      void timeStep(const ACTION& a);

      /**
       * This gives the reward obtained from the last phase transition.
       */
      REWARD reward(void) const;
    };

    template<typename S,
	     typename A,
	     typename R>
    class SarsaCritic {
    public:
      /**
       * This updates the critic internal model from a terminal transition.
       */
      void learn(const S& s,
		 const A& a,
		 const R& r);
      /**
       * This updates the critic internal model from a non-terminal transition.
       */
      void learn(const S& s,
		 const A& a,
		 const R& r,
		 const S& s_,
		 const A& a_);
    };

    template<typename TRANSITION_ITER>
    class BatchCritic {
    public:
      /**
       * This learns from all transitions.
       */
      void operator()(const TRANSITION_ITER& begin,
		      const TRANSITION_ITER& end);
    };

  }
}
