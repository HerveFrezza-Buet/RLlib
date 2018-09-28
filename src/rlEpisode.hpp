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

#pragma once

#include <utility>

#include <rlException.hpp>
#include <rlTraits.hpp>

namespace rl {
    namespace episode {
        /**
         * This triggers an interaction from an action and returns a transition.
         * @param make_transition T = make_transition(s,a,r,ss);
         * @param make_terminal_transition T = make_terminal_transition(s,a,r);
         */
        template<typename SIMULATOR,
            typename ACTION,
            typename fctMAKE_TRANSITION,
            typename fctMAKE_TERMINAL_TRANSITION>
                auto perform(SIMULATOR& simulator,
                        const ACTION& action,
                        const fctMAKE_TRANSITION& make_transition,
                        const fctMAKE_TERMINAL_TRANSITION& make_terminal_transition) 
                -> decltype(make_terminal_transition(simulator.sense(),
                            action,
                            simulator.reward())) {
                    auto current = simulator.sense();
                    try {
                        simulator.timeStep(action);
                        return make_transition(current,action,simulator.reward(),simulator.sense());
                    }
                    catch(rl::exception::Terminal& e) {}
                    return make_terminal_transition(current,action,simulator.reward());
                }

        /**
         * This triggers an interaction from a policy and returns a transition.
         * @param make_transition T = make_transition(s,a,r,ss);
         * @param make_terminal_transition T = make_terminal_transition(s,a,r);
         */
        template<typename SIMULATOR,
            typename POLICY,
            typename fctMAKE_TRANSITION,
            typename fctMAKE_TERMINAL_TRANSITION>
                auto interaction(SIMULATOR& simulator,
                        const POLICY& policy,
                        const fctMAKE_TRANSITION& make_transition,
                        const fctMAKE_TERMINAL_TRANSITION& make_terminal_transition) 
                -> decltype(make_terminal_transition(simulator.sense(),
                            policy(simulator.sense()),
                            simulator.reward())) {
                    auto current = simulator.sense();
                    auto      a  = policy(current);
                    try {
                        simulator.timeStep(a);
                        return make_transition(current,a,simulator.reward(),simulator.sense());
                    }
                    catch(rl::exception::Terminal& e) {}
                    return make_terminal_transition(current,a,simulator.reward());
                }

        /**
         * This triggers an interaction from a policy. From this transition, the critic
         * learning occurs. This function is dedicated to be used with 
         * successive transitions, since it must be given (s,a) of the
         * last transition performed, in order to avoid a useless call to  policy(s). 
         * The rl::exception::Terminal exception is raised in case of terminal transition.
         * @param s The current state, i.e. s = simulator.sense()
         * @param a The action chosen by the policy, i.e. a = policy(s)
         * @return A s',a' pair, or raises an exception if a terminal transition is reached.
         */
        template<typename SIMULATOR,typename POLICY,
            typename STATE, typename ACTION,
            typename SRS_CRITIC>
                typename std::enable_if_t<rl::traits::is_srs_critic<SRS_CRITIC, STATE>::value, std::pair<STATE,ACTION> >
                adaptation(SIMULATOR& simulator,
                        const POLICY& policy,
                        SRS_CRITIC& critic,
                        const STATE& s,
                        const ACTION& a) {
                    try {
                        simulator.timeStep(a);
                        auto next = simulator.sense();
                        auto res = std::make_pair(next,policy(next));
                        critic.learn(s,simulator.reward(),next);
                        return res;
                    }
                    catch(rl::exception::Terminal& e) { 
                        critic.learn(s,simulator.reward());
                        throw e;
                    }
                }

        /**
         * This triggers an interaction from a policy. From this transition, the critic
         * learning occurs. This function is dedicated to be used with 
         * successive transitions, since it must be given (s,a) of the
         * last transition performed, in order to avoid a useless call to  policy(s). 
         * The rl::exception::Terminal exception is raised in case of terminal transition.
         * @param s The current state, i.e. s = simulator.sense()
         * @param a The action chosen by the policy, i.e. a = policy(s)
         * @return A s',a' pair, or raises an exception if a terminal transition is reached.
         */
        template<typename SIMULATOR,typename POLICY,
            typename STATE, typename ACTION,
            typename SARS_CRITIC>
                typename std::enable_if_t<rl::traits::is_sars_critic<SARS_CRITIC, STATE, ACTION>::value, std::pair<STATE,ACTION> >
                adaptation(SIMULATOR& simulator,
                        const POLICY& policy,
                        SARS_CRITIC& critic,
                        const STATE& s,
                        const ACTION& a) {
                    try {
                        simulator.timeStep(a);
                        auto next = simulator.sense();
                        auto res = std::make_pair(next,policy(next));
                        critic.learn(s,a,simulator.reward(),next);
                        return res;
                    }
                    catch(rl::exception::Terminal& e) { 
                        critic.learn(s,a,simulator.reward());
                        throw e;
                    }
                }

        /**
         * This triggers an interaction from a policy. From this transition, the critic
         * learning occurs. This function is dedicated to be used with 
         * successive transitions, since it must be given (s,a) of the
         * last transition performed, in order to avoid a useless call to  policy(s). 
         * The rl::exception::Terminal exception is raised in case of terminal transition.
         * @param s The current state, i.e. s = simulator.sense()
         * @param a The action chosen by the policy, i.e. a = policy(s)
         * @return A s',a' pair, or raises an exception if a terminal transition is reached.
         */
        template<typename SIMULATOR,typename POLICY,
            typename STATE, typename ACTION,
            typename SARSA_CRITIC>
                typename std::enable_if_t<rl::traits::is_sarsa_critic<SARSA_CRITIC, STATE, ACTION>::value, std::pair<STATE,ACTION> >
                adaptation(SIMULATOR& simulator,
                        const POLICY& policy,
                        SARSA_CRITIC& critic,
                        const STATE& s,
                        const ACTION& a) {
                    try {
                        simulator.timeStep(a);
                        auto next = simulator.sense();
                        auto res = std::make_pair(next,policy(next));
                        critic.learn(s,a,simulator.reward(),next,res.second);
                        return res;
                    }
                    catch(rl::exception::Terminal& e) { 
                        critic.learn(s,a,simulator.reward());
                        throw e;
                    }
                }



        /**
         * This triggers an interaction from a policy. From this transition, the critic
         * learning occurs. The rl::exception::Terminal exception is raised in case of terminal transition.
         */
        template<typename SIMULATOR,typename POLICY,
            typename SARSA_CRITIC,
            typename fctMAKE_TRANSITION,
            typename fctMAKE_TERMINAL_TRANSITION>
                void adaptation(SIMULATOR& simulator,
                        const POLICY& policy,
                        SARSA_CRITIC& critic) {

                    auto s = simulator.sense();
                    auto a = policy(s);
                    try {
                        simulator.timeStep(a);
                        auto next = simulator.sense();
                        critic.learn(s,a,simulator.reward(),next,policy(next));
                    }
                    catch(rl::exception::Terminal& e) { 
                        critic.learn(s,a,simulator.reward());
                        throw e;
                    }
                }

        /**
         * This runs an episode. 
         * @param max_episode_duration put a null number to run the episode without length limitation.
         * @return the actual episode length.
         */
        template<typename SIMULATOR,typename POLICY>
            unsigned int run(SIMULATOR& simulator,
                    const POLICY& policy,
                    unsigned int max_episode_duration) {
                unsigned int length=0;
                try {
                    do {
                        ++length;
                        simulator.timeStep(policy(simulator.sense()));
                    } while(length != max_episode_duration);
                }
                catch(rl::exception::Terminal& e) {}
                return length;
            }

        /**
         * This reads the transitions and fills an output iterator.
         * @param max_episode_duration put a null or negative number to run the episode without length limitation.
         * @return the actual episode length.
         * @param out an output iterator
         * @param make_transition *(out++) = make_transition(s,a,r,ss);
         * @param make_terminal_transition *(out++) = make_terminal_transition(s,a,r);

*/
        template<typename SIMULATOR,typename POLICY,typename OUTPUT_ITER,
            typename fctMAKE_TRANSITION,
            typename fctMAKE_TERMINAL_TRANSITION>
                unsigned int run(SIMULATOR& simulator,
                        const POLICY& policy,
                        OUTPUT_ITER out,
                        const fctMAKE_TRANSITION& make_transition,
                        const fctMAKE_TERMINAL_TRANSITION& make_terminal_transition,
                        unsigned int max_episode_duration) {
                    unsigned int length=0;
                    auto s = simulator.sense();
                    auto a = policy(s);
                    try {
                        do {
                            ++length;
                            simulator.timeStep(a);
                            auto s_ = simulator.sense();
                            *(out++) = make_transition(s,a,simulator.reward(),s_);
                            s = s_;
                            a = policy(s);
                        } while(length != max_episode_duration);
                    }
                    catch(rl::exception::Terminal& e) { 
                        *(out++) = make_terminal_transition(s,a,simulator.reward());
                    }
                    return length;
                }

        /**
         * This learn from the transitions.
         * @param max_episode_duration put a null number to run the episode without length limitation.
         * @return the actual episode length.
         */
        template<typename SIMULATOR,typename POLICY,
            typename CRITIC>
                unsigned int learn(SIMULATOR& simulator,
                        const POLICY& policy,
                        CRITIC& critic,
                        unsigned int max_episode_duration) {
                    unsigned int length=0;
                    auto s  = simulator.sense();
                    auto sa = std::make_pair(s,policy(s));
                    try {
                        do {
                            ++length;
                            sa = rl::episode::adaptation(simulator,policy,critic,sa.first,sa.second);
                        } while(length != max_episode_duration);
                    }
                    catch(rl::exception::Terminal& e) {}
                    return length;
                }

        /**
         * This reads the transitions, learn from it, and fills an output iterator.
         * @param max_episode_duration put a null or negative number to run the episode without length limitation.
         * @return the actual episode length.
         * @param out an output iterator
         * @param make_transition *(out++) = make_transition(s,a,r,ss);
         * @param make_terminal_transition *(out++) = make_terminal_transition(s,a,r);
         */
        template<typename SIMULATOR,
            typename POLICY,
            typename SARSA_CRITIC,
            typename OUTPUT_ITER,
            typename fctMAKE_TRANSITION,
            typename fctMAKE_TERMINAL_TRANSITION>
                unsigned int learn(SIMULATOR& simulator,
                        const POLICY& policy,
                        SARSA_CRITIC& critic,
                        OUTPUT_ITER out,
                        const fctMAKE_TRANSITION& make_transition,
                        const fctMAKE_TERMINAL_TRANSITION& make_terminal_transition,
                        unsigned int max_episode_duration) {
                    unsigned int length=0;
                    auto s = simulator.sense();
                    auto sa = std::make_pair(s,policy(s));
                    try {
                        do {
                            ++length;
                            auto sa_ = rl::episode::adaptation(simulator,policy,critic,sa.first,sa.second);
                            auto s_ = simulator.sense();
                            *(out++) = make_transition(sa.first,sa.second,simulator.reward(),sa_.first,sa_.second);
                            sa = sa_;
                        } while(length != max_episode_duration);
                    }
                    catch(rl::exception::Terminal& e) { 
                        *(out++) = make_terminal_transition(sa.first,sa.second,simulator.reward());
                    }
                    return length;
                }
    }		  
}
