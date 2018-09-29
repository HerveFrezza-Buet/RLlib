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

#include <functional>
#include <type_traits>
#include <random>
#include <algorithm>
#include <memory>

#include <rlAlgo.hpp>

namespace rl {

    namespace policy {

        /**
         * @short This builds a greedy policy from an existing Q(S,A) function.
         */
        template<typename Q,
            typename ACTION_ITERATOR>
                auto greedy(const Q& q_function,
                        const ACTION_ITERATOR& action_begin,
                        const ACTION_ITERATOR& action_end) {
                    return [q_function, action_begin, action_end](const auto& s) {
                        return rl::argmax(std::bind(q_function,s,std::placeholders::_1),action_begin,action_end).first;
                    };
                }



        /**
         * @short This builds a epsilon-greedy policy from an existing Q(S,A) function.
         */
        template<typename Q,
            typename ACTION_ITERATOR,
            typename RANDOM_GENERATOR>
                auto epsilon_greedy(const Q& q_function,
                        double& epsilon,
                        const ACTION_ITERATOR& action_begin,
                        const ACTION_ITERATOR& action_end,
                        RANDOM_GENERATOR& gen) {
                    return [&gen,q_function,&epsilon,action_begin,action_end](const auto& s) -> typename std::remove_reference<decltype(*action_begin)>::type {
                        std::bernoulli_distribution dis(epsilon);
                        if(dis(gen)) { 
                            typename std::remove_reference<decltype(*action_begin)>::type selected_value;
                            std::sample(action_begin, action_end, &selected_value, 1, gen);
                            return selected_value;
                        }
                        return rl::argmax(std::bind(q_function,s,std::placeholders::_1),action_begin,action_end).first;
                    }; 
                }

        /**
         * @short This builds a random policy.
         */
        template<typename ACTION_ITERATOR,
            typename RANDOM_GENERATOR>
            auto random(const ACTION_ITERATOR& action_begin,
                    const ACTION_ITERATOR& action_end,
                    RANDOM_GENERATOR& gen) {
                return [&gen, action_begin, action_end](const auto& s) {
                    typename std::remove_reference<decltype(*action_begin)>::type selected_value;
                    std::sample(action_begin, action_end, &selected_value, 1, gen);
                    return selected_value;

                };
            }

        /**
         * @short This builds a softmax policy from an existing Q(S,A) function.
         */
        template<typename Q,
            typename ACTION_ITERATOR,
            typename RANDOM_GENERATOR>
                auto softmax(const Q& q_function,
                        double& temperature,
                        const ACTION_ITERATOR& action_begin,
                        const ACTION_ITERATOR& action_end,
                        RANDOM_GENERATOR& gen) {
                    return [&gen, q_function, &temperature, action_begin, action_end](const auto& s) {
                        return rl::random::softmax(std::bind(q_function, s ,std::placeholders::_1),temperature,action_begin, action_end, gen);
                    };
                }    
    }
}
