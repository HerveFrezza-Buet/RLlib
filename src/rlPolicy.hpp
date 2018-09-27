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
                class Greedy {
                    private:

                        Q q;
                        ACTION_ITERATOR begin,end;

                    public:

                        Greedy(const Q& q_function,
                                const ACTION_ITERATOR& action_begin,
                                const ACTION_ITERATOR& action_end) 
                            : q(q_function),
                            begin(action_begin),
                            end(action_end){}
                        Greedy(const Greedy<Q,ACTION_ITERATOR>& cp) 
                            : q(cp.q), begin(cp.begin), end(cp.end) {}

                        Greedy<Q,ACTION_ITERATOR>& operator=(const Greedy<Q,ACTION_ITERATOR>& cp) {
                            if(&cp != this) {
                                q     = cp.q;
                                begin = cp.begin;
                                end   = cp.end;
                            }
                            return *this;
                        }


                        template<typename STATE>
                            typename std::remove_reference<decltype(*begin)>::type operator()(const STATE& s) const {
                                return rl::argmax(std::bind(q,s,std::placeholders::_1),begin,end).first;
                            }
                };

        template<typename Q,
            typename ACTION_ITERATOR>
                Greedy<Q,ACTION_ITERATOR> greedy(const Q& q_function,
                        const ACTION_ITERATOR& action_begin,
                        const ACTION_ITERATOR& action_end) {
                    return Greedy<Q,ACTION_ITERATOR>(q_function,action_begin,action_end);
                }



        /**
         * @short This builds a epsilon-greedy policy from an existing Q(S,A) function.
         */
        template<typename Q,
            typename ACTION_ITERATOR,
            typename RANDOM_GENERATOR>
                class EpsilonGreedy {
                    private:

                        Q q;
                        ACTION_ITERATOR begin,end;
                        RANDOM_GENERATOR* gen;

                    public:

                        double epsilon;

                        EpsilonGreedy() = delete;
                        EpsilonGreedy(const Q& q_function,
                                double eps,
                                const ACTION_ITERATOR& action_begin,
                                const ACTION_ITERATOR& action_end,
                                RANDOM_GENERATOR& gen) 
                            : q(q_function),
                            begin(action_begin),
                            end(action_end),
                            gen(&gen),
                            epsilon(eps) {}

                        EpsilonGreedy(const EpsilonGreedy<Q,ACTION_ITERATOR,RANDOM_GENERATOR>& cp) = default;
                        EpsilonGreedy<Q,ACTION_ITERATOR,RANDOM_GENERATOR>& operator=(const EpsilonGreedy<Q,ACTION_ITERATOR,RANDOM_GENERATOR>& cp) = default;

                        template<typename STATE>
                                typename std::remove_reference<decltype(*begin)>::type operator()(const STATE& s) const {
                                    std::bernoulli_distribution dis(epsilon);
                                    if(dis(*gen)) { 
                                        typename std::remove_reference<decltype(*begin)>::type selected_value;
                                        std::sample(begin, end, &selected_value, 1, *gen);
                                        return selected_value;
                                    }
                                    return rl::argmax(std::bind(q,s,std::placeholders::_1),begin,end).first;
                                }
                };
        template<typename Q,
            typename ACTION_ITERATOR,
            typename RANDOM_GENERATOR>
                EpsilonGreedy<Q,ACTION_ITERATOR,RANDOM_GENERATOR> epsilon_greedy(const Q& q_function,
                        double epsilon,
                        const ACTION_ITERATOR& action_begin,
                        const ACTION_ITERATOR& action_end,
                        RANDOM_GENERATOR& gen) {
                    return EpsilonGreedy<Q,ACTION_ITERATOR,RANDOM_GENERATOR>(q_function,epsilon,action_begin,action_end, gen);
                }

        /**
         * @short This builds a random policy.
         */
        template<typename ACTION_ITERATOR,
            typename RANDOM_GENERATOR>
                class Random {
                    private:

                        ACTION_ITERATOR   begin,end;
                        RANDOM_GENERATOR* gen;

                    public:

                        Random() = delete;
                        Random(const ACTION_ITERATOR& action_begin,
                                const ACTION_ITERATOR& action_end,
                                RANDOM_GENERATOR& gen) 
                            : begin(action_begin),
                            end(action_end),
                            gen(&gen) {}

                        Random(const Random<ACTION_ITERATOR,RANDOM_GENERATOR>& cp) = default; 
                        Random<ACTION_ITERATOR,RANDOM_GENERATOR>& operator=(const Random<ACTION_ITERATOR,RANDOM_GENERATOR>& cp) = default;
                                
                        template<typename STATE>
                            typename std::remove_reference<decltype(*begin)>::type operator()(const STATE& s) const {
                                typename std::remove_reference<decltype(*begin)>::type selected_value;
                                std::sample(begin, end, &selected_value, 1, *gen);
                                return selected_value;
                            }
                };

        template<typename ACTION_ITERATOR,
            typename RANDOM_GENERATOR>
                Random<ACTION_ITERATOR, RANDOM_GENERATOR> random(const ACTION_ITERATOR& action_begin,
                        const ACTION_ITERATOR& action_end,
                        RANDOM_GENERATOR& gen) {
                    return Random<ACTION_ITERATOR,RANDOM_GENERATOR>(action_begin,action_end, gen);
                }



        /**
         * @short This builds a softmax policy from an existing Q(S,A) function.
         */
        template<typename Q,
            typename ACTION_ITERATOR,
            typename RANDOM_GENERATOR>
                class SoftMax {
                    private:

                        Q q;
                        ACTION_ITERATOR begin,end;
                        RANDOM_GENERATOR* gen;

                    public:

                        double temperature;

                        SoftMax() = delete;
                        SoftMax(const Q& q_function,
                                double temp,
                                const ACTION_ITERATOR& action_begin,
                                const ACTION_ITERATOR& action_end,
                                RANDOM_GENERATOR& gen) 
                            : q(q_function),
                            temperature(temp),
                            begin(action_begin),
                            end(action_end),
                            gen(&gen) {}

                        SoftMax(const SoftMax<Q,ACTION_ITERATOR,RANDOM_GENERATOR>& cp) = default; 
                        SoftMax<Q,ACTION_ITERATOR, RANDOM_GENERATOR>& operator=(const SoftMax<Q,ACTION_ITERATOR,RANDOM_GENERATOR>& cp) = default;

                        template<typename STATE>
                            typename std::remove_reference<decltype(*begin)>::type operator()(const STATE& s) const {
                                return rl::random::softmax(std::bind(q,s,std::placeholders::_1),temperature,begin,end, *gen);
                            }
                };

        template<typename Q,
            typename ACTION_ITERATOR,
            typename RANDOM_GENERATOR>
                SoftMax<Q,ACTION_ITERATOR, RANDOM_GENERATOR> softmax(const Q& q_function,
                        double temperature,
                        const ACTION_ITERATOR& action_begin,
                        const ACTION_ITERATOR& action_end,
                        RANDOM_GENERATOR& gen) {
                    return SoftMax<Q,ACTION_ITERATOR, RANDOM_GENERATOR>(q_function,temperature,action_begin,action_end, gen);
                }    
    }
}
