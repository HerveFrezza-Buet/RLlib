/*   This file is part of rl-lib
 *
 *   Copyright (C) 2017,  CentraleSupelec
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
 *   Contact : Herve.Frezza-Buet@centralesupelec.fr Matthieu.Geist@centralesupelec.fr Jeremy.Fix@centralesupelec.fr
 *
 */

#include <rl.hpp>
#include <map>

#pragma once

namespace rl {
    namespace gsl {
        namespace ActorCritic {
            namespace Architecture {

                /**
                 * @short Tabular Actor-Critic architecture
                 */
                template<typename STATE, typename ACTION, typename RANDOM_GENERATOR>
                    class Tabular {
                        public:
                            using state_type = STATE;
                            using action_type = ACTION;

                        private:
                            unsigned int _nb_features;
                            std::function<unsigned int(const STATE&)> _state_to_idx;
                            unsigned int _nb_actions;
                            rl::enumerator<action_type> _action_begin;
                            rl::enumerator<action_type> _action_end;

                            class Critic {
                                public:
                                    gsl_vector* _params;

                                    Critic(unsigned int nb_features):
                                        _params(gsl_vector_alloc(nb_features)) {
                                            gsl_vector_set_zero(_params);
                                        }

                                    ~Critic() {
                                        gsl_vector_free(_params);
                                    }

                                    double operator()(unsigned int state_idx) const {
                                        return gsl_vector_get(_params, state_idx);
                                    }
                            };

                            class Actor {
                                public:
                                    unsigned int _nb_state_features;
                                    gsl_vector* _params;
                                    rl::enumerator<action_type> _action_begin;
                                    rl::enumerator<action_type> _action_end;
                                    std::function<double(unsigned int, action_type)> _q_function;
                                    double temperature;
                                    std::function<action_type(const state_type&)> _policy;

                                    Actor(unsigned int nb_state_features,
                                            unsigned int nb_actions,
                                            rl::enumerator<action_type> action_begin,
                                            rl::enumerator<action_type> action_end,
                                            RANDOM_GENERATOR& gen):
                                        _nb_state_features(nb_state_features),
                                        _params(gsl_vector_alloc(nb_state_features*nb_actions)),
                                        _action_begin(action_begin), _action_end(action_end),
                                        _q_function(std::bind(&Actor::q_function, std::ref(*this), std::placeholders::_1, std::placeholders::_2)),
                                        temperature(1.0), 
                                        _policy(rl::policy::softmax(_q_function, temperature, _action_begin, _action_end, gen)){
                                            gsl_vector_set_zero(_params);
                                        }

                                    ~Actor() {
                                        gsl_vector_free(_params);
                                    }

                                    double q_function(unsigned int state_idx,
                                            action_type a) const {
                                        unsigned int action_idx = std::distance(_action_begin, rl::enumerator<action_type>(a));
                                        return gsl_vector_get(_params, action_idx*_nb_state_features + state_idx);
                                    }
                                    action_type operator()(unsigned int state_idx) const {
                                        return _policy(state_idx);
                                    }

                            };

                            Critic _critic;
                            Actor _actor;

                        public:
                            Tabular(unsigned int nb_features,
                                    std::function<unsigned int(const STATE&)> state_to_idx,
                                    rl::enumerator<action_type> action_begin,
                                    rl::enumerator<action_type> action_end,
                                    RANDOM_GENERATOR& gen):
                                _nb_features(nb_features),
                                _state_to_idx(state_to_idx),
                                _nb_actions(std::distance(action_begin, action_end)),
                                _action_begin(action_begin), _action_end(action_end),
                                _critic(nb_features),
                                _actor(nb_features, _nb_actions, action_begin, action_end, gen) {
                                }

                            virtual ~Tabular() {
                            }

                            unsigned int getCriticParameterSize() const {
                                return _nb_features;
                            }

                            gsl_vector* getCriticParameters() {
                                return _critic._params;
                            }

                            void grad_critic(gsl_vector* grad, const STATE& s) {
                                gsl_vector_set_basis(grad, _state_to_idx(s));
                            }

                            unsigned int getActorParameterSize() const {
                                return _nb_features;
                            }

                            gsl_vector* getActorParameters() {
                                return _actor._params;
                            }

                            /*
                               Gradient of the log of the policy
                               Here the policy is a softmax
                               */
                            void grad_actor(gsl_vector* grad, const STATE& s, const ACTION& a) {
                                gsl_vector_set_zero(grad);

                                // We need to get the probabilities of the actions
                                std::vector<double> probaActions(_nb_actions);
                                double psum = 0.0;
                                auto aiter = _action_begin;
                                auto piter = probaActions.begin();
                                while(aiter != _action_end) {
                                    *piter = exp(_actor.q_function(_state_to_idx(s), *aiter));
                                    psum += *piter;
                                    ++aiter;
                                    ++piter;
                                }
                                for(auto& p: probaActions)
                                    p /= psum;

                                // And we can then compute the gradient of ln(Pi)
                                piter = probaActions.begin();
                                aiter = _action_begin;
                                while(aiter != _action_end) {
                                    gsl_vector_set(grad, _nb_features * std::distance(_action_begin, aiter) + _state_to_idx(s), ((*aiter)==a) - (*piter));
                                    ++aiter;
                                    ++piter;
                                }
                            }

                            double evaluate_value(const state_type& s) const {
                                return _critic(_state_to_idx(s));
                            }

                            std::map<action_type, double> get_action_probabilities(const state_type& s) const {

                                std::map<action_type, double> proba;

                                // We need to get the probabilities of the actions
                                double psum = 0.0;
                                auto aiter = _action_begin;
                                while(aiter != _action_end) {
                                    proba[*aiter] = exp(_actor.q_function(_state_to_idx(s), *aiter));
                                    psum += proba[*aiter];
                                    ++aiter;
                                }
                                for(auto& ap: proba)
                                    ap.second /= psum;



                                return proba;
                            }

                            action_type sample_action(const state_type& s) const {
                                return _actor(_state_to_idx(s));
                            }

                    };
            }

            namespace Learner {

                /**
                 * @short One-step Actor-Critic (episodic)
                 * Refer to the algorithm Chap 13 of Reinforcement Learning:
                 * An Introduction, R. Sutton, A. Barto 2017, June 19
                 * This implementation does not incorporate the discount I
                 * in the policy update step
                 * because it leads to significantly slower learning
                 * experimented on the cliff walking experiment with Tabular 
                 * state representation with linear value/policy
                 */
                template<typename ARCHITECTURE>
                    class OneStep {
                        using S = typename ARCHITECTURE::state_type;
                        using A = typename ARCHITECTURE::action_type;

                        ARCHITECTURE& _archi;
                        double _gamma;
                        double _alpha_v, _alpha_p;
                        double _discount;
                        gsl_vector* _theta_v;
                        gsl_vector* _grad_v;
                        gsl_vector* _theta_p;
                        gsl_vector* _grad_p;
                        public:

                        OneStep(ARCHITECTURE& archi, double gamma, double alpha_v, double alpha_p):
                            _archi(archi),
                            _gamma(gamma),
                            _alpha_v(alpha_v),
                            _alpha_p(alpha_p),
                            _discount(1.0),
                            _theta_v(_archi.getCriticParameters()),
                            _grad_v(gsl_vector_alloc(_theta_v->size)),
                            _theta_p(_archi.getActorParameters()),
                            _grad_p(gsl_vector_alloc(_theta_p->size)) {
                            }

                        ~OneStep() {
                            gsl_vector_free(_grad_v);
                            gsl_vector_free(_grad_p);
                        }

                        void restart(void) {
                            _discount = 1.0;
                        }

                        void learn(const S &s, const A &a, double rew) {
                            // Evaluate the TD error
                            double td = rew - _archi.evaluate_value(s);

                            // Update the critic
                            _archi.grad_critic(_grad_v, s);
                            // Note : _discount is not present in the original algorithm
                            //        is it a typo ?
                            gsl_blas_daxpy(td*_alpha_v, _grad_v, _theta_v);

                            // Update the actor
                            _archi.grad_actor(_grad_p, s, a);
                            gsl_blas_daxpy(td*_alpha_p, _grad_p, _theta_p);
                            //gsl_blas_daxpy(td*_discount*_alpha_p, _grad_p, _theta_p);

                            //_discount *= _gamma;
                        }

                        void learn(const S &s, const A &a, double rew, const S &s_) {
                            // Evaluate the TD error
                            double td = rew + _gamma * _archi.evaluate_value(s_) - _archi.evaluate_value(s);

                            // Update the critic
                            _archi.grad_critic(_grad_v, s);
                            // Note : _discount is not present in the original algorithm
                            //        is it a typo ?
                            gsl_blas_daxpy(td*_alpha_v, _grad_v, _theta_v);

                            // Update the actor
                            _archi.grad_actor(_grad_p, s, a);
                            gsl_blas_daxpy(td*_alpha_p, _grad_p, _theta_p);
                            //gsl_blas_daxpy(td*_discount*_alpha_p, _grad_p, _theta_p);

                            //_discount *= _gamma;
                        }
                    };

                /**
                 * @short Actor-Critic with Eligibility Traces (episodic)
                 * Refer to the algorithm Chap 13 of Reinforcement Learning:
                 * An Introduction, R. Sutton, A. Barto 2017, June 19
                 * This implementation does not incorporate the discount I
                 * in the value/policy update step
                 * because it leads to significantly slower learning
                 * experimented on the cliff walking experiment with Tabular 
                 * state representation with linear value/policy
                 */	
                template<typename ARCHITECTURE>
                    class EligibilityTraces {
                        using S = typename ARCHITECTURE::state_type;
                        using A = typename ARCHITECTURE::action_type;

                        ARCHITECTURE& _archi;
                        double _gamma;
                        double _alpha_v, _alpha_p;
                        double _lambda_v, _lambda_p;
                        double _discount;
                        gsl_vector* _theta_v;
                        gsl_vector* _grad_v;
                        gsl_vector* _acum_grad_v;
                        gsl_vector* _theta_p;
                        gsl_vector* _grad_p;
                        gsl_vector* _acum_grad_p;
                        public:

                        EligibilityTraces(ARCHITECTURE& archi, double gamma, double alpha_v, double alpha_p, double lambda_v, double lambda_p):
                            _archi(archi),
                            _gamma(gamma),
                            _alpha_v(alpha_v),
                            _alpha_p(alpha_p),
                            _lambda_v(lambda_v),
                            _lambda_p(lambda_p),
                            _discount(1.0),
                            _theta_v(_archi.getCriticParameters()),
                            _grad_v(gsl_vector_alloc(_theta_v->size)),
                            _acum_grad_v(gsl_vector_alloc(_theta_v->size)),
                            _theta_p(_archi.getActorParameters()),
                            _grad_p(gsl_vector_alloc(_theta_p->size)),
                            _acum_grad_p(gsl_vector_alloc(_theta_p->size)) {
                                gsl_vector_set_zero(_acum_grad_v);
                                gsl_vector_set_zero(_acum_grad_p);	    
                            }

                        ~EligibilityTraces() {
                            gsl_vector_free(_grad_v);
                            gsl_vector_free(_acum_grad_v);
                            gsl_vector_free(_grad_p);
                            gsl_vector_free(_acum_grad_p);
                        }

                        void restart(void) {
                            gsl_vector_set_zero(_acum_grad_v);
                            gsl_vector_set_zero(_acum_grad_p);
                            _discount = 1.0;
                        }

                        void learn(const S &s, const A &a, double rew) {
                            // Evaluate the TD error
                            double td = rew - _archi.evaluate_value(s);

                            // Update the critic
                            _archi.grad_critic(_grad_v, s);
                            gsl_vector_scale(_acum_grad_v, _gamma * _lambda_v);
                            gsl_vector_scale(_grad_v, _discount);
                            gsl_vector_add(_acum_grad_v, _grad_v);
                            gsl_blas_daxpy(td*_alpha_v, _acum_grad_v, _theta_v);

                            // Update the actor
                            _archi.grad_actor(_grad_p, s, a);
                            gsl_vector_scale(_acum_grad_p, _gamma * _lambda_p);
                            gsl_vector_scale(_grad_p, _discount);
                            gsl_vector_add(_acum_grad_p, _grad_p);
                            gsl_blas_daxpy(td*_alpha_p, _acum_grad_p, _theta_p);

                            //_discount *= _gamma;
                        }

                        void learn(const S &s, const A &a, double rew, const S &s_) {
                            // Evaluate the TD error
                            double td = rew + _gamma * _archi.evaluate_value(s_) - _archi.evaluate_value(s);

                            // Update the critic
                            _archi.grad_critic(_grad_v, s);
                            gsl_vector_scale(_acum_grad_v, _gamma * _lambda_v);
                            gsl_vector_scale(_grad_v, _discount);
                            gsl_vector_add(_acum_grad_v, _grad_v);
                            gsl_blas_daxpy(td*_alpha_v, _acum_grad_v, _theta_v);

                            // Update the actor
                            _archi.grad_actor(_grad_p, s, a);
                            gsl_vector_scale(_acum_grad_p, _gamma * _lambda_p);
                            gsl_vector_scale(_grad_p, _discount);
                            gsl_vector_add(_acum_grad_p, _grad_p);
                            gsl_blas_daxpy(td*_alpha_p, _acum_grad_p, _theta_p);

                            //_discount *= _gamma;
                        }
                    };

            } // Learner
        } // ActorCritic
    } // gsl
} // rl
