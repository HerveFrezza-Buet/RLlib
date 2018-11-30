/*   This file is part of rl-lib
 *
 *   Copyright (C) 2010,  Supelec
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

#pragma once

#include <sstream>
#include <type_traits>
#include <functional>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include <rlAlgo.hpp>
#include <rlException.hpp>
#include <rlTD.hpp>

namespace rl {

    namespace gsl {
        /**
         * @short QLearning algorithm
         */
        template<typename STATE,
            typename ACTION,
            typename ACTION_ITERATOR>
                class QLearning {

                    public:
                        using q_type  = std::function<double (const gsl_vector*, const STATE&, const ACTION&)>;
                        using gq_type = std::function<void (const gsl_vector*,gsl_vector*,const STATE&, const ACTION&)>;

                    protected:

                        // The parameter vector for the Q-function
                        gsl_vector* theta;
                        // A temporary vector holding the gradient of the value function
                        gsl_vector* grad;
                        // The parametrized Q(theta, s,a) function
                        q_type q;
                        // The grad_theta Q(theta, s, a)
                        gq_type  gq;

                        // Iterators over the collection of actions
                        ACTION_ITERATOR a_begin,a_end;

                        // The function computing the update of the parameter vector
                        // given we were in s executing action a
                        void td_update(const STATE& s, const ACTION& a, double td) {
                            // theta <- theta + alpha*td*grad
                            gq(theta, grad, s, a);
                            gsl_blas_daxpy(td*alpha, grad, theta);
                        }

                    public:

                        // The discount factor
                        double gamma;

                        // The learning rate for theta
                        double alpha;

                        QLearning(void) = delete;
                        QLearning(const QLearning<STATE,ACTION,ACTION_ITERATOR>& cp) = delete; 
                        QLearning<STATE,ACTION,ACTION_ITERATOR>& operator=(const QLearning<STATE,ACTION,ACTION_ITERATOR>& cp) = delete;


                        template<typename fctQ_PARAMETRIZED,
                                 typename fctGRAD_Q_PARAMETRIZED>
                        QLearning(gsl_vector* param,
                                double gamma_coef,
                                double alpha_coef,
                                const ACTION_ITERATOR& begin,
                                const ACTION_ITERATOR& end,
                                const fctQ_PARAMETRIZED& fct_q,
                                const fctGRAD_Q_PARAMETRIZED& fct_grad_q):
                            theta(param), grad(gsl_vector_alloc(param->size)), 
                            q(fct_q), gq(fct_grad_q), a_begin(begin),a_end(end),
                            gamma(gamma_coef), alpha(alpha_coef) {}

                        virtual ~QLearning(void) {
                            gsl_vector_free(grad);
                        }

                        double td_error(const STATE& s, const ACTION& a,
                                double r, const STATE& s_) {
                            auto qq = this->q;
                            auto tt = this->theta;
                            auto qq_s_ = [&qq, &tt, &s_](ACTION aa) -> double {return qq(tt,s_,aa);};
                            return r + this->gamma*rl::argmax(qq_s_, a_begin, a_end).second - q(theta, s, a);
                        }

                        void learn(const STATE& s, const ACTION& a, double r,
                                const STATE& s_) {
                            td_update(s, a, td_error(s, a, r, s_));
                        }

                        double td_error(const STATE& s, const ACTION& a, double r) {
                            return r - q(theta, s, a);
                        }     

                        void learn(const STATE& s, const ACTION& a, double r) {
                            td_update(s, a, td_error(s, a, r));
                        }

                };


        template<typename STATE,
            typename ACTION,
            typename fctQ_PARAMETRIZED,
            typename fctGRAD_Q_PARAMETRIZED,
            typename ACTION_ITERATOR>
                auto q_learning(gsl_vector* param,
                        double gamma_coef,
                        double alpha_coef,
                        const ACTION_ITERATOR& action_begin,
                        const ACTION_ITERATOR& action_end,
                        const fctQ_PARAMETRIZED& fct_q,
                        const fctGRAD_Q_PARAMETRIZED& fct_grad_q) 
                -> QLearning<STATE,ACTION,ACTION_ITERATOR>{
                    return QLearning<STATE,ACTION,ACTION_ITERATOR>
                        (param,
                         gamma_coef,alpha_coef,
                         action_begin,action_end,
                         fct_q,fct_grad_q);
                }

    }
}

