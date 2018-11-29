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

#include <rlTraits.hpp>
#include <gsl/gsl_vector.h>

namespace rl {

    namespace gsl {

        template<typename ...> class TD;

        /**
         * @short TD algorithm for learning a state value function
         */
        template<typename STATE>
            class TD<STATE> {

                public:

                    using v_type  = std::function<double (const gsl_vector*,const STATE&)>;
                    using gv_type = std::function<void (const gsl_vector*,gsl_vector*,const STATE&)>;

                protected:

                    // The parameter vector for the V-function
                    gsl_vector* theta;
                    // A temporary vector holding the gradient of the value function
                    gsl_vector* grad;

                    // The parametrized V(theta, s) function
                    v_type  v;
                    // The parametrized grad_theta V(theta, s)
                    gv_type gv;

                    // The function computing the update of the parameter vector
                    // given we were in s executing action a
                    void td_update(const STATE& s, double td) {
                        // theta <- theta + alpha*td*grad
                        gv(theta, grad, s);
                        gsl_blas_daxpy(td*alpha, grad, theta);
                    }

                public:

                    // The discount factor
                    double gamma;

                    // The learning rate for theta
                    double alpha;

                    TD(void)   = delete;
                    TD(const TD<STATE>& cp) = delete;
                    TD<STATE>& operator=(const TD<STATE>& cp) = delete;

                    template<typename fctV,
                             typename fctGRAD_V>
                            TD(gsl_vector* param,
                                    double gamma_coef,
                                    double alpha_coef,
                                    const fctV&      fct_v,
                                    const fctGRAD_V& fct_grad_v)
                            : theta(param),
                            grad(gsl_vector_alloc(param->size)),
                            v(fct_v), gv(fct_grad_v),
                            gamma(gamma_coef), alpha(alpha_coef) {}

                    virtual ~TD(void) {
                        gsl_vector_free(grad);
                    }

                    double td_error(const STATE& s, double r, const STATE& s_) {
                        return r + gamma*v(theta,s_) - v(theta,s);
                    }

                    // Learning function for a non terminal state
                    void learn(const STATE& s, double r, const STATE& s_) {
                        this->td_update(s,this->td_error(s, r, s_));
                    }

                    double td_error(const STATE& s, double r) {
                        return r - v(theta,s);
                    }

                    // Learning function for a terminal state
                    void learn(const STATE& s, double r) {
                        this->td_update(s,this->td_error(s, r));
                    }
            };

        template<typename STATE, typename fctV_PARAMETRIZED, typename fctGRAD_V_PARAMETRIZED>
            typename std::enable_if_t<rl::traits::gsl::is_parametrized_state_value_function<fctV_PARAMETRIZED, STATE>::value, TD<STATE> >
            td(gsl_vector* param,
                    double gamma_coef,
                    double alpha_coef,
                    const fctV_PARAMETRIZED&  fct_v,
                    const fctGRAD_V_PARAMETRIZED& fct_grad_v) {
                return TD<STATE>(param,
                        gamma_coef,
                        alpha_coef,
                        fct_v,fct_grad_v);
            }

        /**
         * @short TD algorithm for learning a state-action value function
         */
        template<typename STATE, typename ACTION>
            class TD<STATE, ACTION> {

                public:

                    using q_type  = std::function<double (const gsl_vector*, const STATE&, const ACTION&)>;
                    using gq_type = std::function<void (const gsl_vector*,gsl_vector*,const STATE&, const ACTION&)>;

                protected:

                    // The parameter vector for the Q-function
                    gsl_vector* theta;
                    // A temporary vector holding the gradient of the value function
                    gsl_vector* grad;

                    // The parametrized Q(theta, s, a) function
                    q_type  q;
                    // The parametrized grad_theta Q(theta, s, a)
                    gq_type gq;

                        
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

                    TD(void) = delete;
                    TD(const TD<STATE, ACTION>& cp) = delete;
                    TD<STATE, ACTION>& operator=(const TD<STATE, ACTION>& cp) = delete;
                    
                    template<typename fctQ,
                        typename fctGRAD_Q>
                            TD(gsl_vector* param,
                                    double gamma_coef,
                                    double alpha_coef,
                                    const fctQ&      fct_q,
                                    const fctGRAD_Q& fct_grad_q)
                            : theta(param),
                            grad(gsl_vector_alloc(param->size)),
                            q(fct_q), gq(fct_grad_q),
                            gamma(gamma_coef), alpha(alpha_coef) { }


                    virtual ~TD(void) {
                        gsl_vector_free(grad);
                    }

                    double td_error(const STATE& s, const ACTION& a, double r, const STATE& s_, const ACTION& a_) {
                        return r + gamma*q(theta,s_, a_) - q(theta, s, a);
                    }

                    // Learning function for a non terminal state
                    void learn(const STATE& s, const ACTION& a, double r, const STATE& s_, const ACTION& a_) {
                        this->td_update(s, a, this->td_error(s, a, r, s_, a_));
                    }

                    double td_error(const STATE& s, const ACTION& a, double r) {
                        return r - q(theta, s, a);
                    }
                    
                    // Learning function for a terminal state
                    void learn(const STATE& s, const ACTION& a, double r) {
                        this->td_update(s, a, this->td_error(s, a, r));
                    }
            };

        template<typename STATE, typename ACTION, 
            typename fctQ_PARAMETRIZED, typename fctGRAD_Q_PARAMETRIZED>
                typename std::enable_if_t<rl::traits::gsl::is_parametrized_state_action_value_function<fctQ_PARAMETRIZED, STATE, ACTION>::value, TD<STATE, ACTION> >
                td(gsl_vector* param,
                        double gamma_coef,
                        double alpha_coef,
                        const fctQ_PARAMETRIZED&  fct_q,
                        const fctGRAD_Q_PARAMETRIZED& fct_grad_q) {
                    return TD<STATE, ACTION>(param,
                            gamma_coef,
                            alpha_coef,
                            fct_q,fct_grad_q);
                }
    }
}
