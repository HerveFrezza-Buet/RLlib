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

#include <rlTypes.hpp>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <iostream>


namespace rl {

    template<typename fctGRAD_V_PARAMETRIZED,
        typename fctCurrentOf,
        typename fctNextOf,
        typename fctRewardOf, 
        typename fctIsTerminal,
        typename TRANSITION_ITERATOR>
            void lstd(gsl_vector* theta,
                    double gamma_coef,
                    double reg_coef,
                    const TRANSITION_ITERATOR& trans_begin,
                    const TRANSITION_ITERATOR& trans_end,
                    const fctGRAD_V_PARAMETRIZED& fct_grad_v,
                    const fctCurrentOf& current_of,
                    const fctNextOf& next_of,
                    const fctRewardOf& reward_of,
                    const fctIsTerminal& is_terminal) {

                int n = theta->size;
                int signum;
                gsl_matrix *M      = gsl_matrix_calloc(n, n);
                gsl_vector *b      = gsl_vector_calloc(n);
                gsl_vector *tmp1   = gsl_vector_calloc(n);
                gsl_vector *tmp2   = gsl_vector_calloc(n);
                gsl_permutation *p = gsl_permutation_alloc(n);

                gsl_matrix_set_identity(M);
                gsl_matrix_scale(M,reg_coef);

                for(auto i=trans_begin; i!=trans_end; ++i) {
                    const auto& t = *i;
                    fct_grad_v(theta,tmp1,current_of(t));
                    gsl_blas_dger(1, tmp1, tmp1, M);
                    if(!is_terminal(t)) {
                        fct_grad_v(theta,tmp2,next_of(t));
                        gsl_blas_dger(-gamma_coef, tmp1, tmp2, M);
                    }
                    gsl_blas_daxpy(reward_of(t), tmp1, b); 
                }

                // Inversion of M
                gsl_linalg_LU_decomp (M, p, &signum);
                gsl_linalg_LU_solve (M, p, b, theta);

                gsl_vector_free(tmp2);
                gsl_vector_free(tmp1);	
                gsl_permutation_free (p);
                gsl_matrix_free(M);
                gsl_vector_free(b);
            }

    template<typename fctPHI_PARAMETRIZED,
        typename fctCurrentOf,
        typename fctNextOf,
        typename fctRewardOf, 
        typename fctIsTerminal,
        typename TRANSITION_ITERATOR>
            void rlstd(gsl_vector* theta,
                    double gamma_coef,
                    double reg_coef,
                    const TRANSITION_ITERATOR& trans_begin,
                    const TRANSITION_ITERATOR& trans_end,
                    const fctPHI_PARAMETRIZED& fct_phi,
                    const fctCurrentOf& current_of,
                    const fctNextOf& next_of,
                    const fctRewardOf& reward_of,
                    const fctIsTerminal& is_terminal) {

                int n = theta->size;
                gsl_matrix *C      = gsl_matrix_calloc(n, n);
                gsl_vector *b      = gsl_vector_calloc(n);

                gsl_vector *phi_t  = gsl_vector_alloc(n); 
                gsl_vector *vtmp1  = gsl_vector_alloc(n);
                gsl_vector *vtmp2  = gsl_vector_alloc(n);
                gsl_matrix *mtmp1  = gsl_matrix_alloc(n, n);

                double norm_coef;

                gsl_matrix_set_identity(C);
                gsl_matrix_scale(C, reg_coef);

                for(auto i=trans_begin; i!=trans_end; ++i) {
                    const auto& t = *i;
                    // phi_t = Phi(t)
                    fct_phi(phi_t, current_of(t));

                    gsl_vector_memcpy(vtmp1, phi_t);
                    if(!is_terminal(t)) {
                        // vtmp2 = Phi(t+1)
                        fct_phi(vtmp2, next_of(t));
                        // vtmp1 = Phi(t) - gamma Phi(t+1)
                        gsl_blas_daxpy(-gamma_coef, vtmp2, vtmp1);
                    }
                    // Here, we have : vtmp1 = phi_t <- gamma phi_t_>
                    // The second part of the RHS might be absent

                    // Computes vtmp1 = C^T (phi_t <- gamma phi_t_>) = C^T vtmp1
                    // be carefull, for dgemv, you must output the result
                    // in a vector different from the input..
                    gsl_blas_dgemv(CblasTrans, 1., C, vtmp1, 0., vtmp2);
                    gsl_vector_memcpy(vtmp1, vtmp2);

                    // Computes the normalization coefficient :
                    // norm_coeff = <C^T (phi_t <- gamma phi_t_>), Phi(t)>
                    gsl_blas_ddot(vtmp1, phi_t, &norm_coef);
                    norm_coef = 1. + norm_coef;

                    // Computes vtmp2 = C * phi_t
                    gsl_blas_dgemv(CblasNoTrans, 1., C, phi_t, 0., vtmp2);

                    // Perform the rank-1 update of C
                    gsl_blas_dger(-1./norm_coef,vtmp2 ,vtmp1, C);

                    // b(t+1) = b(t) + R(t+1) * Phi(t)
                    gsl_blas_daxpy(reward_of(t), phi_t, b);
                }  
                // theta = C * b
                gsl_blas_dgemv(CblasNoTrans, 1., C, b, 0., theta); 

                gsl_matrix_free(mtmp1);
                gsl_vector_free(vtmp2);
                gsl_vector_free(vtmp1);
                gsl_vector_free(phi_t);
                gsl_vector_free(b);
                gsl_matrix_free(C);
            }


    /**
     * @short State-less one shot application of recursive LSTD
     *        Compared to lstd it makes use of Sherman Morison 
     *        to iteratively builds up the matrix inverse LSTD involves
     */
    template<typename fctPHI_PARAMETRIZED,
        typename fctCurrentOf,
        typename fctNextOf,
        typename fctRewardOf, 
        typename fctIsTerminal,
        typename TRANSITION_ITERATOR>
            void rlstd_lambda(gsl_vector* theta,
                    double gamma_coef,
                    double reg_coef,
                    double lambda_coef,
                    const TRANSITION_ITERATOR& trans_begin,
                    const TRANSITION_ITERATOR& trans_end,
                    const fctPHI_PARAMETRIZED& fct_phi,
                    const fctCurrentOf& current_of,
                    const fctNextOf& next_of,
                    const fctRewardOf& reward_of,
                    const fctIsTerminal& is_terminal) {

                int n = theta->size;
                gsl_matrix *C      = gsl_matrix_calloc(n, n);
                gsl_vector *b      = gsl_vector_calloc(n);

                gsl_vector *e_t    = gsl_vector_calloc(n);
                gsl_vector *phi_t  = gsl_vector_calloc(n); 
                gsl_vector *vtmp1  = gsl_vector_calloc(n);
                gsl_vector *vtmp2  = gsl_vector_calloc(n);
                gsl_matrix *mtmp1  = gsl_matrix_calloc(n, n);

                double norm_coef;

                gsl_matrix_set_identity(C);
                gsl_matrix_scale(C, reg_coef);

                for(auto i=trans_begin; i!=trans_end; ++i) {
                    const auto& t = *i;
                    // phi_t = Phi(t)
                    fct_phi(phi_t, current_of(t));

                    // e(t+1) = lambda gamma e(t) + phi(t)
                    gsl_vector_scale(e_t, gamma_coef * lambda_coef);
                    gsl_vector_add(e_t, phi_t);

                    gsl_vector_memcpy(vtmp1, phi_t);
                    if(!is_terminal(t)) {
                        // vtmp2 = Phi(t+1)
                        fct_phi(vtmp2, next_of(t));
                        // vtmp1 = Phi(t) - gamma Phi(t+1)
                        gsl_blas_daxpy(-gamma_coef, vtmp2, vtmp1);
                    }
                    // Here, we have : vtmp1 = phi_t <- gamma phi_t_>
                    // The second part of the RHS might be absent

                    // Computes vtmp1 = C^T (phi_t <- gamma phi_t_>) = C^T vtmp1
                    // be carefull, for dgemv, you must output the result
                    // in a vector different from the input..
                    gsl_blas_dgemv(CblasTrans, 1., C, vtmp1, 0., vtmp2);
                    gsl_vector_memcpy(vtmp1, vtmp2);

                    // Computes the normalization coefficient :
                    // norm_coeff = <C^T (phi_t <- gamma phi_t_>), e(t+1)>
                    gsl_blas_ddot(vtmp1, e_t, &norm_coef);
                    norm_coef = 1. + norm_coef;

                    // Computes vtmp2 = C * e(t+1)
                    gsl_blas_dgemv(CblasNoTrans, 1., C, e_t, 0., vtmp2);

                    // Perform the rank-1 update of C
                    gsl_blas_dger(-1./norm_coef,vtmp2 ,vtmp1, C);

                    // b(t+1) = b(t) + R(t+1) * e(t+1)
                    gsl_blas_daxpy(reward_of(t), e_t, b);
                }

                // theta = C * b
                gsl_blas_dgemv(CblasNoTrans, 1., C, b, 0., theta);

                gsl_matrix_free(mtmp1);
                gsl_vector_free(vtmp2);
                gsl_vector_free(vtmp1);
                gsl_vector_free(phi_t);
                gsl_vector_free(e_t);
                gsl_vector_free(b);
                gsl_matrix_free(C);
            }  

    namespace gsl {

        /**
         * @short recursive LSTD with eligibility traces
         *        Compared to the functions rlstd and rlstd_lambda
         *        this class keeps internally the statistics that
         *        recurrently updated
         *        This is a SARSA critic
         */
        template<typename STATE, typename ACTION>
            class LSTDQ {

                private:
                    gsl_vector * _theta_q;
                    double _gamma;
                    std::function<void(gsl_vector*, const STATE&, const ACTION&)> _phi;

                    gsl_matrix* C;
                    gsl_vector* b;
                    gsl_vector* phi_t;
                    gsl_vector* vtmp1;
                    gsl_vector* vtmp2;
                    gsl_matrix* mtmp1;

                    int _nb_warm_up_transitions;
                    int _nb_accumulated_transitions;

                public:
                    template<typename fctPhi_sa_parametrized>
                        LSTDQ(gsl_vector* param,
                                double gamma_coef,
                                double reg_coef,
                                int nb_warm_up_transitions,
                                const fctPhi_sa_parametrized& phi_sa):
                            _theta_q(param),
                            _gamma(gamma_coef),
                            _phi(phi_sa),
                            C(gsl_matrix_calloc(param->size, param->size)),
                            b(gsl_vector_calloc(param->size)),
                            phi_t(gsl_vector_calloc(param->size)),
                            vtmp1(gsl_vector_calloc(param->size)),
                            vtmp2(gsl_vector_calloc(param->size)),
                            mtmp1(gsl_matrix_calloc(param->size, param->size)),
                            _nb_warm_up_transitions(nb_warm_up_transitions),
                            _nb_accumulated_transitions(0) {
                                gsl_matrix_set_identity(C);
                                gsl_matrix_scale(C, reg_coef);
                            }

                    ~LSTDQ() {
                        gsl_matrix_free(mtmp1);
                        gsl_vector_free(vtmp2);
                        gsl_vector_free(vtmp1);
                        gsl_vector_free(phi_t);
                        gsl_vector_free(b);
                        gsl_matrix_free(C);
                    }

                    double td_error (const STATE &s, const ACTION& a, double r, const STATE &s_, const ACTION& a_) {
                        double vt, vt_;
                        _phi(phi_t,  s, a);
                        _phi(vtmp2, s_, a_);
                        gsl_blas_ddot(_theta_q, phi_t, &vt);
                        gsl_blas_ddot(_theta_q, vtmp2, &vt_);
                        return r + _gamma * vt_ - vt;
                    }

                    double td_error (const STATE& s, const ACTION& a, double r) {
                        double vt;
                        _phi(phi_t,  s, a);
                        gsl_blas_ddot(_theta_q, phi_t, &vt);
                        return r - vt;
                    }

                    void learn(const STATE& s, const ACTION& a, double r,
                            const STATE& s_, const ACTION& a_) {
                        ++_nb_accumulated_transitions;

                        _phi(phi_t,  s, a);

                        gsl_vector_memcpy(vtmp1, phi_t);

                        // vtmp2 = Phi(t+1)
                        _phi(vtmp2, s_, a_);

                        // vtmp1 = Phi(t) - gamma Phi(t+1)
                        gsl_blas_daxpy(-_gamma, vtmp2, vtmp1);

                        // Here, we have : vtmp1 = phi_t - gamma phi(t+1)

                        // Computes vtmp1 = C^T (phi_t - gamma phi_t_) = C^T vtmp1
                        // be carefull, for dgemv, you must output the result
                        // in a vector different from the input..
                        gsl_blas_dgemv(CblasTrans, 1., C, vtmp1, 0., vtmp2);
                        gsl_vector_memcpy(vtmp1, vtmp2);

                        // Computes the normalization coefficient :
                        // norm_coeff = <C^T (phi(t) - gamma phi(t+1)), e(t+1)>
                        double norm_coef;
                        gsl_blas_ddot(vtmp1, phi_t, &norm_coef);
                        norm_coef = 1. + norm_coef;

                        // Computes vtmp2 = C * phi_t
                        gsl_blas_dgemv(CblasNoTrans, 1., C, phi_t, 0., vtmp2);

                        // Perform the rank-1 update of C
                        gsl_blas_dger(-1./norm_coef,vtmp2 ,vtmp1, C);

                        // b(t+1) = b(t) + R(t+1) * Phi(t)
                        gsl_blas_daxpy(r, phi_t, b);

                        // If we accumulated a sufficient number of transitions
                        // we begin updating the parameter vector
                        if(_nb_accumulated_transitions >= _nb_warm_up_transitions)
                            // theta = C * b
                            gsl_blas_dgemv(CblasNoTrans, 1., C, b, 0., _theta_q);
                    }

                    void learn(const STATE& s, const ACTION& a, double r) {
                        ++_nb_accumulated_transitions;

                        _phi(phi_t,  s, a);

                        // Computes vtmp1 = C^T phi_t
                        gsl_blas_dgemv(CblasTrans, 1., C, phi_t, 0., vtmp1);

                        // Computes the normalization coefficient :
                        // norm_coeff = <C^T phi_t, phi_t>
                        double norm_coef;
                        gsl_blas_ddot(vtmp1, phi_t, &norm_coef);
                        norm_coef = 1. + norm_coef;

                        // Computes vtmp2 = C * phi_t
                        gsl_blas_dgemv(CblasNoTrans, 1., C, phi_t, 0., vtmp2);

                        // Perform the rank-1 update of C
                        gsl_blas_dger(-1./norm_coef,vtmp2 ,vtmp1, C);

                        // b(t+1) = b(t) + R(t+1) * phi_t
                        gsl_blas_daxpy(r, phi_t, b);

                        // If we accumulated a sufficient number of transitions
                        // we begin updating the parameter vector
                        if(_nb_accumulated_transitions >= _nb_warm_up_transitions)
                            // theta = C * b
                            gsl_blas_dgemv(CblasNoTrans, 1., C, b, 0., _theta_q);	
                    }

            };



        /**
         * @short recursive LSTD with eligibility traces
         *        Compared to the functions rlstd and rlstd_lambda
         *        this class keeps internally the statistics that
         *        recurrently updated
         *        This is a SARSA critic
         */
        template<typename STATE, typename ACTION>
            class LSTDQ_Lambda {

                private:
                    gsl_vector * _theta_q;
                    double _gamma, _lambda;
                    std::function<void(gsl_vector*, const STATE&, const ACTION&)> _phi;

                    gsl_matrix* C;
                    gsl_vector* b;
                    gsl_vector* e_t;
                    gsl_vector* phi_t;
                    gsl_vector* vtmp1;
                    gsl_vector* vtmp2;
                    gsl_matrix* mtmp1;

                    int _nb_warm_up_transitions;
                    int _nb_accumulated_transitions;

                public:
                    template<typename fctPhi_sa_parametrized>
                        LSTDQ_Lambda(gsl_vector* param,
                                double gamma_coef,
                                double reg_coef,
                                double lambda_coef,
                                int nb_warm_up_transitions,
                                const fctPhi_sa_parametrized& phi_sa):
                            _theta_q(param),
                            _gamma(gamma_coef),
                            _lambda(lambda_coef),
                            _nb_warm_up_transitions(nb_warm_up_transitions),
                            _nb_accumulated_transitions(0),
                            _phi(phi_sa),
                            C(gsl_matrix_calloc(param->size, param->size)),
                            b(gsl_vector_calloc(param->size)),
                            e_t(gsl_vector_calloc(param->size)),
                            phi_t(gsl_vector_calloc(param->size)),
                            vtmp1(gsl_vector_calloc(param->size)),
                            vtmp2(gsl_vector_calloc(param->size)),
                            mtmp1(gsl_matrix_calloc(param->size, param->size)) {
                                gsl_matrix_set_identity(C);
                                gsl_matrix_scale(C, reg_coef);
                            }

                    ~LSTDQ_Lambda() {
                        gsl_matrix_free(mtmp1);
                        gsl_vector_free(vtmp2);
                        gsl_vector_free(vtmp1);
                        gsl_vector_free(phi_t);
                        gsl_vector_free(e_t);
                        gsl_vector_free(b);
                        gsl_matrix_free(C);
                    }

                    double td_error (const STATE &s, const ACTION& a, double r, const STATE &s_, const ACTION& a_) {
                        double vt, vt_;
                        _phi(phi_t,  s, a);
                        _phi(vtmp2, s_, a_);
                        gsl_blas_ddot(_theta_q, phi_t, &vt);
                        gsl_blas_ddot(_theta_q, vtmp2, &vt_);
                        return r + _gamma * vt_ - vt;
                    }

                    double td_error (const STATE& s, const ACTION& a, double r) {
                        double vt;
                        _phi(phi_t,  s, a);
                        gsl_blas_ddot(_theta_q, phi_t, &vt);
                        return r - vt;
                    }

                    void learn(const STATE& s, const ACTION& a, double r,
                            const STATE& s_, const ACTION& a_) {
                        ++_nb_accumulated_transitions;

                        _phi(phi_t,  s, a);
                        gsl_vector_memcpy(vtmp1, phi_t);

                        // e(t+1) = lambda gamma e(t) + phi(t)
                        gsl_vector_scale(e_t, _gamma * _lambda);
                        gsl_vector_add(e_t, phi_t);

                        // vtmp2 = Phi(t+1)
                        _phi(vtmp2, s_, a_);

                        // vtmp1 = Phi(t) - gamma Phi(t+1)
                        gsl_blas_daxpy(-_gamma, vtmp2, vtmp1);

                        // Here, we have : vtmp1 = phi_t - gamma phi(t+1)

                        // Computes vtmp1 = C^T (phi_t - gamma phi_t_) = C^T vtmp1
                        // be carefull, for dgemv, you must output the result
                        // in a vector different from the input..
                        gsl_blas_dgemv(CblasTrans, 1., C, vtmp1, 0., vtmp2);
                        gsl_vector_memcpy(vtmp1, vtmp2);

                        // Computes the normalization coefficient :
                        // norm_coeff = <C^T (phi(t) - gamma phi(t+1)), e(t+1)>
                        double norm_coef;
                        gsl_blas_ddot(vtmp1, e_t, &norm_coef);
                        norm_coef = 1. + norm_coef;

                        // Computes vtmp2 = C * e(t+1)
                        gsl_blas_dgemv(CblasNoTrans, 1., C, e_t, 0., vtmp2);

                        // Perform the rank-1 update of C
                        gsl_blas_dger(-1./norm_coef,vtmp2 ,vtmp1, C);

                        // b(t+1) = b(t) + R(t+1) * e(t+1)
                        gsl_blas_daxpy(r, e_t, b);

                        // If we accumulated a sufficient number of transitions
                        // we begin updating the parameter vector
                        if(_nb_accumulated_transitions >= _nb_warm_up_transitions)
                            // theta = C * b
                            gsl_blas_dgemv(CblasNoTrans, 1., C, b, 0., _theta_q);
                    }

                    void learn(const STATE& s, const ACTION& a, double r) {
                        ++_nb_accumulated_transitions;

                        _phi(phi_t,  s, a);
                        gsl_vector_memcpy(vtmp1, phi_t);

                        // e(t+1) = lambda gamma e(t) + phi(t)
                        gsl_vector_scale(e_t, _gamma * _lambda);
                        gsl_vector_add(e_t, phi_t);

                        // Computes vtmp1 = C^T phi_t
                        gsl_blas_dgemv(CblasTrans, 1., C, phi_t, 0., vtmp1);

                        // Computes the normalization coefficient :
                        // norm_coeff = <C^T phi_t, e(t+1)>
                        double norm_coef;
                        gsl_blas_ddot(vtmp1, e_t, &norm_coef);
                        norm_coef = 1. + norm_coef;

                        // Computes vtmp2 = C * e(t+1)
                        gsl_blas_dgemv(CblasNoTrans, 1., C, e_t, 0., vtmp2);

                        // Perform the rank-1 update of C
                        gsl_blas_dger(-1./norm_coef,vtmp2 ,vtmp1, C);

                        // b(t+1) = b(t) + R(t+1) * e(t+1)
                        gsl_blas_daxpy(r, e_t, b);

                        // If we accumulated a sufficient number of transitions
                        // we begin updating the parameter vector
                        if(_nb_accumulated_transitions >= _nb_warm_up_transitions)
                            // theta = C * b
                            gsl_blas_dgemv(CblasNoTrans, 1., C, b, 0., _theta_q);	
                    }

            };
    }

}
