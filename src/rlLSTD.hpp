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

  template<typename fctGRAD_V_PARAMETRIZED,
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
	     const fctGRAD_V_PARAMETRIZED& fct_grad_v,
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
      fct_grad_v(theta, phi_t, current_of(t));
	
      gsl_vector_memcpy(vtmp1, phi_t);
      if(!is_terminal(t)) {
	// vtmp2 = Phi(t+1)
	fct_grad_v(theta, vtmp2, next_of(t));
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
      
      // theta = C * b
      gsl_blas_dgemv(CblasNoTrans, 1., C, b, 0., theta);
    }   
    
    gsl_matrix_free(mtmp1);
    gsl_vector_free(vtmp2);
    gsl_vector_free(vtmp1);
    gsl_vector_free(phi_t);
    gsl_vector_free(b);
    gsl_matrix_free(C);
  }

  template<typename fctGRAD_V_PARAMETRIZED,
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
		    const fctGRAD_V_PARAMETRIZED& fct_grad_v,
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
      fct_grad_v(theta, phi_t, current_of(t));

      // e(t+1) = lambda gamma e(t) + phi(t)
      gsl_vector_scale(e_t, gamma_coef * lambda_coef);
      gsl_vector_add(e_t, phi_t);
      
      gsl_vector_memcpy(vtmp1, phi_t);
      if(!is_terminal(t)) {
	// vtmp2 = Phi(t+1)
	fct_grad_v(theta, vtmp2, next_of(t));
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
	
      // theta = C * b
      gsl_blas_dgemv(CblasNoTrans, 1., C, b, 0., theta);
    }
    
    gsl_matrix_free(mtmp1);
    gsl_vector_free(vtmp2);
    gsl_vector_free(vtmp1);
    gsl_vector_free(phi_t);
    gsl_vector_free(e_t);
    gsl_vector_free(b);
    gsl_matrix_free(C);
  }  
  
}
