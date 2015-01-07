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

}
