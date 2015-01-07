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
	     typename fctQ_PARAMETRIZED,
	     typename fctGRAD_Q_PARAMETRIZED,
	     typename ACTION_ITERATOR>
    class QLearning : public TD<rl::sa::Pair<STATE,ACTION> > {

    public:

      typedef TD<rl::sa::Pair<STATE,ACTION> > super_type;
      
    private:

      QLearning(void) {}

      ACTION_ITERATOR a_begin,a_end;
      
    public:

      QLearning(gsl_vector* param,
		double gamma_coef,
		double alpha_coef,
		const ACTION_ITERATOR& begin,
		const ACTION_ITERATOR& end,
		const fctQ_PARAMETRIZED& fct_q,
		const fctGRAD_Q_PARAMETRIZED& fct_grad_q)
	: super_type(param,
		     gamma_coef,alpha_coef,
		     rl::sa::gsl::vparam_of_qparam<STATE,ACTION,double>(fct_q),
		     rl::sa::gsl::gradvparam_of_gradqparam<STATE,ACTION,double>(fct_grad_q)),
	  a_begin(begin),a_end(end) {}

      QLearning(const QLearning<STATE,ACTION,fctQ_PARAMETRIZED,fctGRAD_Q_PARAMETRIZED,ACTION_ITERATOR>& cp) 
	: super_type(cp),
	  a_begin(cp.a_begin),
	  a_end(cp.a_end) {}

      QLearning<STATE,ACTION,fctQ_PARAMETRIZED,fctGRAD_Q_PARAMETRIZED,ACTION_ITERATOR>& operator=(const QLearning<STATE,ACTION,fctQ_PARAMETRIZED,fctGRAD_Q_PARAMETRIZED,ACTION_ITERATOR>& cp) {
	if(this != &cp) {
	  this->super_type::operator=(cp);
	  a_begin = cp.a_begin;
	  a_end   = cp.a_end;
	}
	return *this;
      }

      virtual ~QLearning(void) {}

      void learn(const STATE& s, const ACTION& a, double r,
		 const STATE& s_, const ACTION& a_) {
	auto vv    = this->v;
	auto tt    = this->theta;
	auto qq_s_ = [&vv,&tt,&s_](ACTION aa) -> double {return vv(tt,{s_,aa});};
	this->td_update({s,a},
			r + this->gamma*rl::argmax(qq_s_,
						   a_begin,
						   a_end).second 
			- vv(tt,{s,a}));
      }

      void learn(const STATE& s, const ACTION& a, double r) {
	this->super_type::learn({s,a},r);
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
      -> QLearning<STATE,ACTION,fctQ_PARAMETRIZED,fctGRAD_Q_PARAMETRIZED,ACTION_ITERATOR>{
      return QLearning<STATE,ACTION,fctQ_PARAMETRIZED,fctGRAD_Q_PARAMETRIZED,ACTION_ITERATOR>
	(param,
	 gamma_coef,alpha_coef,
	 action_begin,action_end,
	 fct_q,fct_grad_q);
    }
  
  }
}

