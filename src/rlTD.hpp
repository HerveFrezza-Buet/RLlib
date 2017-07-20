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

namespace rl {
  namespace exception {

    class TDBadParam : public Any {
    public:
      
      TDBadParam(std::string comment) 
	: Any(std::string("Bad theta parameter in TD: ")+comment) {}
    };


    
  }

  namespace gsl {

    /**
     * @short TD algorithm
     * Z can be S or (S,A).
     */
    template<typename Z>
    class TD {

    public:
      
      typedef std::function<double (const gsl_vector*,const Z&)>           v_type;
      typedef std::function<void (const gsl_vector*,gsl_vector*,const Z&)> gv_type;

    private:

      TD(void) {}

    protected:

      gsl_vector* theta;

    private:

      gsl_vector* grad;

    protected:


      v_type  v;
      gv_type gv;
      
      void td_update(const Z& z, double td) {
	// theta <- theta + alpha*td*grad
	gv(theta,grad,z);
	gsl_blas_daxpy(td*alpha,grad,theta);
      }

    public:
      double gamma;
      double alpha;
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
	  gamma(gamma_coef), alpha(alpha_coef) {
      }

      TD(const TD<Z>& cp) {
	*this = cp;
      }


       TD<Z>& operator=(const TD<Z>& cp) {
	if(this != &cp) {
	  if(theta != cp.theta) {
	    if(theta == 0 || cp.theta == 0) 
	      throw rl::exception::TDBadParam("Null parameter in copy");
	    if(theta->size != cp.theta->size)
	      throw rl::exception::TDBadParam("Incompatible parameter size in copy");
	    gsl_vector_memcpy(theta,cp.theta);
	  }
	  gsl_vector_memcpy(grad,cp.grad);
	  v = cp.v;
	  gv = cp.gv;
	  alpha = cp.alpha;
	  gamma = cp.gamma;
	}
	return *this;
      }

      virtual ~TD(void) {
	gsl_vector_free(grad);
      }

      virtual double td_error(const Z& z, double r, const Z& z_) {
	return r + gamma*v(theta,z_) - v(theta,z);
      }

      virtual double td_error(const Z& z, double r) {
	return r - v(theta,z);
      }
      
      void learn(const Z& z, double r, const Z& z_) {
	this->td_update(z,this->td_error(z, r, z_));
      }

      void learn(const Z& z, double r) {
	this->td_update(z,this->td_error(z, r));
      }

      /**
       * This is for complience rl::concept::SarsaCritic, in order to
       * be used in episode handling.  The action is ignored.
       */
      template<typename ACTION>
      void learn(const Z& z, const ACTION& a, double r, const Z& z_, const ACTION& a_) {
	learn(z,r,z_);
      }

      /**
       * This is for complience rl::concept::SarsaCritic, in order to
       * be used in episode handling.  The action is ignored.
       */
      template<typename ACTION>
      void learn(const Z& z, const ACTION& a, double r) {
	learn(z,r);
      }
    };
    
    template<typename Z, typename fctV_PARAMETRIZED, typename fctGRAD_V_PARAMETRIZED>
    TD<Z> td(gsl_vector* param,
	     double gamma_coef,
	     double alpha_coef,
	     const fctV_PARAMETRIZED&  fct_v,
	     const fctGRAD_V_PARAMETRIZED& fct_grad_v) {
      return TD<Z>(param,
		   gamma_coef,
		   alpha_coef,
		   fct_v,fct_grad_v);
    }
  }
}
