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
#include <rlConcept.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <functional>

namespace rl {
  namespace transfer {
    inline double identity(double weighted_sum) {return weighted_sum;}

    /**
     * @short This acts as a sigmoid, since it is y=ax kept in [-1,1]
     */
    inline double saturation(double weighted_sum,
			     double a) {
      double res = weighted_sum*a;
      if(res > 1)
	return 1;
      if(res < -1)
	return -1;
      return weighted_sum;
    } 
      
    /**
     * @short This is tanh(ax) saturation.
     */
    /**
     * @short This acts as a sigmoid, since it is y=ax kept in [-1,1]
     */
    inline double tanh(double weighted_sum,
		       double a) {
      return ::tanh(weighted_sum*a);
    } 
  }

  namespace gsl {
    namespace mlp {

      

      /**
       * @short This defines the input layer of the neural network.
       */
      template<typename STATE,
	       typename ACTION,
	       typename fctFEATURE> 
      class Input {
      private:

	gsl_vector* xx;
	std::function<void (gsl_vector*,const STATE&, const ACTION&)> phi;
	unsigned int phi_dim;
	
      public:
	
	typedef STATE  state_type;
	typedef ACTION action_type;
	
	unsigned int rank(void)         const {return 0;}
	unsigned int minParamRank(void) const {return 0;}
	unsigned int nbParams(void)     const {return 0;}
	unsigned int layerSize(void)    const {return phi_dim;}
	
	unsigned int size;

	void displayParameters(std::ostream& os) const {
	  os << "Input  #" << std::setw(3) << rank()
	     << " :        no weight"
	     << " : size = " << std::setw(4) << layerSize() << std::endl;
	}

	Input(const fctFEATURE& f, unsigned int feature_dimension) 
	  : xx(0),
	    phi(f),
	    phi_dim(feature_dimension),
	    size(0) {
	  xx = gsl_vector_alloc(phi_dim);
	}
	Input(const Input<STATE,ACTION,fctFEATURE>& cp) 
	  : xx(0), phi(cp.phi), phi_dim(cp.phi_dim), size(cp.size) {
	  xx = gsl_vector_alloc(cp.xx->size);
	  gsl_vector_memcpy(xx,cp.xx);
	}

	Input<STATE,ACTION,fctFEATURE> operator=(const Input<STATE,ACTION,fctFEATURE>& cp) {
	  if(this != &cp) {
	    phi = cp.phi;
	    phi_dim = cp.phi_dim;
	    gsl_vector_free(xx);
	    xx = gsl_vector_alloc(phi_dim);
	    gsl_vector_memcpy(xx,cp.xx);
	    size = cp.size;
	  }
	  return this;
	}

	~Input(void) {gsl_vector_free(xx);}

	void operator()(const gsl_vector* theta, 
			const state_type& s, const action_type& a,
			double* y) const {
	  unsigned int i;
	  phi(xx,s,a);
	  double* end = y+layerSize();
	  double* iter;
	  for(iter = y, i=0; iter != end; ++i,++iter) 
	    *iter = gsl_vector_get(xx,i);
	}
      };

      template<typename STATE,
	       typename ACTION,
	       typename fctFEATURE> 
      Input<STATE,ACTION,fctFEATURE> input(const fctFEATURE& f,unsigned int feature_dimension) {
	return Input<STATE,ACTION,fctFEATURE>(f,feature_dimension);
      }

      /**
       * @short This defines the some hidden layer of the neural network.
       */
      template<typename PREVIOUS_LAYER,typename MLP_TRANSFER>
      class Hidden {
      private:
	unsigned int layer_size;
	mutable std::vector<double> yy;

      public:
	typedef typename PREVIOUS_LAYER::state_type  state_type;
	typedef typename PREVIOUS_LAYER::action_type action_type;

	PREVIOUS_LAYER& input;
	std::function<double (double)> f;
	unsigned int size;
	
	unsigned int rank(void)         const {return 1+input.rank();}
	unsigned int minParamRank(void) const {return input.minParamRank()+input.nbParams();}
	unsigned int nbParams(void)     const {return layer_size*(1+input.layerSize());}
	unsigned int layerSize(void)    const {return layer_size;}


	void displayParameters(std::ostream& os) const {
	  input.displayParameters(os);
	  os << "Hidden #" << std::setw(3) << rank()
	     << " : " << "[" << std::setw(6) << minParamRank() << ", " << std::setw(6) << minParamRank()+nbParams() << "["
	     << " : size = " << std::setw(4) << layerSize() << std::endl;
	}

	Hidden(PREVIOUS_LAYER& in,
	       unsigned int nb_neurons,
	       const MLP_TRANSFER& transfer) 
	  : layer_size(nb_neurons), yy(), input(in), f(transfer) {
	  size = minParamRank() + nbParams();
	}

	Hidden(const Hidden<PREVIOUS_LAYER,MLP_TRANSFER>& cp) 
	  : layer_size(cp.layer_size), yy(cp.yy), input(cp.input), f(cp.f), size(cp.size) {}

	Hidden<PREVIOUS_LAYER,MLP_TRANSFER>& operator=(const Hidden<PREVIOUS_LAYER,MLP_TRANSFER>& cp) {
	  if(this != &cp) {
	    layer_size = cp.layer_size; 
	    yy = cp.yy; 
	    input = cp.input;
	    f = cp.f;
	    size = cp.size;
	  }
	  return *this;
	}

	void operator()(const gsl_vector* theta,
			const state_type& s, const action_type& a,
			double* y) const {
	  unsigned int k;
	  double sum;

	  yy.resize(input.layerSize());
	  input(theta,s,a,&(*(yy.begin())));

	  typename std::vector<double>::const_iterator j,yyend;
	  double* i;
	  double* yend = y+layerSize();

	  k=minParamRank();
	  yyend = yy.end();
	  for(i=y;i!=yend;++i) {
	    sum = gsl_vector_get(theta,k);++k;
	    for(j=yy.begin();j!=yyend;++j,++k)
	      sum += gsl_vector_get(theta,k)*(*j);
	    *i = f(sum);
	  }
	}
      };

      template<typename PREVIOUS_LAYER,typename MLP_TRANSFER>
      Hidden<PREVIOUS_LAYER,MLP_TRANSFER> hidden(PREVIOUS_LAYER& in,
						 unsigned int layer_size,
						 const MLP_TRANSFER& transfer) {
	return Hidden<PREVIOUS_LAYER,MLP_TRANSFER>(in,layer_size,transfer);
      }

      /**
       * @short This defines the output layer of the neural network.
       */
      template<typename PREVIOUS_LAYER,typename MLP_TRANSFER>
      class Output {
      private:
	mutable std::vector<double> y;
      public:

	
	typedef typename PREVIOUS_LAYER::state_type  state_type;
	typedef typename PREVIOUS_LAYER::action_type action_type;

	unsigned int rank(void)         const {return 1+input.rank();}
	unsigned int minParamRank(void) const {return input.minParamRank()+input.nbParams();}
	unsigned int nbParams(void)     const {return 1*(1+input.layerSize());}
	unsigned int layerSize(void)    const {return 1;}
	
	PREVIOUS_LAYER& input;
	std::function<double (double)> f;
	unsigned int size;
	

	void displayParameters(std::ostream& os) const {
	  input.displayParameters(os);
	  os << "Output #" << std::setw(3) << rank()
	     << " : " << "[" << std::setw(6) << minParamRank() << ", " << std::setw(6) << minParamRank()+nbParams() << "["
	     << " : size = " << std::setw(4) << layerSize() << std::endl;
	}

	Output(PREVIOUS_LAYER& in,
	       const MLP_TRANSFER& transfer) 
	  : y(),input(in), f(transfer) {
	  size = minParamRank() + nbParams();
	}
	Output(const Output<PREVIOUS_LAYER,MLP_TRANSFER>& cp)
	  : y(cp.y), input(cp.input), f(cp.f), size(cp.size) {}
	Output<PREVIOUS_LAYER,MLP_TRANSFER>& operator=(const Output<PREVIOUS_LAYER,MLP_TRANSFER>& cp) {
	  if(this != &cp) {
	    y = cp.y; 
	    input = cp.input;
	    f = cp.f;
	    size = cp.size;
	  }
	  return *this;
	}


	double operator()(const gsl_vector* theta, const state_type& s, const action_type& a) const {
	  unsigned int k;
	  double sum;
	  
	  y.resize(input.layerSize());

	  input(theta,s,a,&(*(y.begin())));

	  typename std::vector<double>::const_iterator j,yend;

	  k=minParamRank();
	  yend = y.end();
	  sum = gsl_vector_get(theta,k);++k;
	  for(j=y.begin();j!=yend;++j,++k)
	    sum += gsl_vector_get(theta,k)*(*j);
	  return f(sum);
	}
      };

      template<typename PREVIOUS_LAYER,typename MLP_TRANSFER>
      Output<PREVIOUS_LAYER,MLP_TRANSFER> output(PREVIOUS_LAYER& in,
						 const MLP_TRANSFER& transfer) {
	return Output<PREVIOUS_LAYER,MLP_TRANSFER>(in,transfer);
      }

    }
  }
}
