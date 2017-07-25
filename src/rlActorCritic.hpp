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
      template<typename STATE, typename ACTION,
	       typename CRITIC>
      class OneStep {

      private:
	CRITIC& _critic;
	gsl_vector* _theta_p;
	gsl_vector* _grad;
	double _alpha_p;
	std::function<void(const gsl_vector*, gsl_vector*, const STATE&, const ACTION&)> _grad_log_p; 

      public:

	template<typename fctGRAD_LOGP_PARAMETRIZED>
	OneStep(CRITIC& critic,
		gsl_vector* theta_p,
		double alpha_p,
		const fctGRAD_LOGP_PARAMETRIZED& grad_log_p):
	  _critic(critic), 
	  _theta_p(theta_p),
	  _grad(gsl_vector_alloc(theta_p->size)),
	  _alpha_p(alpha_p),
	  _grad_log_p(grad_log_p){
	}

	~OneStep() {
	  gsl_vector_free(_grad);
	}

	template<typename=void>
	std::enable_if_t<rl::traits::is_srs_critic<CRITIC, STATE>::value>
	learn(const STATE &s, const ACTION &a, double rew) {
	  // Evaluate the TD error for later updating the actor
	  double td = _critic.td_error(s, rew);
	    
	  // Update the critic
	  _critic.learn(s, rew);
	  
	  // Update the actor
	  _grad_log_p(_theta_p, _grad, s, a);
	  gsl_blas_daxpy(td*_alpha_p, _grad, _theta_p);
	}
	
	template<typename=void>
	std::enable_if_t<rl::traits::is_srs_critic<CRITIC, STATE>::value>
	learn(const STATE &s, const ACTION &a, double rew, const STATE &s_) {
	  // Evaluate the TD error
	  double td = _critic.td_error(s, rew, s_);;
	    
	  // Update the critic
	  _critic.learn(s, rew, s_);
	    
	  // Update the actor
	  _grad_log_p(_theta_p, _grad, s, a);
	  gsl_blas_daxpy(td*_alpha_p, _grad, _theta_p);
	}

	template<typename=void>
	std::enable_if_t<rl::traits::is_sarsa_critic<CRITIC, STATE, ACTION>::value>
	learn(const STATE &s, const ACTION &a, double rew) {
	  // Evaluate the TD error for later updating the actor
	  double td = _critic.td_error(s, a, rew);
	    
	  // Update the critic
	  _critic.learn(s, a, rew);
	  
	  // Update the actor
	  _grad_log_p(_theta_p, _grad, s, a);
	  gsl_blas_daxpy(td*_alpha_p, _grad, _theta_p);
	}
	
	template<typename=void>
	std::enable_if_t<rl::traits::is_sarsa_critic<CRITIC, STATE, ACTION>::value>
	learn(const STATE &s, const ACTION &a, double rew, const STATE &s_, const STATE &a_) {
	  // Evaluate the TD error
	  double td = _critic.td_error(s, a, rew, s_, a_);;
	    
	  // Update the critic
	  _critic.learn(s, a, rew, s_, a_);
	    
	  // Update the actor
	  _grad_log_p(_theta_p, _grad, s, a);
	  gsl_blas_daxpy(td*_alpha_p, _grad, _theta_p);
	}

	
      };

      template<typename STATE, typename ACTION,
	       typename CRITIC, typename fctGRAD_LOGP_PARAMETRIZED>
      OneStep<STATE, ACTION, CRITIC> 
      one_step(CRITIC& critic, gsl_vector* theta_p, double alpha_p, const fctGRAD_LOGP_PARAMETRIZED& grad_log_p) {
	return OneStep<STATE, ACTION, CRITIC>(critic, theta_p, alpha_p, grad_log_p);
      }
	
      /**
       * @short Actor-Critic with Eligibility Traces (episodic)
       * Refer to the algorithm Chap 13 of Reinforcement Learning:
       * An Introduction, R. Sutton, A. Barto 2017, June 19
       * This implementation does not incorporate the discount I
       * in the value/policy update step
       * because it leads to significantly slower learning
       * experimented on the cliff walking experiment with Tabular 
       * state representation with linear value/policy
       * This actor critic implements eligibility traces
       * for training the actor. It is your responsibility to use a critic
       * with eligibility traces
       */	
      template<typename STATE, typename ACTION,
	       typename CRITIC>
      class EligibilityTraces {
	
      private:
	CRITIC& _critic;
	double _gamma;
	gsl_vector* _theta_p;
	gsl_vector* _grad;
	gsl_vector* _acum_grad;
	double _alpha_p, _lambda_p;
	std::function<void(const gsl_vector*, gsl_vector*, const STATE&, const ACTION&)> _grad_log_p; 

      public:

	template<typename fctGRAD_LOGP_PARAMETRIZED>
	EligibilityTraces(CRITIC& critic,
			  double gamma,
			  gsl_vector* theta_p,
			  double alpha_p,
			  double lambda_p,
			  const fctGRAD_LOGP_PARAMETRIZED& grad_log_p):
	  _critic(critic),
	  _gamma(gamma),
	  _theta_p(theta_p),
	  _grad(gsl_vector_alloc(theta_p->size)),
	  _acum_grad(gsl_vector_alloc(theta_p->size)),
	  _alpha_p(alpha_p),
	  _lambda_p(lambda_p),
	  _grad_log_p(grad_log_p) {
	  gsl_vector_set_zero(_acum_grad);
	}

	~EligibilityTraces() {
	  gsl_vector_free(_grad);
	  gsl_vector_free(_acum_grad);
	}
	  
	void restart(void) {
	  gsl_vector_set_zero(_acum_grad);
	}
	  
	void learn(const STATE &s, const ACTION &a, double rew) {
	  // Evaluate the TD error
	  double td = _critic.td_error(s, rew);
	    
	  // Update the critic
	  _critic.learn(s, rew);
	    
	  // Update the actor
	  _grad_log_p(_theta_p, _grad, s, a);
	  gsl_vector_scale(_acum_grad, _gamma * _lambda_p);
	  gsl_vector_add(_acum_grad, _grad);
	  gsl_blas_daxpy(td*_alpha_p, _acum_grad, _theta_p);
	}

	void learn(const STATE &s, const ACTION &a, double rew, const STATE &s_) {
	  // Evaluate the TD error
	  double td = _critic.td_error(s, rew, s_);

	  // Update the critic
	  _critic.learn(s, rew, s_);
	    
	  // Update the actor
	  _grad_log_p(_theta_p, _grad, s, a);
	  gsl_vector_scale(_acum_grad, _gamma * _lambda_p);
	  gsl_vector_add(_acum_grad, _grad);
	  gsl_blas_daxpy(td*_alpha_p, _acum_grad, _theta_p);
	}
      };

      template<typename STATE, typename ACTION,
	       typename CRITIC, typename fctGRAD_LOGP_PARAMETRIZED>
      EligibilityTraces<STATE, ACTION, CRITIC> 
      eligibility_traces(CRITIC& critic, double gamma, gsl_vector* theta_p, double alpha_p, double lambda_p, const fctGRAD_LOGP_PARAMETRIZED& grad_log_p) {
	return EligibilityTraces<STATE, ACTION, CRITIC>(critic, gamma, theta_p, alpha_p, lambda_p, grad_log_p);
      }
      
    } // ActorVCritic

    namespace ActorQCritic {
	
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
      template<typename STATE, typename ACTION,
	       typename QCRITIC>
      class OneStep {

      private:
	QCRITIC& _critic;
	gsl_vector* _theta_p;
	gsl_vector* _grad;
	double _alpha_p;
	std::function<void(const gsl_vector*, gsl_vector*, const STATE&, const ACTION&)> _grad_log_p; 

      public:

	template<typename fctGRAD_LOGP_PARAMETRIZED>
	OneStep(QCRITIC& critic,
		gsl_vector* theta_p,
		double alpha_p,
		const fctGRAD_LOGP_PARAMETRIZED& grad_log_p):
	  _critic(critic), 
	  _theta_p(theta_p),
	  _grad(gsl_vector_alloc(theta_p->size)),
	  _alpha_p(alpha_p),
	  _grad_log_p(grad_log_p){
	}

	~OneStep() {
	  gsl_vector_free(_grad);
	}
	    
	void learn(const STATE &s, const ACTION &a, double rew) {
	  // Evaluate the TD error for later updating the actor
	  double td = _critic.td_error(s, a, rew);
	    
	  // Update the critic
	  _critic.learn(s, rew);
	    
	  // Update the actor
	  _grad_log_p(_theta_p, _grad, s, a);
	  gsl_blas_daxpy(td*_alpha_p, _grad, _theta_p);
	}

	void learn(const STATE &s, const ACTION &a, double rew, const STATE &s_, const STATE &a_) {
	  // Evaluate the TD error
	  double td = _critic.td_error(s, a, rew, s_, a_);;
	    
	  // Update the critic
	  _critic.learn(s, rew, s_);
	    
	  // Update the actor
	  _grad_log_p(_theta_p, _grad, s, a);
	  gsl_blas_daxpy(td*_alpha_p, _grad, _theta_p);
	}
      };

      template<typename STATE, typename ACTION,
	       typename QCRITIC, typename fctGRAD_LOGP_PARAMETRIZED>
      OneStep<STATE, ACTION, QCRITIC> 
      one_step(QCRITIC& critic, gsl_vector* theta_p, double alpha_p, const fctGRAD_LOGP_PARAMETRIZED& grad_log_p) {
	return OneStep<STATE, ACTION, QCRITIC>(critic, theta_p, alpha_p, grad_log_p);
      }
	
      /**
       * @short Actor-Critic with Eligibility Traces (episodic)
       * Refer to the algorithm Chap 13 of Reinforcement Learning:
       * An Introduction, R. Sutton, A. Barto 2017, June 19
       * This implementation does not incorporate the discount I
       * in the value/policy update step
       * because it leads to significantly slower learning
       * experimented on the cliff walking experiment with Tabular 
       * state representation with linear value/policy
       * This actor critic implements eligibility traces
       * for training the actor. It is your responsibility to use a critic
       * with eligibility traces
       */	
      template<typename STATE, typename ACTION,
	       typename QCRITIC>
      class EligibilityTraces {
	
      private:
	QCRITIC& _critic;
	double _gamma;
	gsl_vector* _theta_p;
	gsl_vector* _grad;
	gsl_vector* _acum_grad;
	double _alpha_p, _lambda_p;
	std::function<void(const gsl_vector*, gsl_vector*, const STATE&, const ACTION&)> _grad_log_p; 

      public:

	template<typename fctGRAD_LOGP_PARAMETRIZED>
	EligibilityTraces(QCRITIC& critic,
			  double gamma,
			  gsl_vector* theta_p,
			  double alpha_p,
			  double lambda_p,
			  const fctGRAD_LOGP_PARAMETRIZED& grad_log_p):
	  _critic(critic),
	  _gamma(gamma),
	  _theta_p(theta_p),
	  _grad(gsl_vector_alloc(theta_p->size)),
	  _acum_grad(gsl_vector_alloc(theta_p->size)),
	  _alpha_p(alpha_p),
	  _lambda_p(lambda_p),
	  _grad_log_p(grad_log_p) {
	  gsl_vector_set_zero(_acum_grad);
	}

	~EligibilityTraces() {
	  gsl_vector_free(_grad);
	  gsl_vector_free(_acum_grad);
	}
	  
	void restart(void) {
	  gsl_vector_set_zero(_acum_grad);
	}
	  
	void learn(const STATE &s, const ACTION &a, double rew) {
	  // Evaluate the TD error
	  double td = _critic.td_error(s, a, rew);
	    
	  // Update the critic
	  _critic.learn(s, rew);
	    
	  // Update the actor
	  _grad_log_p(_theta_p, _grad, s, a);
	  gsl_vector_scale(_acum_grad, _gamma * _lambda_p);
	  gsl_vector_add(_acum_grad, _grad);
	  gsl_blas_daxpy(td*_alpha_p, _acum_grad, _theta_p);
	}

	void learn(const STATE &s, const ACTION &a, double rew, const STATE &s_, const ACTION& a_) {
	  // Evaluate the TD error
	  double td = _critic.td_error(s, a, rew, s_, a_);

	  // Update the critic
	  _critic.learn(s, rew, s_);
	    
	  // Update the actor
	  _grad_log_p(_theta_p, _grad, s, a);
	  gsl_vector_scale(_acum_grad, _gamma * _lambda_p);
	  gsl_vector_add(_acum_grad, _grad);
	  gsl_blas_daxpy(td*_alpha_p, _acum_grad, _theta_p);
	}
      };

      template<typename STATE, typename ACTION,
	       typename QCRITIC, typename fctGRAD_LOGP_PARAMETRIZED>
      EligibilityTraces<STATE, ACTION, QCRITIC> 
      eligibility_traces(QCRITIC& critic, double gamma, gsl_vector* theta_p, double alpha_p, double lambda_p, const fctGRAD_LOGP_PARAMETRIZED& grad_log_p) {
	return EligibilityTraces<STATE, ACTION, QCRITIC>(critic, gamma, theta_p, alpha_p, lambda_p, grad_log_p);
      }
      
    } // ActorQCritic


    
  } // gsl
} // rl
