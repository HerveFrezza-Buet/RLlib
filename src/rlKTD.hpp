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

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <cmath>
#include <functional>

#include <rlException.hpp>
#include <rlAlgo.hpp>
#include <rlTypes.hpp>

namespace rl {

    namespace gsl {

        template<typename STATE,
            typename ACTION,
            typename fctQ_PARAMETRIZED,
            typename RANDOM_GENERATOR>
                class KTD {
                    public:

                        using self_type = KTD<STATE,ACTION,fctQ_PARAMETRIZED, RANDOM_GENERATOR>;


                        double gamma;
                        double eta_noise;             // default    0
                        double observation_noise;     // default    1
                        double prior_var;             // default   10
                        double random_amplitude;      // default    0
                        double ut_alpha;              // default 1e-1
                        double ut_beta;               // default    2
                        double ut_kappa;              // default    0
                        bool   use_linear_evaluation; // default false, use true for linear methods, i.e q(theta,s,a) = theta.phi(s,a).


                    protected:


                        gsl_vector* theta;
                        unsigned int theta_size;
                        unsigned int theta_bound;
                        gsl_matrix* sigmaTheta;
                        gsl_matrix* sigmaPointsSet;
                        gsl_matrix *U;
                        gsl_vector *D;
                        gsl_vector *y;
                        gsl_vector *ktdQ_images_SP;
                        gsl_vector *P_theta_r;
                        gsl_vector *kalmanGain;
                        gsl_vector *centeredSP;

                        double w_m0;
                        double w_c0;
                        double w_i;
                        double lambdaUt;


                        std::function<double (const gsl_vector*, const STATE&, const ACTION&)> q;

                        void read(std::istream& is) {
                            is >> w_m0 
                                >> w_c0
                                >> w_i
                                >> theta
                                >> sigmaTheta
                                >> sigmaPointsSet;
                        }

                        void write(std::ostream& os) const {
                            os << w_m0 << ' '
                                << w_c0 << ' '
                                << w_i << ' '
                                << theta
                                << sigmaTheta
                                << sigmaPointsSet;
                        }

                        friend std::ostream& operator<<(std::ostream& os, const self_type& ktd) {
                            ktd.write(os);
                            return os;
                        }

                        friend std::istream& operator>>(std::istream& is, self_type& ktd) {
                            ktd.read(is);
                            return is;
                        }

                        virtual double nextValue(const STATE& next_state,
                                const ACTION& next_action,
                                unsigned int i) const = 0;

                    private:


                        void initWeights(void) {
                            lambdaUt = ut_alpha*ut_alpha*(theta_size+ut_kappa) - theta_size ;
                            w_m0 = lambdaUt/(theta_size+lambdaUt) ;
                            w_c0 = w_m0 + 1 - ut_alpha*ut_alpha + ut_beta ;
                            w_i = 1.0/(2*(theta_size+lambdaUt)) ;
                        }


                        void centralDifferencesTransform(void){

                            // allocate size L  
                            unsigned int i;
                            gsl_vector_view this_sigmaPoint;
                            gsl_vector_view that_sigmaPoint;

                            // Then the sigma points can be computed
                            // first i=0;
                            this_sigmaPoint = gsl_matrix_column(sigmaPointsSet,0);
                            gsl_vector_memcpy(&(this_sigmaPoint.vector), theta);
                            // the 1<= i <= L
                            for(i=1;i<theta_size+1; ++i){
                                this_sigmaPoint = gsl_matrix_column(sigmaPointsSet,i);
                                gsl_vector_memcpy(&(this_sigmaPoint.vector), theta);
                                that_sigmaPoint = gsl_matrix_column(sigmaTheta,i-1);
                                gsl_blas_daxpy(sqrt(theta_size+lambdaUt), &(that_sigmaPoint.vector), &(this_sigmaPoint.vector) );
                            }
                            // the L+1<= i <= 2*L
                            for(i=theta_size+1;i<theta_bound; ++i){
                                this_sigmaPoint = gsl_matrix_column(sigmaPointsSet,i);
                                gsl_vector_memcpy(&(this_sigmaPoint.vector), theta);
                                that_sigmaPoint = gsl_matrix_column(sigmaTheta,i-1-theta_size);
                                gsl_blas_daxpy(-sqrt(theta_size+lambdaUt) , &(that_sigmaPoint.vector), &(this_sigmaPoint.vector) );
                            }
                        }

                        void choleskyUpdate(double alpha, gsl_vector *x){

                            /*
                               This function performs a cholesky update of the cholesky factorization sigmaTheta, that is it replaces 
                               the Cholesky factorization sigmaTheta by the cholesky factorization of 
                               sigmaTheta*sigmaTheta^T - alpha * x * x^T

                               The algorithm is an adaptation of a LU factorization rank one update. Reference is :
                               Peter Strange, Andreas Griewank and Matthias Bollhöfer.
                               On the Efficient Update of Rectangular LU Factorizations subject to Low Rank Modifications.
                               Electronic Transactions on Numerical Analysis, 26:161-177, 2007.
                               alg. is given in left part of fig.2.1. 

                               Perhaps a more efficient algorithm exists, however it should do the work for now. And the code is probably not optimal...

                               WARNING
                               */

                            unsigned int i,j;
                            double tmp;



                            // A first thing is to set SS' (chol factor) in a LU form, L being unitriangular
                            // Compute U = L^T and D = diag(L)
                            gsl_matrix_set_zero(U);
                            for(i=0; i<theta_size; ++i){
                                gsl_vector_set(D,i,gsl_matrix_get(sigmaTheta,i,i));
                                for(j=0; j<=i; ++j){
                                    gsl_matrix_set(U,j,i,gsl_matrix_get(sigmaTheta,i,j));
                                }
                            }
                            // Replace L by L*D^{-1} and U by D*U
                            for(i=0; i<theta_size; ++i){
                                for(j=0; j<=i; ++j){
                                    tmp = gsl_matrix_get(sigmaTheta,i,j);
                                    tmp /= gsl_vector_get(D,j);
                                    gsl_matrix_set(sigmaTheta,i,j,tmp);
                                    tmp = gsl_matrix_get(U,j,i);
                                    tmp *= gsl_vector_get(D,j);
                                    gsl_matrix_set(U,j,i,tmp);
                                }
                            }

                            // compute the y = alpha x vector
                            gsl_vector_memcpy(y,x);
                            gsl_vector_scale(y,alpha);

                            // perform the rank 1 LU modification
                            for(i=0; i<theta_size; ++i){

                                // diagonal update 
                                tmp = gsl_matrix_get(U,i,i) + gsl_vector_get(x,i)*gsl_vector_get(y,i);
                                gsl_matrix_set(U,i,i,tmp);
                                tmp = gsl_vector_get(y,i);
                                tmp /= gsl_matrix_get(U,i,i);
                                gsl_vector_set(y,i,tmp);

                                for(j=i+1; j<theta_size; ++j){
                                    // L (that is sigmaTheta) update 
                                    tmp = gsl_vector_get(x,j) - gsl_vector_get(x,i)*gsl_matrix_get(sigmaTheta,j,i);
                                    gsl_vector_set(x,j,tmp);
                                    tmp = gsl_matrix_get(sigmaTheta,j,i) + gsl_vector_get(y,i) * gsl_vector_get(x,j);
                                    gsl_matrix_set(sigmaTheta,j,i,tmp);
                                }

                                for(j=i+1; j<theta_size; ++j){
                                    // U update 
                                    tmp = gsl_matrix_get(U,i,j) + gsl_vector_get(x,i)*gsl_vector_get(y,j);
                                    gsl_matrix_set(U,i,j,tmp);
                                    tmp = gsl_vector_get(y,j) - gsl_vector_get(y,i) * gsl_matrix_get(U,i,j);
                                    gsl_vector_set(y,j,tmp);
                                }
                            }

                            // Now we want the chol decomposition
                            // first D = sqrt(diag(U));
                            for(i=0; i<theta_size; ++i){
                                tmp =  gsl_matrix_get(U,i,i);
                                if(tmp<=0)
                                    throw exception::NotPositiveDefiniteMatrix("in ..::KTD::choleskyUpdate");
                                gsl_vector_set(D,i,sqrt(tmp));
                            }
                            // then L = L*D;
                            for(i=0; i<theta_size; ++i){
                                for(j=0; j<theta_size; ++j){
                                    tmp = gsl_matrix_get(sigmaTheta,i,j) * gsl_vector_get(D,j);
                                    gsl_matrix_set(sigmaTheta,i,j,tmp);
                                }
                            }
                            // that's all folks !
                        }


                        void kalmanUpdate(const STATE& state, const ACTION& action,
                                double reward,
                                const STATE& next_state,const ACTION& next_action,
                                bool is_terminal) {

                            // initializations
                            unsigned int i;
                            double d, P_r, pred_r;
                            double qValue;
                            gsl_vector sigmaPoint; /* Not a pointer ! */

                            /*
                               -------Prediction Step ----------------------------------------------------
                               */

                            // nothing to do for thetaPred
                            gsl_matrix_scale(sigmaTheta,sqrt(1+eta_noise));

                            /*
                               -------Compute the sigma-Points and their images ---------------------------
                               */

                            // compute the sigma-points (weights are initialized at the creation of the agent)
                            centralDifferencesTransform();

                            //compute their images
                            if(is_terminal)
                                for(i=0; i<theta_bound; ++i){
                                    sigmaPoint = gsl_matrix_column(sigmaPointsSet,i).vector;
                                    qValue = q(&sigmaPoint,
                                            state,action);
                                    gsl_vector_set(ktdQ_images_SP,i,(double)(qValue));
                                }
                            else {
                                for(i=0; i<theta_bound; ++i){
                                    sigmaPoint = gsl_matrix_column(sigmaPointsSet,i).vector;
                                    qValue = q(&sigmaPoint,state,action);
                                    gsl_vector_set(ktdQ_images_SP,i,
                                            (double)(qValue - gamma * nextValue(next_state,next_action,i)));
                                }
                            }


                            /*
                               -------Compute Statistics of interest --------------------------------------
                               */

                            // predicted reward
                            pred_r = w_m0 * gsl_vector_get(ktdQ_images_SP,0) ;
                            for(i=1;i<theta_bound; ++i){
                                pred_r += w_i * gsl_vector_get(ktdQ_images_SP,i);
                            }


                            // associated variance(s)
                            d = gsl_vector_get(ktdQ_images_SP,0) - pred_r ;
                            P_r = w_c0 * d * d  ;
                            for(i=1; i<theta_bound; ++i){
                                // reward
                                d = gsl_vector_get(ktdQ_images_SP,i) - pred_r ;
                                P_r += w_i * d * d ;
                            }
                            P_r += observation_noise;

                            // Correlation between parameters and reward
                            gsl_vector_set_zero(P_theta_r) ;
                            for(i=1; i<theta_bound; ++i){
                                sigmaPoint = gsl_matrix_column(sigmaPointsSet,i).vector;
                                gsl_vector_memcpy(centeredSP, &sigmaPoint) ;
                                gsl_vector_sub(centeredSP,theta) ;
                                gsl_blas_daxpy(w_i * (gsl_vector_get(ktdQ_images_SP,i) - pred_r), centeredSP, P_theta_r) ;
                            }

                            /*
                               -------Correction equations --------------------------------------
                               */

                            // kalman gain
                            gsl_vector_memcpy(kalmanGain, P_theta_r);
                            gsl_vector_scale(kalmanGain, 1.0/P_r);

                            // Update mean
                            gsl_blas_daxpy(reward - pred_r, kalmanGain, theta);

                            // Update Covariance
                            choleskyUpdate(-P_r,kalmanGain);
                        }


                        void paramCopy(const gsl_matrix* sigmaTheta_,
                                const gsl_matrix* sigmaPointsSet_,
                                const gsl_matrix* U_,
                                const gsl_vector* y_,
                                const gsl_vector* D_,
                                const gsl_vector* ktdQ_images_SP_,
                                const gsl_vector* P_theta_r_,
                                const gsl_vector* kalmanGain_,
                                const gsl_vector* centeredSP_) {

                            gsl_matrix_free(sigmaTheta);
                            gsl_matrix_free(sigmaPointsSet);
                            gsl_matrix_free(U);
                            gsl_vector_free(y);
                            gsl_vector_free(D);
                            gsl_vector_free(ktdQ_images_SP);
                            gsl_vector_free(P_theta_r);
                            gsl_vector_free(kalmanGain);
                            gsl_vector_free(centeredSP);

                            theta = 0;
                            sigmaTheta = 0;
                            sigmaPointsSet = 0;
                            U = 0;
                            y = 0;
                            D = 0;
                            ktdQ_images_SP = 0;
                            P_theta_r = 0;
                            kalmanGain = 0;
                            centeredSP = 0;



                            if(sigmaTheta_ != 0) {
                                sigmaTheta = gsl_matrix_alloc(sigmaTheta_->size1,
                                        sigmaTheta_->size2);
                                gsl_matrix_memcpy(sigmaTheta, sigmaTheta_);
                            }

                            if(sigmaPointsSet_ != 0) {
                                sigmaPointsSet = gsl_matrix_alloc(sigmaPointsSet_->size1,
                                        sigmaPointsSet_->size2);
                                gsl_matrix_memcpy(sigmaPointsSet, sigmaPointsSet_);
                            }

                            if(U_ != 0) {
                                U = gsl_matrix_alloc(U_->size1,
                                        U_->size2);
                                gsl_matrix_memcpy(U, U_);
                            }

                            if(y_ != 0) {
                                y = gsl_vector_alloc(y_->size);
                                gsl_vector_memcpy(y, y_);
                            }

                            if(D_ != 0) {
                                D = gsl_vector_alloc(D_->size);
                                gsl_vector_memcpy(D, D_);
                            }

                            if(ktdQ_images_SP_ != 0) {
                                ktdQ_images_SP = gsl_vector_alloc(ktdQ_images_SP_->size);
                                gsl_vector_memcpy(ktdQ_images_SP, ktdQ_images_SP_);
                            }

                            if(P_theta_r_ != 0) {
                                P_theta_r = gsl_vector_alloc(P_theta_r_->size);
                                gsl_vector_memcpy(P_theta_r, P_theta_r_);
                            }

                            if(kalmanGain_ != 0) {
                                kalmanGain = gsl_vector_alloc(kalmanGain_->size);
                                gsl_vector_memcpy(kalmanGain, kalmanGain_);
                            }

                            if(centeredSP_ != 0) {
                                centeredSP = gsl_vector_alloc(centeredSP_->size);
                                gsl_vector_memcpy(centeredSP, centeredSP_);
                            }
                        }


                    public:


                        /**
                         *
                         * @param  eta_noise             default value is    0
                         * @param  observation_noise     default value is    1
                         * @param  prior_var              default value is   10
                         * @param  random_amplitude       default value is    0
                         * @param  ut_alpha              default value is 1e-1
                         * @param  ut_beta               default value is    2
                         * @param  ut_kappa              default value is    0
                         * @param  use_linear_evaluation default value is false, use true for linear methods, i.e q(theta,s,a) = theta.phi(s,a).
                         * @param  gen                   random device used to initialize the parameters (e.g. std::mt19937) 
                         */

                            KTD(gsl_vector* param,
                                    const fctQ_PARAMETRIZED& fct_q,
                                    double param_gamma,
                                    double param_eta_noise,    
                                    double param_observation_noise,  
                                    double param_prior_var,            
                                    double param_random_amplitude,       
                                    double param_ut_alpha,              
                                    double param_ut_beta,              
                                    double param_ut_kappa,              
                                    bool   param_use_linear_evaluation,
                                    RANDOM_GENERATOR& gen) 
                            : gamma(param_gamma),
                            eta_noise(param_eta_noise),    
                            observation_noise(param_observation_noise),  
                            prior_var(param_prior_var),            
                            random_amplitude(param_random_amplitude),       
                            ut_alpha(param_ut_alpha),              
                            ut_beta(param_ut_beta),              
                            ut_kappa(param_ut_kappa),              
                            use_linear_evaluation(param_use_linear_evaluation),
                            theta(param),
                            theta_size(theta->size),
                            theta_bound(2*theta->size+1),
                            sigmaTheta(gsl_matrix_alloc(theta_size,theta_size)),
                            sigmaPointsSet(gsl_matrix_calloc(theta_size,theta_bound)),
                            U(gsl_matrix_alloc(theta_size,theta_size)),
                            D(gsl_vector_alloc(theta_size)),
                            y(gsl_vector_alloc(theta_size)),
                            ktdQ_images_SP(gsl_vector_alloc(theta_bound)),
                            P_theta_r(gsl_vector_alloc(theta_size)),
                            kalmanGain(gsl_vector_alloc(theta_size)),
                            centeredSP(gsl_vector_alloc(theta_size)),
                            q(fct_q) {

                                std::uniform_real_distribution<> dis(-random_amplitude, random_amplitude);
                                for(unsigned int i=0;i<theta_size;++i)
                                    gsl_vector_set(theta,i,dis(gen));
                                gsl_matrix_set_identity(sigmaTheta);
                                gsl_matrix_scale(sigmaTheta,prior_var);
                                initWeights();
                            }

                        KTD(const self_type& cp) 
                            : gamma(cp.gamma),
                            eta_noise(cp.eta_noise),
                            observation_noise(cp.observation_noise),
                            prior_var(cp.prior_var),
                            random_amplitude(cp.random_amplitude),
                            ut_alpha(cp.ut_alpha),
                            ut_beta(cp.ut_beta),
                            ut_kappa(cp.ut_kappa),
                            use_linear_evaluation(cp.use_linear_evaluation),
                            theta(cp.theta),
                            theta_size(cp.theta_size),
                            theta_bound(cp.theta_bound),
                            sigmaTheta(0),
                            sigmaPointsSet(0),
                            U(0),
                            D(0),
                            y(0),
                            ktdQ_images_SP(0),
                            P_theta_r(0),
                            kalmanGain(0),
                            centeredSP(0),
                            q(cp.q) {
                                paramCopy(cp.sigmaTheta,
                                        cp.sigmaPointsSet,
                                        cp.U,
                                        cp.y,
                                        cp.D,
                                        cp.ktdQ_images_SP,
                                        cp.P_theta_r,
                                        cp.kalmanGain,
                                        cp.centeredSP);
                            }

                        self_type& operator=(const self_type& cp) {
                            if(this != &cp) {
                                gamma                 = cp.gamma;
                                eta_noise             = cp.eta_noise;
                                observation_noise     = cp.observation_noise;
                                prior_var             = cp.prior_var;
                                random_amplitude      = cp.random_amplitude;
                                ut_alpha              = cp.ut_alpha;
                                ut_beta               = cp.ut_beta;
                                ut_kappa              = cp.ut_kappa;
                                use_linear_evaluation = cp.use_linear_evaluation;
                                paramCopy(cp.sigmaTheta,
                                        cp.sigmaPointsSet,
                                        cp.U,
                                        cp.y,
                                        cp.D,
                                        cp.ktdQ_images_SP,
                                        cp.P_theta_r,
                                        cp.kalmanGain,
                                        cp.centeredSP);
                                q = cp.q;
                            }
                            return *this;
                        }

                        virtual ~KTD(void) {
                            gsl_matrix_free(sigmaTheta);
                            gsl_matrix_free(sigmaPointsSet);
                            gsl_matrix_free(U);
                            gsl_vector_free(y);
                            gsl_vector_free(D);
                            gsl_vector_free(ktdQ_images_SP);
                            gsl_vector_free(P_theta_r);
                            gsl_vector_free(kalmanGain);
                            gsl_vector_free(centeredSP);
                        }

                        double operator()(const STATE &s, const ACTION &a) const {
                            unsigned int i;
                            double pred_r;
                            double  qval;
                            gsl_vector sigmaPoint; /* not a pointer ! */

                            if(use_linear_evaluation)
                                pred_r = q(theta,s,a);
                            else {
                                for(i=0; i<theta_bound; ++i){
                                    sigmaPoint = gsl_matrix_column(sigmaPointsSet,i).vector;
                                    qval = q(&sigmaPoint,s,a);
                                    gsl_vector_set(ktdQ_images_SP,i,(double)(qval));
                                }

                                pred_r = w_m0 * gsl_vector_get(ktdQ_images_SP,0);
                                for(i=1;i<theta_bound; ++i){
                                    pred_r += w_i * gsl_vector_get(ktdQ_images_SP,i);
                                }
                            }


                            return pred_r;
                        }

                        double operator()(const STATE &s, const ACTION &a, double& variance) const {
                            unsigned int i;
                            double pred_r;
                            double d;

                            pred_r = (*this)(s,a);

                            d = gsl_vector_get(ktdQ_images_SP,0) - pred_r ;
                            variance = w_c0 * d * d  ;
                            for(i=1; i<theta_bound; ++i){
                                // reward
                                d = gsl_vector_get(ktdQ_images_SP,i) - pred_r ;
                                variance += w_i * d * d ;
                            }

                            return pred_r;
                        }

                        void learn(const STATE& s,
                                const ACTION& a,
                                double r) {
                            kalmanUpdate(s,a,r,s,a,true);
                        }

                        void learn(const STATE& s,
                                const ACTION& a,
                                double r,
                                const STATE& s_,
                                const ACTION& a_) {
                            kalmanUpdate(s,a,r,s_,a_,false);
                        }
                };

        /**
         * @short KTDQ algorithm
         *
         */
        template<typename STATE,
            typename ACTION,
            typename ACTION_ITERATOR,
            typename fctQ_PARAMETRIZED,
            typename RANDOM_GENERATOR>
                class KTDQ : public KTD<STATE,ACTION,fctQ_PARAMETRIZED, RANDOM_GENERATOR> {
                    private:

                        ACTION_ITERATOR a_begin;
                        ACTION_ITERATOR a_end;

                    public:

                        using super_type = KTD<STATE,ACTION,fctQ_PARAMETRIZED,RANDOM_GENERATOR>;
                        using self_type  = KTDQ<STATE,ACTION,ACTION_ITERATOR,fctQ_PARAMETRIZED,RANDOM_GENERATOR>; 

                            KTDQ(gsl_vector* param,
                                    const fctQ_PARAMETRIZED& fct_q,
                                    const ACTION_ITERATOR& begin, const ACTION_ITERATOR& end,
                                    double param_gamma,
                                    double param_eta_noise,    
                                    double param_observation_noise,  
                                    double param_prior_var,            
                                    double param_random_amplitude,       
                                    double param_ut_alpha,              
                                    double param_ut_beta,              
                                    double param_ut_kappa,              
                                    bool   param_use_linear_evaluation,
                                    RANDOM_GENERATOR& gen)
                            : super_type(param,fct_q,
                                    param_gamma,
                                    param_eta_noise, 
                                    param_observation_noise,  
                                    param_prior_var, 
                                    param_random_amplitude,  
                                    param_ut_alpha, 
                                    param_ut_beta,      
                                    param_ut_kappa,  
                                    param_use_linear_evaluation,
                                    gen),
                            a_begin(begin), 
                            a_end(end) {}

                        KTDQ(const self_type& cp) : super_type(cp), a_begin(cp.a_begin), a_end(cp.a_end) {}

                        self_type& operator=(const self_type& cp) {
                            if(this != &cp) {
                                this->super_type::operator=(cp);
                                a_begin = cp.a_begin;
                                a_end   = cp.a_end;
                            }
                            return *this;
                        }


                    protected:

                        virtual double nextValue(const STATE& next_state,
                                const ACTION& next_action,
                                unsigned int i) const {
                            gsl_vector sigmaPoint = gsl_matrix_column(this->sigmaPointsSet,i).vector;
                            return rl::max(std::bind(this->q,&sigmaPoint,next_state,std::placeholders::_1),
                                    a_begin,a_end);
                        }
                };


        /**
         *
         * @param  eta_noise             default value is    0
         * @param  observation_noise     default value is    1
         * @param  prior_var              default value is   10
         * @param  random_amplitude       default value is    0
         * @param  ut_alpha              default value is 1e-1
         * @param  ut_beta               default value is    2
         * @param  ut_kappa              default value is    0
         * @param  use_linear_evaluation default value is false, use true for linear methods, i.e q(theta,s,a) = theta.phi(s,a).
         * @param  gen                   random device used to initialize the parameters (e.g. std::mt19937) 
         */
        template<typename STATE,
            typename ACTION,
            typename ACTION_ITERATOR,
            typename fctQ_PARAMETRIZED,
            typename RANDOM_GENERATOR>
                KTDQ<STATE,ACTION,ACTION_ITERATOR,fctQ_PARAMETRIZED,RANDOM_GENERATOR> ktd_q(gsl_vector* param,
                        const fctQ_PARAMETRIZED& fct_q,
                        const ACTION_ITERATOR& begin, 
                        const ACTION_ITERATOR& end,
                        double param_gamma,
                        double param_eta_noise,    
                        double param_observation_noise,  
                        double param_prior_var,            
                        double param_random_amplitude,       
                        double param_ut_alpha,              
                        double param_ut_beta,              
                        double param_ut_kappa,              
                        bool   param_use_linear_evaluation,
                        RANDOM_GENERATOR& gen) {
                    return KTDQ<STATE,ACTION,ACTION_ITERATOR,fctQ_PARAMETRIZED, RANDOM_GENERATOR>(param,fct_q,begin,end,
                            param_gamma,
                            param_eta_noise,    
                            param_observation_noise,  
                            param_prior_var,            
                            param_random_amplitude,       
                            param_ut_alpha,              
                            param_ut_beta,              
                            param_ut_kappa,              
                            param_use_linear_evaluation,
                            gen);
                }



        /**
         * @short KTDSARSA algorithm
         *
         */
        template<typename STATE,
            typename ACTION,
            typename fctQ_PARAMETRIZED,
            typename RANDOM_GENERATOR>
                class KTDSARSA : public KTD<STATE,ACTION,fctQ_PARAMETRIZED,RANDOM_GENERATOR> {

                    public:

                        using super_type = KTD<STATE,ACTION,fctQ_PARAMETRIZED,RANDOM_GENERATOR>;
                        using self_type  = KTDSARSA<STATE,ACTION,fctQ_PARAMETRIZED,RANDOM_GENERATOR>;

                            KTDSARSA(gsl_vector* param,
                                    const fctQ_PARAMETRIZED& fct_q,
                                    double param_gamma,
                                    double param_eta_noise,    
                                    double param_observation_noise,  
                                    double param_prior_var,            
                                    double param_random_amplitude,       
                                    double param_ut_alpha,              
                                    double param_ut_beta,              
                                    double param_ut_kappa,              
                                    bool   param_use_linear_evaluation,
                                    RANDOM_GENERATOR& gen)
                            : super_type(param,fct_q,
                                    param_gamma,
                                    param_eta_noise, 
                                    param_observation_noise,  
                                    param_prior_var, 
                                    param_random_amplitude,  
                                    param_ut_alpha, 
                                    param_ut_beta,      
                                    param_ut_kappa,  
                                    param_use_linear_evaluation,
                                    gen) {}

                        KTDSARSA(const self_type& cp) : super_type(cp){}

                        self_type& operator=(const self_type& cp) {
                            if(this != &cp)
                                this->super_type::operator=(cp);
                            return *this;
                        }


                    protected:

                        virtual double nextValue(const STATE& next_state,
                                const ACTION& next_action,
                                unsigned int i) const {
                            gsl_vector sigmaPoint = gsl_matrix_column(this->sigmaPointsSet,i).vector;
                            return this->q(&sigmaPoint,next_state,next_action);
                        }
                };


        /**
         *
         * @param  eta_noise             default value is    0
         * @param  observation_noise     default value is    1
         * @param  prior_var              default value is   10
         * @param  random_amplitude       default value is    0
         * @param  ut_alpha              default value is 1e-1
         * @param  ut_beta               default value is    2
         * @param  ut_kappa              default value is    0
         * @param  use_linear_evaluation default value is false, use true for linear methods, i.e q(theta,s,a) = theta.phi(s,a).
         * @param  gen                   random device used to initialize the parameters (e.g. std::mt19937) 
         */
        template<typename STATE,
            typename ACTION,
            typename fctQ_PARAMETRIZED,
            typename RANDOM_GENERATOR>
                KTDSARSA<STATE,ACTION,fctQ_PARAMETRIZED, RANDOM_GENERATOR> ktd_sarsa(gsl_vector* param,
                        const fctQ_PARAMETRIZED& fct_q,
                        double param_gamma,
                        double param_eta_noise,    
                        double param_observation_noise,  
                        double param_prior_var,            
                        double param_random_amplitude,       
                        double param_ut_alpha,              
                        double param_ut_beta,              
                        double param_ut_kappa,              
                        bool   param_use_linear_evaluation,
                        RANDOM_GENERATOR& gen) {
                    return KTDSARSA<STATE,ACTION,fctQ_PARAMETRIZED,RANDOM_GENERATOR>(param,fct_q,
                            param_gamma,
                            param_eta_noise,    
                            param_observation_noise,  
                            param_prior_var,            
                            param_random_amplitude,       
                            param_ut_alpha,              
                            param_ut_beta,              
                            param_ut_kappa,              
                            param_use_linear_evaluation, gen);
                }







    }
}





