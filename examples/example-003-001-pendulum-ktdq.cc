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

/*
   This example shows how to use a parametric representation of the
   Q-function and apply KTD-Q. The inverted pendulum problem is solved
   here. It also provide the computed variance (useless here, but show how it can be obtained).
   */

#include <rl.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <cmath>
#include <functional>

using namespace std::placeholders;

// We define our own parameters for the inverted pendulum
class ipParams{
    public:
        // This is the amplitude of the noise (relative) applied to the action.
        inline static double actionNoise(void)        {return  0.2;}
        // This is the noise of angle perturbation from the equilibrium state at initialization.
        inline static double angleInitNoise(void)     {return 1e-3;}
        // This is the noise of speed perturbation from the equilibrium state at initialization.
        inline static double speedInitNoise(void)     {return 1e-3;}
};

// This is our simulator.
using Simulator = rl::problem::inverted_pendulum::Simulator<ipParams, std::mt19937>;

// Definition of Reward, S, A, Transition and TransitionSet.
#include "example-defs-transition.hpp"

// Features and a RBF architecture.
#include "example-defs-pendulum-architecture.hpp"


// Let us define the parameters.
#define paramGAMMA                    .95

#define paramETA_NOISE                 0
#define paramOBSERVATION_NOISE         1
#define paramPRIOR_VAR                10
#define paramRANDOM_AMPLITUDE          0
#define paramUT_ALPHA               1e-1
#define paramUT_BETA                   2
#define paramUT_KAPPA                  0
#define paramUSE_LINEAR_EVALUATION  true

#define NB_OF_EPISODES         1000
#define NB_LENGTH_SAMPLES         5
#define MAX_EPISODE_LENGTH     3000
#define TEST_PERIOD             100

#include "example-defs-test-iteration.hpp"
#include "example-defs-ktdq-experiments.hpp"

int main(int argc, char* argv[]) {


    std::random_device rd;
    std::mt19937 gen(rd());

    gsl_vector* theta = gsl_vector_alloc(PHI_RBF_DIMENSION);
    gsl_vector_set_zero(theta);
    gsl_vector* tmp = gsl_vector_alloc(PHI_RBF_DIMENSION);
    gsl_vector_set_zero(tmp);

    auto q_parametrized = [tmp](const gsl_vector* th,S s, A a) -> Reward {double res;
        phi_rbf(tmp,s,a);           // phi_sa = phi(s,a)
        gsl_blas_ddot(th,tmp,&res); // res    = th^T  . phi_sa
        return res;};

    auto q = std::bind(q_parametrized,theta,_1,_2);


    rl::enumerator<A> a_begin(rl::problem::inverted_pendulum::Action::actionNone);
    rl::enumerator<A> a_end = a_begin+3;


    auto critic = rl::gsl::ktd_q<S,A>(theta,
            q_parametrized,
            a_begin,a_end,
            paramGAMMA,
            paramETA_NOISE, 
            paramOBSERVATION_NOISE, 
            paramPRIOR_VAR,    
            paramRANDOM_AMPLITUDE,    
            paramUT_ALPHA,         
            paramUT_BETA,                
            paramUT_KAPPA,                
            paramUSE_LINEAR_EVALUATION, gen);

    make_experiment(critic,q,a_begin,a_end, gen);

    return 0;
}
