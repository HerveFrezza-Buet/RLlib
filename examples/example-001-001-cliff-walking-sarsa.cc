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


#include <rl.hpp>
#include <random>

using     Cliff = rl::problem::cliff_walking::Cliff<20,6>;
using     Param = rl::problem::cliff_walking::Param;     
using Simulator = rl::problem::cliff_walking::Simulator<Cliff,Param>;

// Definition of Reward, S, A, SA, Transition and TransitionSet.
#include "example-defs-transition.hpp"

// Definition a tabular parametrization of the Q-Value.
#include "example-defs-tabular-cliff.hpp"


// Let us define the parameters.
#define paramGAMMA   .99
#define paramALPHA   .05
#define paramEPSILON .2

// This stores pieces of codes shared by our example experiments.
#include "example-defs-cliff-experiments.hpp"

using namespace std::placeholders;

int main(int argc, char* argv[]) {
    std::random_device rd;
    std::mt19937 gen(rd());

    gsl_vector* theta = gsl_vector_alloc(TABULAR_Q_CARDINALITY);
    gsl_vector_set_zero(theta);

    auto      q = std::bind(q_parametrized,theta,_1,_2);
    auto critic = rl::gsl::sarsa<S,A>(theta,
            paramGAMMA,paramALPHA,
            q_parametrized,
            grad_q_parametrized);

    make_experiment(critic,q, gen);
    gsl_vector_free(theta);
    return 0;
}
