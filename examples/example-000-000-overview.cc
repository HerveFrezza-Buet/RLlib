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



// This example is an overview that gives you the taste of the
// library. You may have to read the next examples to have a better
// understanding of the details. It implements sarsa for the
// cliff-walking problem.

#include <iostream>
#include <iomanip>
#include <string>
#include <array>
#include <iterator> 
#include <gsl/gsl_vector.h> 

// This is the rl header
#include <rl.hpp>

// These are useful typedefs

typedef rl::problem::cliff_walking::Cliff<30,15>             Cliff;      // World size.
typedef rl::problem::cliff_walking::Param                    Param;      // The default parameters (reward values).
typedef rl::problem::cliff_walking::Simulator<Cliff,Param>   Simulator;  // This is the dynamical system to control.
typedef Simulator::reward_type                               Reward; 
typedef Simulator::observation_type                          S;
typedef Simulator::action_type                               A;

// In reinforcement learning, the main object used for learning is a
// state transition. Let us use our own type to store the transition
// elements.
struct Transition {
    S      s;
    A      a;
    Reward r;
    S      s_; // read s_ as s'
    bool   is_terminal;
};

std::string string_of_action(A a) {
    std::string res;
    switch(a) {
        case rl::problem::cliff_walking::Action::actionNorth: res = "North"; break;
        case rl::problem::cliff_walking::Action::actionSouth: res = "South"; break;
        case rl::problem::cliff_walking::Action::actionEast:  res = "East "; break;
        case rl::problem::cliff_walking::Action::actionWest:  res = "West "; break;
        default:                                      res = "?????";
    }
    return res;
}

// This prints a transition.
std::ostream& operator<<(std::ostream& os, const Transition& t) {
    os << std::setw(3) << t.s  << ' ' << string_of_action(t.a)
        << " ---" << std::setw(5) << t.r << " ---> ";
    if(t.is_terminal)
        os << "End-of-Episode";
    else
        os << std::setw(3) << t.s_;
    return os;
}

// This functions makes a transition from its elements.
Transition make_transition(S s, A a, Reward r, S s_) {
    return {s,a,r,s_,false};
}
Transition make_terminal_transition(S s, A a, Reward r) {
    return {s,a,r,s /* unused */,true};
}

// Let us define the parameters.
#define paramGAMMA   .99
#define paramALPHA   .05
#define paramEPSILON .7

// The Q-function is tabular, i.e. the Q(s,a) values are stored in a
// vector. As the rllib is oriented toward function approximation for
// Q functions, dealing with some tabular representation requires an
// encapsulation of the table, since a tabular representation is a
// specific case of a more general function representation.

// These are definitions for associating the index in a
// monodimentional array to an (s,a) pair. For the cliff-walking
// simulator, actions are consecutive enum values starting from
// 0. This simplifies the TABULAR_Q_RANK macro. States start from 0 as
// well.
#define S_CARDINALITY         Cliff::size
#define A_CARDINALITY         rl::problem::cliff_walking::actionSize
#define TABULAR_Q_CARDINALITY S_CARDINALITY*A_CARDINALITY  // Array size for storing the Q[s,a].
#define TABULAR_Q_RANK(s,a)   (static_cast<int>(a)*S_CARDINALITY+s)            // Index of the Q[s,a] value in the monodimentional array.

// This method simply retrives a q value from a gsl vector.
double q_parametrized(const gsl_vector* theta,
        S s, A a) { 
    return gsl_vector_get(theta,TABULAR_Q_RANK(s,a));
}

// In the Q-Learning algorithm, updates are made according to the
// gradient of the Q-function according to its parameters, taken at
// some specific (s,a) value. With a tabular coding here, this
// gradient is straightforward, since it is a (00..00100..00) vector
// with a 1 at the (s,a) rank position.
void grad_q_parametrized(const gsl_vector* theta,   
        gsl_vector* grad_theta_sa,
        S s, A a) {
    gsl_vector_set_basis(grad_theta_sa,TABULAR_Q_RANK(s,a));
}


using namespace std::placeholders;



// Let us start some experiment
int main(int argc, char* argv[]) {

    std::random_device rd;
    std::mt19937 gen(rd());

    // We need to provide iterators for enumerating all the state and action
    // values. This can be done easily from an enumerators.
    auto action_begin = rl::enumerator<A>(rl::problem::cliff_walking::Action::actionNorth);
    auto action_end   = action_begin + rl::problem::cliff_walking::actionSize;
    auto state_begin  = rl::enumerator<S>(Cliff::start);
    auto state_end    = state_begin + Cliff::size;


    // This is the dynamical system we want to control.
    Param      param;
    Simulator  simulator(param);            

    // Our Q-function is determined by some vector parameter. It is a
    // gsl_vector since we use the GSL-based algorithm provided by the
    // library.
    gsl_vector* theta = gsl_vector_alloc(TABULAR_Q_CARDINALITY);
    gsl_vector_set_zero(theta);

    // If we need to use the Q-function parametrized by theta as q(s,a),
    // we only have to bind our q_from_table function and get a
    // functional object.
    auto q = std::bind(q_parametrized,theta,_1,_2);

    // Let us now define policies, related to q. The learning policy
    // used is an epsilon-greedy one in the following, while we test the
    // learned Q-function with a geedy policy.
    double epsilon       = paramEPSILON;
    auto learning_policy = rl::policy::epsilon_greedy(q,epsilon,action_begin,action_end, gen);
    auto test_policy     = rl::policy::greedy(q,action_begin,action_end);

    // We intend to learn q on-line, by running episodes, and updating a
    // critic fro the transition we get during the episodes. Let us use
    // some GSL-based critic for that purpose.
    auto critic = rl::gsl::sarsa<S,A>(theta,
            paramGAMMA,paramALPHA,
            q_parametrized,
            grad_q_parametrized);

    // We have now all the elements to start experiments.


    // Let us run 10000 episodes with the agent that learns the Q-values.

    std::cout << "Learning " << std::endl
        << std::endl;

    int episode;
    for(episode = 0; episode < 10000; ++episode) {
        simulator.restart();
        auto actual_episode_length = rl::episode::learn(simulator,learning_policy,critic,
                0);
        if(episode % 200 == 0)
            std::cout << "episode " << std::setw(5) << episode+1 
                << " : length = " << std::setw(5) << actual_episode_length << std::endl;
    }
    std::cout << std::endl;

    // Let us print the parameters. This can be dumped in a file, rather
    // than printed, for saving the learned Q-value function.
    std::cout << "Learned theta : " << std::endl
        << std::endl
        << theta << std::endl
        << std::endl;


    // Let us define v as v(s) = max_a q(s_a) with a labda function.
    auto v = [&action_begin,&action_end,&q](S s) -> double {return rl::max(std::bind(q,s,_1),
            action_begin,
            action_end);};
    // We can draw the Value function a image file.
    auto v_range = rl::range(v,state_begin,state_end); 
    std::cout << std::endl
        << " V in [" << v_range.first << ',' << v_range.second << "]." << std::endl
        << std::endl;
    Cliff::draw("V-overview",0,v,v_range.first,v_range.second);
    std::cout << "Image file \"V-overview-000000.ppm\" generated." << std::endl
        << std::endl;

    // Let us be greedy on the policy we have found, using the greedy
    // agent to run an episode.
    simulator.restart();
    unsigned int nb_steps = rl::episode::run(simulator,test_policy,0);
    std::cout << "Best policy episode ended after " << nb_steps << " steps." << std::endl;

    // We can also gather the transitions from an episode into a collection.
    std::vector<Transition> transition_set;
    simulator.restart();
    nb_steps = rl::episode::run(simulator,test_policy,
            std::back_inserter(transition_set),
            make_transition,make_terminal_transition,
            0);
    std::cout << std::endl
        << "Collected transitions :" << std::endl
        << "---------------------" << std::endl
        << nb_steps << " == " << transition_set.size() << std::endl
        << std::endl;
    for(auto& t : transition_set)
        std::cout << t << std::endl;


    gsl_vector_free(theta);
    return 0;
}

