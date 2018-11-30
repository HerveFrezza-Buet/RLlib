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



// First, you have to include the rllib header file, and the other
// usual includes that you may need.
#include <iostream>
#include <string>
#include <vector>

#include <rl.hpp>
#include <gsl/gsl_vector.h>

// Here, the purpose is to introduce the concepts that are related to
// learning. Let us start by defining a problem to solve. Let us
// consider, once again, a silly problem. The state is a scalar value
// taken in [0,1]. There are 3 action, names 'raise', 'lower' and
// 'none', that add or remove .05 to x. The system gets rewarded when
// the value reaches 1... that is also the terminal state.
class Simulator {
    public:
        typedef double                    phase_type;       
        typedef phase_type                observation_type; 
        typedef enum Action {actionLower,
            actionNone,
            actionRaise} action_type; 
        typedef double                    reward_type;  

    private:
        phase_type x;
        reward_type r;

    public:

        class SomethingWrong :  public rl::exception::Any {
            public: 
                SomethingWrong(std::string comment) 
                    : rl::exception::Any(std::string("Something went wrong : ")
                            + comment) {} 
        };

        Simulator(void)
            : x(.5), r(0) {}
        ~Simulator(void) {}

        void setPhase(const phase_type &s)         {x = s;}
        const observation_type& sense(void) const  {return x;}
        reward_type reward(void) const             {return r;}

        void timeStep (const action_type &a) {
            double dx;

            switch(a) {
                case actionLower: x-=.05; break;
                case actionNone:          break;
                case actionRaise: x+=.05; break;
                default:
                                  throw SomethingWrong("Simulator::timeStep : Bad action.");
            }

            r = 0;
            x += dx;
            if(x<0)
                x = 0;
            else if(x>=1) {
                x=1;
                r=1;
                throw rl::exception::Terminal("1 is reached");
            }
        }
};

// Ok, now let us rename the types with usual names.
typedef Simulator::observation_type S;
typedef Simulator::action_type      A;
typedef Simulator::reward_type      Reward;

// Let us now start with the definition of function-related concepts,
// that are used in the rl-library for learning.

// For linear algorithms, it is sometimes required by the rl library
// to implement a feature space. It consists in representing a (s,a)
// pair by a vector. Let us implement a Gaussian radial basis functions.

// The feature phi(s,a) is a vector handled by a gsl_vector.

void phi(gsl_vector *phi, const S& s, const A& a) {
    int offset;
    double dist;

    if(phi->size != 9)
        throw Simulator::SomethingWrong("Feature::operator() : Bad phi size");

    switch(a) {
        case Simulator::actionLower: offset = 0; break;
        case Simulator::actionNone:  offset = 3; break;
        case Simulator::actionRaise: offset = 6; break;
        default:
                                     throw Simulator::SomethingWrong("Feature::operator()  : Bad action.");
    }

    gsl_vector_set_zero(phi);
    dist = s;    gsl_vector_set(phi,offset,  exp(-20*dist*dist));
    dist = s-.5; gsl_vector_set(phi,offset+1,exp(-20*dist*dist));
    dist = s-1;  gsl_vector_set(phi,offset+2,exp(-20*dist*dist));
}

// Some other algorithms use parametrized functions, to approximate
// Q-values for example. This is what a regression is. Let us define a
// very simple linear represention for the Q values.

Reward q_parametrized(const gsl_vector* theta,
        S s, A act) { 
    double a,b;
    int offset;

    if(theta->size != 6)
        throw Simulator::SomethingWrong("Architecture::operator() : Bad theta size");

    switch(act) {
        case Simulator::actionLower: offset = 0; break;
        case Simulator::actionNone:  offset = 2; break;
        case Simulator::actionRaise: offset = 4; break;
        default:
                                     throw Simulator::SomethingWrong("Architecture::operator()  : Bad action.");
    }

    a = gsl_vector_get(theta,offset);
    b = gsl_vector_get(theta,offset+1);

    return a*s+b;
}

// Nothing to do here, the idea was only to instanciate a feature
// function and a parametrized function, since those concepts are used
// in further examples.
int main(int argc, char* argv[]) {
    std::cout << "That's it." << std::endl;
}


