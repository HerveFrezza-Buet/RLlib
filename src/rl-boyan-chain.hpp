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

#include <string>
#include <sstream>
#include <random>

#include <rlException.hpp>
#include <rlAlgo.hpp>

namespace rl {
    namespace problem {
        namespace boyan_chain {

            using Phase = unsigned int;
            enum class Action {actionNone} ;
            constexpr int actionSize = 1;

            /**
             * This is the exception for bad phases.
             */
            class BadPhase : public rl::exception::Any {
                private:

                    std::string error(Phase p) {
                        std::ostringstream os;
                        os << "Bad phase (got " << p << ")";
                        return os.str();
                    }

                public:

                    BadPhase(Phase p,std::string comment) 
                        : Any(error(p)+comment) {}

            };

            class Feature {

                public:
                    using input_type = Phase;

                    int dimension(void) const {return 4;}

                    class BadFeature : public rl::exception::Any {
                        private:

                            std::string error(int dim,const gsl_vector* phi) {
                                std::ostringstream os;

                                os << "Bad Feature size : ";
                                if(phi != 0)
                                    os << "got " << phi->size 
                                        << " instead of " << dim << " : ";
                                return os.str();
                            }

                        public:

                            BadFeature(int dim, const gsl_vector* phi,std::string comment) 
                                : Any(error(dim,phi)+comment) {}

                    };


                private:

                    void check(const gsl_vector* phi) const {
                        if(phi == (gsl_vector*)0)
                            throw BadFeature(dimension(),phi,"null vector");
                        else if((int)(phi->size) != dimension())
                            throw BadFeature(dimension(),phi,"dimensions do not fit");
                    }

                public:

                    void operator()(gsl_vector* phi,const input_type& input) const {
                        gsl_vector_set_zero(phi);
                        check(phi);
                        switch(input) {
                            case 0:
                                gsl_vector_set(phi,3,1); 
                                break;
                            case 1:
                                gsl_vector_set(phi,2,0.25); 
                                gsl_vector_set(phi,3,0.75); 
                                break;
                            case 2:
                                gsl_vector_set(phi,2,0.5); 
                                gsl_vector_set(phi,3,0.5); 
                                break;
                            case 3:
                                gsl_vector_set(phi,2,0.75); 
                                gsl_vector_set(phi,3,0.25); 
                                break;
                            case 4:
                                gsl_vector_set(phi,2,1); 
                                break;
                            case 5:
                                gsl_vector_set(phi,1,0.25); 
                                gsl_vector_set(phi,2,0.75); 
                                break;
                            case 6:
                                gsl_vector_set(phi,1,0.5); 
                                gsl_vector_set(phi,2,0.5); 
                                break;
                            case 7:
                                gsl_vector_set(phi,1,0.75); 
                                gsl_vector_set(phi,2,0.25); 
                                break;
                            case 8:
                                gsl_vector_set(phi,1,1); 
                                break;
                            case 9:
                                gsl_vector_set(phi,0,0.25); 
                                gsl_vector_set(phi,1,0.75); 
                                break;
                            case 10:
                                gsl_vector_set(phi,0,0.5); 
                                gsl_vector_set(phi,1,0.5); 
                                break;
                            case 11:
                                gsl_vector_set(phi,0,0.75); 
                                gsl_vector_set(phi,1,0.25); 
                                break;
                            case 12:
                                gsl_vector_set(phi,0,1); 
                                break;
                            default:
                                throw BadPhase(input,"in rl::problem::boyan_chain::Feature::operator()");
                        }
                    }
            };

            template<typename RANDOM_GENERATOR>
            class Simulator {
                public:

                    using       phase_type = Phase;
                    using      action_type = Action;
                    using observation_type = phase_type ;
                    using      reward_type = double;

                private:

                    phase_type current;
                    reward_type r;
                    RANDOM_GENERATOR gen;

                public:

                    Simulator(RANDOM_GENERATOR& generator) : current(12),r(0), gen(generator()) {}
                    Simulator(const Simulator& other)            = delete;
                    Simulator& operator=(const Simulator& other) = delete;
                    Simulator(Simulator&& other)                 = delete;
                    Simulator& operator=(Simulator&& other)      = delete;

                    ~Simulator(void) {}

                    void setPhase(const phase_type& s) {
                        if(s>12)
                            throw BadPhase(s,"in boyan_chain::Simulator::setPhase");
                    }

                    void initPhase(void) {
                        current = 12;
                    }

                    const observation_type& sense(void) const {
                        return current;
                    }

                    void timeStep(const action_type& a) {
                        std::bernoulli_distribution dis(0.5);
                        if(current>=2) {
                            if(dis(gen))
                                current -= 1;
                            else 
                                current -= 2;
                            r = -3;
                        }
                        else if(current == 1) {
                            current = 0;
                            r = -2;
                        }
                        else {
                            r = 0;
                            throw rl::exception::Terminal("in boyan_chain::Simulator::timeStep");
                        }
                    }


                    reward_type reward(void) const {
                        return r;
                    }
            };

            template<typename RANDOM_GENERATOR>
                Simulator<RANDOM_GENERATOR> make_simulator(RANDOM_GENERATOR& gen) {
                    return Simulator<RANDOM_GENERATOR>(gen);
                }
        }
    }
}
