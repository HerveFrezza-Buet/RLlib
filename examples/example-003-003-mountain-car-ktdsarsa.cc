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
#include <cstdlib>
#include <functional>
#include <random>

using namespace std::placeholders;

// This is our simulator.
typedef rl::problem::mountain_car::DefaultParam        mcParam;
typedef rl::problem::mountain_car::Simulator<mcParam>  Simulator;


// Definition of Reward, S, A, Transition and TransitionSet.
#include "example-defs-transition.hpp"

// Features and a RBF architecture.
#include "example-defs-mountain-car-architecture.hpp"


// Let us define the parameters.
#define paramGAMMA                           .95
#define paramEPSILON                         .1
#define paramETA_NOISE                       1e-5
#define paramOBSERVATION_NOISE               1
#define paramPRIOR_VAR                       10
#define paramRANDOM_AMPLITUDE                1e-1
#define paramUT_ALPHA                        1e-1
#define paramUT_BETA                         2
#define paramUT_KAPPA                        0
#define paramUSE_LINEAR_EVALUATION        true     // We actually use a linear architecture.



typedef rl::problem::mountain_car::Gnuplot<Simulator>  Gnuplot;

#define MAX_EPISODE_LENGTH_LEARN          1500
#define MAX_EPISODE_LENGTH_TEST            300
#define KTDSARSA_FILENAME   "mountain-car.ktdsarsa"

template<typename RANDOM_GENERATOR>
void test(const Simulator::phase_type& start, RANDOM_GENERATOR& gen);

template<typename RANDOM_GENERATOR>
void train(int nb_episodes, bool make_movie, RANDOM_GENERATOR& gen);

int main(int argc, char* argv[]) {
    bool                  learn_mode;
    bool                  movie_mode=false;
    int                   nb_episodes;
    Simulator::phase_type init_phase;
    Simulator             simulator;
    std::string           arg;

    std::random_device rd;
    std::mt19937 gen(rd());

    if(argc < 2) {
        std::cerr << "Usage : " << std::endl
            << "  " << argv[0] << " learn <nb-episodes>   (100 episode should be enough)" << std::endl
            << "  " << argv[0] << " learnandmovie <nb-episodes>   (100 episode should be enough)" << std::endl
            << "  " << argv[0] << " test bottom" << std::endl
            << "  " << argv[0] << " test random" << std::endl
            << "  " << argv[0] << " test <position> <speed>" << std::endl;
        return 0;
    }


    arg = argv[1];

    if(arg == "learnandmovie")
        movie_mode=true;

    if(arg == "learn" || arg == "learnandmovie") {
        learn_mode = true;
        if(argc == 3)
            nb_episodes = atoi(argv[2]);
        else {
            std::cerr << "Bad command syntax. Aborting." << std::endl;
            return 1;
        }
    }
    else if(arg == "test") {
        learn_mode = false;
        if(argc == 3) {
            arg = argv[2];
            if(arg == "bottom")
                init_phase = Simulator::phase_type(Simulator::bottom(),0);
            else if(arg == "random") {
                init_phase = Simulator::phase_type::random(gen);
            }
            else {
                std::cerr << "Bad command syntax. Aborting." << std::endl;
                return 1;
            }
        }
        else if(argc==4)
            init_phase = Simulator::phase_type(atof(argv[2]),atof(argv[3]));
        else {
            std::cerr << "Bad command syntax. Aborting." << std::endl;
            return 1;
        }
    }
    else {
        std::cerr << "Set learning mode to test or learn. Aborting." << std::endl;
        return 1;
    }


    if(learn_mode)
        train(nb_episodes,movie_mode, gen);
    else
        test(init_phase, gen);
    return 0;
}

void execute_command(const std::string& command) {
    int status = std::system(command.c_str());
    if(status != EXIT_SUCCESS) 
        throw std::runtime_error(std::string("Errors raised when executing '" + command + "'"));
}

template<typename RANDOM_GENERATOR>
void train(int nb_episodes, bool make_movie, RANDOM_GENERATOR& gen) {
    int            episode, step, episode_length;
    std::string    command;
    std::ofstream  file;

    Simulator      simulator;
    RBFFeature     phi;

    gsl_vector* theta = gsl_vector_alloc(PHI_RBF_DIMENSION);
    gsl_vector_set_zero(theta);
    gsl_vector* tmp = gsl_vector_alloc(PHI_RBF_DIMENSION);
    gsl_vector_set_zero(tmp);

    auto q_parametrized = [tmp,&phi](const gsl_vector* th,S s, A a) -> Reward {
        double res;
        phi(tmp,s,a);           // phi_sa = phi(s,a)
        gsl_blas_ddot(th,tmp,&res); // res    = th^T  . phi_sa
        return res;};

    auto q = std::bind(q_parametrized,theta,_1,_2);


    // std::array<A, rl::problem::mountain_car::actionSize> actions = {rl::problem::mountain_car::Action::actionBackward, rl::problem::mountain_car::Action::actionNone, rl::problem::mountain_car::Action::actionForward};
    // auto a_begin = actions.begin();
    // auto a_end = actions.end();

    rl::enumerator<A> a_begin(rl::problem::mountain_car::Action::actionNone); // This MUST be the lowest value of the enum type of actions and action enum values are consecutive for mountain_car
    rl::enumerator<A> a_end = a_begin+rl::problem::mountain_car::actionSize;

    double     epsilon = paramEPSILON;
    auto explore_agent = rl::policy::epsilon_greedy(q,epsilon,a_begin,a_end, gen);
    auto greedy_agent  = rl::policy::greedy(q,a_begin,a_end);

    auto critic = rl::gsl::ktd_sarsa<S,A>(theta,
            q_parametrized,
            paramGAMMA,
            paramETA_NOISE, 
            paramOBSERVATION_NOISE, 
            paramPRIOR_VAR,    
            paramRANDOM_AMPLITUDE,    
            paramUT_ALPHA,         
            paramUT_BETA,                
            paramUT_KAPPA,                
            paramUSE_LINEAR_EVALUATION,
            gen);

    try {

        step = 0;
        for(episode = 0; episode < nb_episodes; ++episode) {

            std::cout << "Running episode " << episode+1 << "/" << nb_episodes << "." << std::endl;
            simulator.setPhase(Simulator::phase_type::random(gen)); 
            episode_length = rl::episode::learn(simulator,explore_agent,critic,MAX_EPISODE_LENGTH_LEARN);
            std::cout << "... length is " << episode_length << "." << std::endl;

            ++step;

            if(make_movie)
                Gnuplot::drawQ("KTD Sarsa + RBF",
                        "ktd",step,
                        critic,greedy_agent);
        }

        // Let us save the results.
        file.open(KTDSARSA_FILENAME);
        if(!file)
            std::cerr << "Cannot open \"" << KTDSARSA_FILENAME << "\"." << std::endl;
        else {
            file << std::setprecision(20) << critic;
            file.close();
        }

        if(make_movie) {

            std::string command;

            command = "find . -name \"ktd-*.plot\" -exec gnuplot \\{} \\;";
            std::cout << "Executing : " << command << std::endl;
            execute_command(command.c_str());

            command = "find . -name \"ktd-*.png\" -exec convert \\{} -quality 100 \\{}.jpg \\;";
            std::cout << "Executing : " << command << std::endl;
            execute_command(command.c_str());

            command = "ffmpeg -i ktd-%06d.png.jpg -b 1M rllib.avi";
            std::cout << "Executing : " << command << std::endl;
            execute_command(command.c_str());

            command = "find . -name \"ktd-*.plot\" -exec rm \\{} \\;";
            std::cout << "Executing : " << command << std::endl;
            execute_command(command.c_str());

            command = "find . -name \"ktd-*.png\" -exec rm \\{} \\;";
            std::cout << "Executing : " << command << std::endl;
            execute_command(command.c_str());

            command = "find . -name \"ktd-*.png.jpg\" -exec rm \\{} \\;";
            std::cout << "Executing : " << command << std::endl;
            execute_command(command.c_str());
        }
    }
    catch(rl::exception::Any& e) {
        std::cerr << "Exception caught : " << e.what() << std::endl; 
    }
}

template<typename RANDOM_GENERATOR>
void test(const Simulator::phase_type& start, RANDOM_GENERATOR& gen) {
    std::string    command;
    std::ifstream  file;

    Simulator      simulator;
    RBFFeature     phi;

    gsl_vector* theta = gsl_vector_alloc(PHI_RBF_DIMENSION);
    gsl_vector_set_zero(theta);
    gsl_vector* tmp = gsl_vector_alloc(PHI_RBF_DIMENSION);
    gsl_vector_set_zero(tmp);

    auto q_parametrized = [tmp,&phi](const gsl_vector* th,S s, A a) -> Reward {double res;
        phi(tmp,s,a);           // phi_sa = phi(s,a)
        gsl_blas_ddot(th,tmp,&res); // res    = th^T  . phi_sa
        return res;};

    auto q = std::bind(q_parametrized,theta,_1,_2);


    rl::enumerator<A> a_begin(rl::problem::mountain_car::Action::actionNone); // This MUST be the lowest value of the enum type of actions and action enum values are consecutive for mountain_car
    rl::enumerator<A> a_end = a_begin+rl::problem::mountain_car::actionSize;

    auto greedy_agent  = rl::policy::greedy(q,a_begin,a_end);

    auto critic = rl::gsl::ktd_sarsa<S,A>(theta,
            q_parametrized,
            paramGAMMA,
            paramETA_NOISE, 
            paramOBSERVATION_NOISE, 
            paramPRIOR_VAR,    
            paramRANDOM_AMPLITUDE,    
            paramUT_ALPHA,         
            paramUT_BETA,                
            paramUT_KAPPA,                
            paramUSE_LINEAR_EVALUATION, gen);

    try {

        file.open(KTDSARSA_FILENAME);
        if(!file) {
            std::cerr << "Cannot open \"" << KTDSARSA_FILENAME << "\"." << std::endl;
            ::exit(1);
        }

        // Let us load some critic ...
        file >> critic;

        // ... and run an episode.
        simulator.setPhase(start); 
        Gnuplot::drawEpisode("Mountain car run",
                "mountain-car-run",-1,
                simulator,critic,
                greedy_agent,
                MAX_EPISODE_LENGTH_TEST);
    }
    catch(rl::exception::Any& e) {
        std::cerr << "Exception caught : " << e.what() << std::endl; 
    }
}
