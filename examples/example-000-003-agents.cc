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
#include <rl.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <random>

// In the rl library, there are ready to use templates for building
// agents. An agent is something that knows what to do when it is
// given a state. In other words, it implements a policy.

// We will consider here agents built from an available Q-Value
// representation. This Q-Value is the actual critic. Here, we will
// consider a n-armed bandit problem, where the Q value are related
// to a single state. The critic is then only an evaluation of each
// arm (the actions are "play with arm #i"). 


using      S = int;   // a dummy state here
using      A = int;   // An action is the arm number (from 0)
using Reward = double;


// This implements a Q function... the class behaves as any function.
#define NB_ARMS 50
class Q {
    public:
        // Here is a reference to our stored Q values
        std::array<double,NB_ARMS>& tabular_values;

        Q(std::array<double,NB_ARMS>& q_values) : tabular_values(q_values) {}

        // This is mandatory since rl::policy::softmax stores a copy of q.
        Q(const Q& cp) : tabular_values(cp.tabular_values) {}

        double operator()(S s, A a) const {return tabular_values[a];}
};


// Let us define functions for plotting histograms

template<typename QFUNCTION,
    typename ACTION_ITERATOR>
    void plotQ(std::string title, const QFUNCTION q_function, 
            const ACTION_ITERATOR& a_begin, 
            const ACTION_ITERATOR& a_end,
            std::string filename) {
        std::ofstream file;

        file.open(filename.c_str());
        if(!file) {
            std::cerr << "Cannot open \"" << filename << "\". Aborting";
            return;
        }
        file << "set title '" << title << "';" << std::endl
            << "set xrange [0:" << NB_ARMS-1 << "];" << std::endl
            << "set yrange [0:1];" << std::endl
            << "set xlabel 'Actions'" << std::endl
            << "plot '-' with lines notitle" << std::endl;
        S dummy;
        for(auto ait = a_begin; ait != a_end ; ++ait)
            file << *ait << ' ' << q_function(dummy, *ait) << std::endl;

        file.close();
        std::cout << "\"" << filename << "\" generated." << std::endl;
    };

#define HISTO_NB_SAMPLES 20000

template<typename POLICY>
void plot1D(std::string title,const POLICY& policy,std::string filename) {
    std::ofstream file;

    file.open(filename.c_str());
    if(!file) {
        std::cerr << "Cannot open \"" << filename << "\". Aborting";
        return;
    }

    int histogram[NB_ARMS];
    A a;
    S dummy;
    int i;
    double max=0;

    for(a = 0; a < NB_ARMS; ++a)
        histogram[a] = 0;
    for(i = 0; i < HISTO_NB_SAMPLES; ++i)
        histogram[policy(dummy)]++;
    for(a = 0; a < NB_ARMS; ++a)
        if(histogram[a] > max)
            max = histogram[a];
    max /= (double)HISTO_NB_SAMPLES;
    file << "set title '" << title << "';" << std::endl
        << "set xrange [0:" << NB_ARMS-1 << "];" << std::endl
        << "set yrange [0:" << max*1.1 << "];" << std::endl
        << "set xlabel 'Actions'" << std::endl
        << "plot '-' with lines notitle" << std::endl;
    for(a = 0; a < NB_ARMS; ++a)
        file << a << ' ' << histogram[a]/(double)HISTO_NB_SAMPLES << std::endl;

    file.close();
    std::cout << "\"" << filename << "\" generated." << std::endl;
}

template<typename POLICY>
void plot2D(POLICY& policy, double& temperature) {
    std::ofstream file;

    file.open("SoftMaxPolicy.plot");
    if(!file) {
        std::cerr << "Cannot open \"SoftMaxPolicy.plot\". Aborting";
        return;
    }

    int histogram[NB_ARMS];
    A a;
    S dummy;
    int i;
    int tpt;

    file << "set title 'SoftMax policy action choices';" << std::endl
        << "set xrange [0:" << NB_ARMS-1 << "];" << std::endl
        << "set xlabel 'Temperature'" << std::endl
        << "set ylabel 'Actions'" << std::endl
        << "set hidden3d;" << std::endl
        << "set ticslevel 0;" << std::endl
        << "splot '-' using 1:2:3 with lines notitle" << std::endl;
    for(tpt = 0, temperature = 100; 
            tpt < 50; 
            file << std::endl, temperature *= .85, ++tpt) {
        for(a = 0; a < NB_ARMS; ++a)
            histogram[a] = 0;
        for(i = 0; i < HISTO_NB_SAMPLES; ++i)
            histogram[policy(dummy)]++;
        for(a = 0; a < NB_ARMS; ++a)
            file << tpt << ' ' << a << ' ' << histogram[a]/(double)HISTO_NB_SAMPLES << std::endl;
        std::cout << "line " << std::setw(3) << tpt+1 << "/50 generated.   \r" << std::flush;
    }

    file.close();
    std::cout << "\"SoftMaxPolicy.plot\" generated.                 " << std::endl;
}

int main(int argc, char* argv[]) {

    std::random_device rd;
    std::mt19937 gen(rd());

    std::array<double,NB_ARMS> q_tab;     // tabular values...
    Q                          q(q_tab);  // ...q handle them.
    A                          a;
    double                     x;

    rl::enumerator<A> a_begin(0);
    rl::enumerator<A> a_end(NB_ARMS);

    auto random_policy         = rl::policy::random(a_begin,a_end,gen);
    auto greedy_policy         = rl::policy::greedy(q,a_begin,a_end);
    double epsilon = .75; 
    auto epsilon_greedy_policy = rl::policy::epsilon_greedy(q,epsilon,
            a_begin,a_end, gen);
    double temperature = 0;
    auto softmax_policy        = rl::policy::softmax(q,temperature,
            a_begin,a_end, gen);

    try {
        // Let us initialize the values with a bi-modal distribution...
        // I have found it empirically while playing with gnuplot.
        for(a = 0; a < NB_ARMS; ++a) {
            x = a/(double)NB_ARMS;
            q.tabular_values[a] = (1-.2*x)*pow(sin(5*(x+.15)),2);
        }
        plotQ("Q values", q, a_begin, a_end, "Qvalues.plot");

        // Let us plot histograms of policys.
        plot1D("Random policy choices",random_policy,"RandomPolicy.plot");
        plot1D("Greedy policy choices",greedy_policy,"GreedyPolicy.plot");
        plot1D("Epsilon-greedy policy choices",epsilon_greedy_policy,"EpsilonGreedyPolicy.plot");

        plot2D(softmax_policy, temperature);
    }
    catch(rl::exception::Any& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
