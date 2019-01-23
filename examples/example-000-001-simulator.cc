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

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <array>
#include <utility>
#include <iterator> 
#include <random>

// Using the rl library implies having a simulator at your
// disposal. This what the agent has to interact with. Your simulator
// has to fit the rl::concept::Simulator. It means that it must define
// the same methods than rl::concept::Simulator. The classes defined
// within the namespace rl::concept play the role of a chart that you
// have to be compliant with.

// Let us define our own simulator. It consists in a 6 letter word,
// whose letters can only be B,O and N. An action consists of pushing
// one of those letters on the left of the word, thus popping away the
// rightmost letter. Reward of 1 is obtained when the word "BONOBO" is
// made. The simulation stops, with a very bad reward, if the word is
// a palindrom. This is quite a silly problem, but let us define it.

class Bonobo {
    private:

        // Here is our internal cuisine.
        std::string word;
        double r;

    public:

        // This is for debugging
        bool verbose;

        // The following methods are also required by the rl::concept::Simulator.
        Bonobo(void) 
            : word("BONBON"), r(0), verbose(false)   {}
        ~Bonobo (void)                             {}
        void setPhase(const std::string &s)        {word = s;}
        const std::string& sense(void) const       {return word;}
        double reward(void) const                  {return r;}

        // Let us define an exception if bad letters are provided.
        class BadLetter : public rl::exception::Any {
            public:
                BadLetter(char letter, std::string comment) 
                    : rl::exception::Any(std::string("Bad letter '")
                            + std::string(1,letter)
                            + std::string("' received : ")
                            + comment) {} 
        };


        // This is where things happen. Usually, this method is the most
        // difficult thing to write when using the library, since it
        // expresses the problem itself.
        void timeStep (const char &a) {
            bool terminated;

            // Let us check the letter
            switch(a) {
                case 'B':
                case 'O':
                case 'N':
                    break;
                default:
                    throw BadLetter(a,"in Bonobo::timeStep");
                    break;
            }

            // ok, now let us push the letter in front of the word.
            std::string tmp = std::string(1,a)+word;
            word = tmp.substr(0,6);

            // Let us check if the simulation is finished (i.e if word is a
            // palindrom). If so, we have to raise the appropriate exception,
            // expected by the rllib algorithms.
            terminated = 
                word[0] == word[5]
                &&  word[1] == word[4]
                &&  word[2] == word[3];

            // Let us compute the reward.
            if(terminated)
                r = -100;
            else if(word == "BONOBO")
                r = 1;
            else
                r = 0;

            if(verbose)
                std::cout << word << " : " << r << std::endl;

            if(terminated)
                throw rl::exception::Terminal(std::string("Word : ") + word);
        }
};

// Ok, now let us rename the types with usual names.

typedef Bonobo      Simulator;
typedef std::string S;
typedef char        A;
typedef double      Reward;

// This gathers actions in an iterable set. We would have used
// rl::enumerable if action values were contiguous.
std::array<A,3> actions = {{'B','O','N'}};


// This function shows how to run an episode with the types we have
// defined so far.
template<typename RANDOM_GENERATOR>
void run_episode_version_01(RANDOM_GENERATOR& gen) {
    Simulator simulator;
    Reward sum = 0;
    auto policy = rl::policy::random(actions.begin(),actions.end(), gen);

    std::cout << std::endl
        << "Version 01" << std::endl
        << "----------" << std::endl
        << std::endl;

    simulator.verbose = true;
    simulator.setPhase("BONBON");
    try {
        while(true) {
            simulator.timeStep(policy(simulator.sense()));
            sum += simulator.reward();
        }
    }
    catch(rl::exception::Terminal& e) {
        sum += simulator.reward();
        std::cout << "Terminated : " << e.what() << std::endl;
    }

    std::cout << "Total reward during episode : " << sum << std::endl;
}

// This function is much simplier, but it does not compute any reward
// while the episode is executed.
template<typename RANDOM_GENERATOR>
void run_episode_version_02(RANDOM_GENERATOR& gen) {
    Simulator simulator;
    auto policy = rl::policy::random(actions.begin(),actions.end(), gen);

    std::cout << std::endl
        << "Version 02" << std::endl
        << "----------" << std::endl
        << std::endl;

    simulator.verbose = true;
    simulator.setPhase("BONBON");
    rl::episode::run(simulator,policy,0);
}

// We can interact with a running episode by an output iterator. Here,
// let us define an output iterator that sums the rewards, and
// collects the actions. The usage af an output iterator is
// *(output++) = value. Here, value is expected to be a
// std::pair<Reward,A>.
class Accum {
    private:
        double& sum_r;
        std::string& act_seq;

    public:

        Accum(double& s, std::string& as) 
            : sum_r(s), act_seq(as) {}
        Accum(const Accum& cp) 
            : sum_r(cp.sum_r), 
            act_seq(cp.act_seq) {}
        Accum& operator*()     {return *this;}
        Accum& operator++(int) {return *this;}        

        void operator=(const std::pair<Reward,A>& ra) {
            sum_r += ra.first;
            act_seq += std::string(1,ra.second);
        }
};

// This function runs an episode. At each step, it uses a lambda
// function for computing the std::pair<Reward,A> expected by the
// output iterator from a (S,A,R,S) tuple. The second lambda function
// is used for terminal transitions (S,A,R).
template<typename RANDOM_GENERATOR>
void run_episode_version_03(RANDOM_GENERATOR& gen) {
    Simulator simulator;
    auto policy = rl::policy::random(actions.begin(),actions.end(), gen);

    std::cout << std::endl
        << "Version 03" << std::endl
        << "----------" << std::endl
        << std::endl;

    simulator.verbose = true;
    simulator.setPhase("BONBON");

    double      sum_r=0; 
    std::string action_sequence;
    Accum       accum(sum_r,action_sequence);
    rl::episode::run(simulator,policy,
            accum,
            [](S s, A a, Reward r, S s_) -> std::pair<Reward,A> {return {r,a};}, 
            [](S s, A a, Reward r)       -> std::pair<Reward,A> {return {r,a};}, 
            0);

    std::cout << "The sequence of actions " << action_sequence 
        << " generated a " << sum_r 
        << " reward accumulation." << std::endl;
}

// The typical use of output iterator is the collection of transitions
// from an episode. Let us define our transition type...
struct Transition {
    S      s;
    A      a;
    Reward r;
    S      s_; // read s_ as s'
    bool   is_terminal;
};
// ... and its display function.
std::ostream& operator<<(std::ostream& os, const Transition& t) {
    os << std::setw(3) << t.s  << ' ' << t.a
        << " ---" << std::setw(5) << t.r << " ---> ";
    if(t.is_terminal)
        os << "End-of-Episode";
    else
        os << std::setw(3) << t.s_;
    return os;
}

template<typename RANDOM_GENERATOR>
void run_episode_version_04(RANDOM_GENERATOR& gen) {
    Simulator simulator;
    auto policy = rl::policy::random(actions.begin(),actions.end(), gen);

    std::cout << std::endl
        << "Version 04" << std::endl
        << "----------" << std::endl
        << std::endl;

    simulator.verbose = true;
    simulator.setPhase("BONBON");

    std::vector<Transition> transitions;
    rl::episode::run(simulator,policy,
            std::back_inserter(transitions),
            [](S s, A a, Reward r, S s_) -> Transition {return {s,a,r,s_,false};}, 
            [](S s, A a, Reward r)       -> Transition {return {s,a,r,s ,true};}, 
            0);

    std::cout << "Here are the transitions that we have collected :" << std::endl
        << std::endl;
    for(auto& t : transitions)
        std::cout << t << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {

    std::random_device rd;
    std::mt19937 gen(rd());

    run_episode_version_01(gen);
    run_episode_version_02(gen);
    run_episode_version_03(gen);
    run_episode_version_04(gen);
    return 0;
}

