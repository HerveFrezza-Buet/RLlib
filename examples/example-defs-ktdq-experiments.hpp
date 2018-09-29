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


template<typename CRITIC, typename fctQ, typename ACTION_ITERATOR, typename RANDOM_GENERATOR>
void make_experiment(CRITIC& critic, const fctQ& q,
        const ACTION_ITERATOR& a_begin,
        const ACTION_ITERATOR& a_end,
        RANDOM_GENERATOR& gen) {
    int               episode,step;
    std::ofstream     ofile;
    std::ifstream     ifile;

    Simulator         simulator(gen);
    CRITIC            critic_loaded = critic;

    auto              explore_agent = rl::policy::random(a_begin,a_end, gen);
    auto              greedy_agent  = rl::policy::greedy(q,a_begin,a_end);

    try {
        step = 0;

        for(episode = 0; episode < NB_OF_EPISODES; ++episode) {
            simulator.setPhase(Simulator::phase_type()); 
            rl::episode::learn(simulator,explore_agent,critic,MAX_EPISODE_LENGTH);
            if((episode % TEST_PERIOD)==0) {
                ++step;
                test_iteration(greedy_agent,step, gen);
            }
        }

        // Now, we can save the ktdq object.
        std::cout << "Writing ktdq.data" << std::endl;
        ofile.open("ktdq.data");
        if(!ofile)
            std::cerr << "cannot open file for writing" << std::endl;
        else {
            ofile << critic;
            ofile.close();
        }

        // You can load back with >>
        std::cout << "Reading ktdq.data" << std::endl;
        ifile.open("ktdq.data");
        if(!ifile)
            std::cerr << "cannot open file for reading" << std::endl;
        else {
            ifile >> critic_loaded;
            ifile.close();
        }

        // As the theta parameter is shared by q and the critic, the load
        // of the critic modifies q, and thus the greedy agent.

        // let us try this loaded ktdq
        test_iteration(greedy_agent,step, gen);           
    }
    catch(rl::exception::Any& e) {
        std::cerr << "Exception caught : " << e.what() << std::endl;
    }
}
