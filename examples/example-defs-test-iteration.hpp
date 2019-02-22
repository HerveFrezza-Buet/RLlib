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

template<typename POLICY, typename RANDOM_GENERATOR>
void test_iteration(const POLICY& policy, int step, RANDOM_GENERATOR& gen) {
    Simulator       simulator(gen);
    int             episode,length;
    double          mean_length;

    mean_length=0;
    for(episode = 0; episode < NB_LENGTH_SAMPLES; ++episode) {

        // Let us generate an episode and get its length
        Simulator::phase_type start_phase;
        start_phase.random(gen);
        simulator.setPhase(start_phase);
        length = rl::episode::run(simulator,policy,MAX_EPISODE_LENGTH);
        // We display the length
        std::cout << "\rStep " << std::setw(4) << std::setfill('0') << step
            << " : " << std::setfill('.') << std::setw(4) << episode+1 << " length = "
            << std::setw(10) << std::setfill(' ') 
            << length << std::flush;
        // Mean update
        mean_length += length;
    }

    mean_length /= NB_LENGTH_SAMPLES;
    std::cout << "\rStep " 
        << std::setw(4) << std::setfill('0') << step
        << " : mean length = "
        << std::setw(10) << std::setfill(' ') 
        << .01*(int)(mean_length*100+.5) << std::endl;
}
