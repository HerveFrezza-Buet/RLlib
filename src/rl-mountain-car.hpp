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

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <iterator>
#include <utility>
#include <rlAlgo.hpp>
#include <rlEpisode.hpp>
#include <rlException.hpp>

namespace rl {
    namespace problem {
        namespace mountain_car {

            // The action space
            enum class Action: int {
                actionNone = 0,
                actionBackward = 1,
                actionForward = 2};

            constexpr int actionSize = 3;

            // some exceptions for state and action consistancy
            class BadAction : public rl::exception::Any {
                public:
                    BadAction(std::string comment) 
                        : Any(std::string("Bad action performed : ")+comment) {} 
            };

            class BadState : public rl::exception::Any {
                public:
                    BadState(std::string comment) 
                        : Any(std::string("Bad state found : ")+comment) {} 
            };


            // Mountain car parameters
            class DefaultParam {
                public:
                    inline static double minPosition(void)        {return -1.200;}
                    inline static double maxPosition(void)        {return  0.500;}
                    inline static double minSpeed(void)           {return -0.070;}
                    inline static double maxSpeed(void)           {return  0.070;}
                    inline static double goalPosition(void)       {return  maxPosition();}
                    inline static double goalSpeed(void)          {return  0.000;}
                    inline static double goalSpeedMargin(void)    {return  maxSpeed();}
                    inline static double rewardGoal(void)         {return  1.0;};
                    inline static double rewardStep(void)         {return  0.0;};
            };

            // This is the phase space
            template<typename PARAM>
                class Phase {
                    public:
                        using param_type = PARAM;

                        double position,speed;

                        Phase(void) {}
                        Phase(const Phase& copy) : position(copy.position), speed(copy.speed) {}
                        Phase(double p, double s) : position(p), speed(s) {}
                        ~Phase(void) {}
                        Phase& operator=(const Phase& copy) {
                            if(this != &copy) {
                                position = copy.position;
                                speed    = copy.speed;
                            }
                            return *this;
                        }

                        void check(void) const {
                            if( (position > param_type::maxPosition()) || (position < param_type::minPosition())
                                    || (speed > param_type::maxSpeed()) || (speed < param_type::minSpeed()) ) {
                                std::ostringstream ostr;
                                ostr << "mountain_car::Phase::check : At position = " << position << " and speed = " << speed << ".";
                                throw BadState(ostr.str());
                            }
                        }

                        template<typename RANDOM_DEVICE>
                            static Phase<PARAM> random(RANDOM_DEVICE& gen) {
                                return Phase<PARAM>(std::uniform_real_distribution<>(param_type::minPosition(), param_type::maxPosition())(gen),
                                        std::uniform_real_distribution<>(param_type::minSpeed(), param_type::maxSpeed())(gen));
                            }

                        void saturateSpeed(void) {
                            if(speed < param_type::minSpeed())
                                speed = param_type::minSpeed();
                            else if(speed > param_type::maxSpeed())
                                speed = param_type::maxSpeed();
                        }
                };

            /**
             * Mountain car simulator  
             * @author <a href="mailto:Herve.Frezza-Buet@supelec.fr">Herve.Frezza-Buet@supelec.fr</a>
             */
            template<typename MOUNTAIN_CAR_PARAM>
                class Simulator {

                    public:

                        using param_type = MOUNTAIN_CAR_PARAM;

                        using       phase_type = Phase<param_type>;
                        using observation_type = phase_type;
                        using      action_type = Action;
                        using      reward_type = double;

                    private:

                        phase_type current_state;
                        double r;

                    public:

                        // This can be usefull for drawing graphics.
                        void location(double& position, 
                                double& speed,
                                double& height) {
                            position = current_state.position;
                            speed    = current_state.speed;
                            height   = heightOf(position);
                        }

                        static double heightOf(double position) {
                            return sin(3*position);
                        }

                        // The bottom position
                        static double bottom(void) {
                            return - M_PI/6;
                        }

                        void setPhase(const phase_type& s) {
                            current_state = s;
                            current_state.check();
                        }

                        const observation_type& sense(void) const {
                            current_state.check();
                            return current_state;
                        }

                        void timeStep(const action_type& a) {
                            double aa;

                            switch(a) {
                                case Action::actionForward:
                                    aa = 1;
                                    break;
                                case Action::actionBackward: 
                                    aa = -1;
                                    break;
                                case Action::actionNone:
                                    aa = 0;
                                    break;
                                default:
                                    std::ostringstream ostr;
                                    ostr << "mountain_car::Simulator::timeStep(" << static_cast<int>(a) << ")";
                                    throw BadAction(ostr.str());
                            }

                            current_state.speed += (0.001*aa - 0.0025*cos(3*current_state.position));
                            current_state.saturateSpeed();
                            current_state.position += current_state.speed;

                            r=param_type::rewardStep();
                            if(current_state.position < param_type::minPosition()) {
                                current_state.position = param_type::minPosition();
                                current_state.speed    = 0;
                            }
                            else if(current_state.position > param_type::maxPosition()) {

                                if((current_state.speed >= param_type::goalSpeed()) 
                                        && 
                                        (current_state.speed <= param_type::goalSpeed() + param_type::goalSpeedMargin())) {
                                    r = param_type::rewardGoal();
                                    throw rl::exception::Terminal("Goal reached");
                                }

                                throw rl::exception::Terminal("Upper position bound violated");
                            }

                        }

                        reward_type reward(void) const {
                            return r;
                        }

                        Simulator(void) : current_state(), r(0) {}
                        Simulator(const Simulator& copy) 
                            : current_state(copy.current_state),
                            r(copy.r) {}
                        ~Simulator(void) {}

                        Simulator& operator=(const Simulator& copy) {
                            if(this != &copy) 
                                current_state = copy.current_state;
                            return *this;
                        }
                };

            /**
             * @short This plots nice graphics for representing the Q function.
             * @param rank use a negative rank to avoid ranks in file names.
             */
            template<typename SIMULATOR>
                class Gnuplot {
                    private:


                        template<typename Q, typename POLICY>
                            static void Qdata(std::ostream& file, 
                                    const Q& q,
                                    const POLICY& policy,
                                    int points_per_side,
                                    bool draw_q) {

                                double coef_p,coef_s;
                                double position,speed;
                                int p,s;
                                Action a;
                                typename SIMULATOR::phase_type current;
                                coef_p = (SIMULATOR::param_type::maxPosition()-SIMULATOR::param_type::minPosition())/((double)(points_per_side-1));
                                coef_s = (SIMULATOR::param_type::maxSpeed()-SIMULATOR::param_type::minSpeed())/((double)(points_per_side-1));
                                for(s=0;s<points_per_side;++s) {
                                    speed = SIMULATOR::param_type::minSpeed() + coef_s*s;
                                    for(p=0;p<points_per_side;++p) {
                                        position = SIMULATOR::param_type::minPosition() + coef_p*p;
                                        current = typename SIMULATOR::phase_type(position,speed);
                                        a = policy(current);
                                        if(draw_q)
                                            file << position << ' ' << speed << ' ' << q(current,a) << ' ' << static_cast<int>(a) << std::endl;
                                        else
                                            file << position << ' ' << speed << ' ' << static_cast<int>(a) << std::endl;
                                    }
                                    file << std::endl;
                                }
                            }

                    public:

                        template<typename Q, typename POLICY>
                            static void drawQ(std::string title,
                                    std::string file_prefix, int rank,
                                    const Q& q,
                                    const POLICY& policy,
                                    int points_per_side=50) {
                                std::ostringstream ostr;
                                std::ofstream file;
                                std::string numbered_prefix;
                                std::string filename;

                                ostr << file_prefix;
                                if(rank >=0) 
                                    ostr << '-' << std::setfill('0') << std::setw(6) << rank;
                                numbered_prefix = ostr.str();
                                filename = numbered_prefix + ".plot";

                                file.open(filename.c_str());
                                if(!file) {
                                    std::cerr << "Cannot open \"" << filename << "\". Plotting skipped." << std::endl;
                                    return;
                                }
                                file << "unset hidden3d;" << std::endl
                                    << "set xrange [" << SIMULATOR::param_type::minPosition()
                                    << ":" << SIMULATOR::param_type::maxPosition() << "];" << std::endl
                                    << "set yrange [" << SIMULATOR::param_type::minSpeed()
                                    << ":" << SIMULATOR::param_type::maxSpeed() << "];" << std::endl
                                    << "set zrange [-1:1.5];" << std::endl
                                    << "set cbrange [0:2];" << std::endl
                                    << "set view 48,336;" << std::endl
                                    << "set palette defined ( 0 \"yellow\", 1 \"red\",2 \"blue\");" << std::endl
                                    << "set ticslevel 0;" << std::endl
                                    << "set title \"" << title << "\";" << std::endl
                                    << "set xlabel \"position\";" << std::endl
                                    << "set ylabel \"speed\";" << std::endl
                                    << "set zlabel \"Q(max_a)\";" << std::endl
                                    << "set cblabel \"none=" <<  static_cast<int>(Action::actionNone)
                                    << ", forward=" << static_cast<int>(Action::actionForward)
                                    << ", backward=" << static_cast<int>(Action::actionBackward)
                                    << "\";" << std::endl
                                    << "set style line 100 linecolor rgb \"black\";" << std::endl
                                    << "set pm3d at s hidden3d 100;" << std::endl
                                    << "set output \"" << numbered_prefix << ".png\";" << std::endl
                                    << "set term png enhanced size 600,400;"<< std::endl
                                    << "splot '-' using 1:2:3:4 with pm3d notitle;" << std::endl;



                                Qdata(file,q,policy,points_per_side,true);
                                file.close();
                                std::cout << "\"" << filename << "\" generated." << std::endl;
                            }

                        /**
                         * @param rank use a negative rank to avoid ranks in file names.
                         */
                        template<typename Q,typename POLICY>
                            static void drawEpisode(std::string title,
                                    std::string file_prefix, int rank,
                                    SIMULATOR& simulator,
                                    const Q& q,
                                    const POLICY& policy,
                                    unsigned int max_episode_length,
                                    int points_per_side=50) {
                                std::ostringstream ostr;
                                std::ostringstream titleostr;
                                std::ofstream file;
                                std::string numbered_prefix;
                                std::string filename,policyfilename;
                                double cumrew;



                                ostr << file_prefix;
                                if(rank >=0) 
                                    ostr << '-' << std::setfill('0') << std::setw(6) << rank;
                                numbered_prefix = ostr.str();
                                filename = numbered_prefix + ".plot";
                                policyfilename = numbered_prefix + "-policy.data";

                                file.open(filename.c_str());
                                if(!file) {
                                    std::cerr << "Cannot open \"" << filename << "\". Plotting skipped." << std::endl;
                                    return;
                                }

                                std::vector<std::pair<typename SIMULATOR::phase_type,typename SIMULATOR::reward_type>> transitions;
                                rl::episode::run(simulator,policy,
                                        std::back_inserter(transitions),
                                        [](const typename SIMULATOR::phase_type& s, 
                                            const typename SIMULATOR::action_type& a,
                                            const typename SIMULATOR::reward_type r,
                                            const typename SIMULATOR::phase_type& s_) 
                                        ->  std::pair<typename SIMULATOR::phase_type,typename SIMULATOR::reward_type> {return std::make_pair(s,r);},
                                        [](const typename SIMULATOR::phase_type& s, 
                                            const typename SIMULATOR::action_type& a,
                                            const typename SIMULATOR::reward_type r) 
                                        ->  std::pair<typename SIMULATOR::phase_type,typename SIMULATOR::reward_type> {return std::make_pair(s,r);},
                                        max_episode_length);

                                cumrew=0;
                                for(auto& t : transitions)
                                    cumrew += t.second;

                                titleostr << title << "\\n cumulated reward = " << cumrew;

                                file << "set xrange [" << SIMULATOR::param_type::minPosition()
                                    << ":" << SIMULATOR::param_type::maxPosition() << "];" << std::endl
                                    << "set yrange [" << SIMULATOR::param_type::minSpeed()
                                    << ":" << SIMULATOR::param_type::maxSpeed() << "];" << std::endl
                                    << "set zrange [0:3];" << std::endl
                                    << "set cbrange [0:3];" << std::endl
                                    << "set title \"" << titleostr.str() << "\";" << std::endl
                                    << "set palette defined ( 0 \"yellow\", 1 \"red\",2 \"blue\", 3 \"black\");" << std::endl
                                    << "set xlabel \"position\";" << std::endl
                                    << "set ylabel \"speed\";" << std::endl
                                    << "set cblabel \"none=" <<  static_cast<int>(Action::actionNone)
                                    << ", forward=" << static_cast<int>(Action::actionForward)
                                    << ", backward=" << static_cast<int>(Action::actionBackward)
                                    << "\";" << std::endl
                                    << "set view map;" << std::endl
                                    << "set pm3d at s;" << std::endl
                                    << "splot '" << policyfilename << "' with pm3d notitle, \\" << std::endl
                                    << "  '-' with linespoints notitle pt 7 ps 0.5 lc rgb \"black\"" << std::endl;

                                for(auto& t : transitions)
                                    file << t.first.position << ' ' 
                                        << t.first.speed    << ' '
                                        << 3 << std::endl;
                                file.close();
                                std::cout << "\"" << filename << "\" generated." << std::endl;


                                file.open(policyfilename.c_str());
                                if(!file) {
                                    std::cerr << "Cannot open \"" << filename << "\". Plotting skipped." << std::endl;
                                    return;
                                }
                                Qdata(file,q,policy,points_per_side,false);
                                file.close();
                                std::cout << "\"" << policyfilename << "\" generated." << std::endl;
                            }
                };
        }
    }
}
