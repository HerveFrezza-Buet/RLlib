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
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <rlAlgo.hpp>
#include <rlException.hpp>
#include <gsl/gsl_vector.h>


namespace rl {
  namespace problem {
    namespace cliff_walking {

      // The action space
      enum class Action {actionNorth=0,
			   actionSouth=1,
			   actionEast=2,
			   actionWest=3};
      constexpr int actionSize = 4;

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
      

      // The phase space

      // This defines external world. Here, we have the following descrete
      // world. '#' is a cliff. When the agent goes there, is restarts at
      // state S and is given a -CLIFF_REWARD reward. Otherwise, each
      // transition gives a -STEP_REWARD reward, except from the goal G that
      // leads to a terminal state for all actions. 

      /*
       *   <----LENGTH---->
       *   ................ line number WIDTH-1
       *   ................
       *   ................
       *   ................ line number 0
       *   S##############G 
       */

      // The phase space is thus an integer, that we will define by an
      // enum. Ths class Cliff handles the mapping to actual grid
      // positions shown on the previous figure into an integer.
      

      /**
       * Cliff-Walking  parameter handler.
       * @author <a href="mailto:Herve.Frezza-Buet@supelec.fr">Herve.Frezza-Buet@supelec.fr</a>
       */
      template<int LENGTH,int WIDTH> 
      class Cliff {
      public:
	enum {
	  size  = LENGTH*WIDTH+2, //!< The number of positions
	  start = 0,              //!< The starting position
	  goal  = size-1          //!< The goal position
	};

	enum {
	  length = LENGTH, 
	  width  = WIDTH
	};

	enum {
	  drawingWidth  = LENGTH,
	  drawingHeight = WIDTH+1
	};

	typedef int phase_type;

      private:


      public:

	/**
	 * @return the position (w,h) in the grid world corresponding to phase p.
	 */
	static void drawingPosition(int& w, int& h, phase_type p) {
	  switch(p) {
	  case start:
	    w = 0;
	    h = WIDTH;
	    break;
	  case goal:
	    w = LENGTH-1;
	    h = WIDTH;
	    break;
	  default:
	    if(p<0 || p >= LENGTH*WIDTH+2) {
	      std::ostringstream os;
	      os << "Cliff<" << LENGTH << ',' << WIDTH << ">::drawingPosition(w,h," 
		 << p << ") : Out of bounds.";
	      throw BadState(os.str());
	    }

	    p--;
	    w = p%LENGTH;
	    h = WIDTH-1-p/LENGTH;
	    break;
	  }
	}
	
	/**
	 * @param min,max is used to scale the value color : min is 0, max is 255.
	 */
	template<typename V>
	static void draw(std::string file_prefix,
			 int file_rank,
			 const V& v,
			 double min, double max) {
	  std::ofstream file;
	  std::ostringstream filename;
	  unsigned char img[drawingWidth*drawingHeight*3];
	  phase_type p;
	  int size,dim_size;
	  int w,h;
	  unsigned char value;
	  unsigned char* rgb;
	  unsigned char* rgb_end;

	  dim_size = drawingWidth*drawingHeight;
	  size = dim_size*3;
	  
	  filename << file_prefix << '-' 
		   << std::setfill('0') << std::setw(6) << file_rank
		   << ".ppm";
	  file.open(filename.str().c_str());
	  if(!file) {
	    std::cerr << "Cannot open \"" << filename.str() << "\". Aborting"
		      << std::endl;
	    return;
	  }
	  rgb = img;
	  rgb_end = rgb + size;
	  while(rgb != rgb_end) {
	    *(rgb++) = 0; 
	    *(rgb++) = 0; 
	    *(rgb++) = 255;
	  }

	  for(p=start;
	      p<=goal;
	      ++p) {
	    drawingPosition(w,h,p);
	    value = (unsigned char)(255*(v(p)-min)/(max-min)+.5);
	    rgb = img + (drawingWidth*h+w)*3;
	    
	    *(rgb++) = value; 
	    *(rgb++) = value; 
	    *(rgb++) = value;
	  }
	  
	  file << "P6" << std::endl
	       << drawingWidth << ' ' << drawingHeight << std::endl
	       << "255" << std::endl;
	  file.write((char*)img,size);
	  file.close();
	}

	template<typename V,typename VISITED>
	static void draw_visited(std::string file_prefix,
				 int file_rank,
				 const V& v,
				 const VISITED& visited,
				 double min, double max) {
	  std::ofstream file;
	  std::ostringstream filename;
	  unsigned char img[drawingWidth*drawingHeight*3];
	  phase_type p;
	  int size,dim_size;
	  int w,h;
	  unsigned char value;
	  unsigned char* rgb;
	  unsigned char* rgb_end;

	  dim_size = drawingWidth*drawingHeight;
	  size = dim_size*3;
	  
	  filename << file_prefix << '-' 
		   << std::setfill('0') << std::setw(6) << file_rank
		   << ".ppm";
	  file.open(filename.str().c_str());
	  if(!file) {
	    std::cerr << "Cannot open \"" << filename.str() << "\". Aborting"
		      << std::endl;
	    return;
	  }
	  rgb = img;
	  rgb_end = rgb + size;
	  while(rgb != rgb_end) {
	    *(rgb++) = 0; 
	    *(rgb++) = 0; 
	    *(rgb++) = 255;
	  }

	  for(p=start;
	      p<=goal;
	      ++p) {
	    drawingPosition(w,h,p);
	    value = (unsigned char)(255*(v(p)-min)/(max-min)+.5);
	    rgb = img + (drawingWidth*h+w)*3;
	    
	    if(visited(p)) {
	      *rgb = value; ++rgb;
	      *rgb = value; ++rgb;
	      *rgb = 0;
	    }
	    else {
	      *rgb = value; ++rgb;
	      *rgb = value; ++rgb;
	      *rgb = value;
	    }
	      
	  }
	  
	  file << "P6" << std::endl
	       << drawingWidth << ' ' << drawingHeight << std::endl
	       << "255" << std::endl;
	  file.write((char*)img,size);
	  file.close();
	}
      };

      
      /**
       * Default parameters for cliff walking simulators.
       * @author <a href="mailto:Herve.Frezza-Buet@supelec.fr">Herve.Frezza-Buet@supelec.fr</a>
       */
      class Param {
      public:
	double goalReward() const {return    0;}
	double stepReward() const {return   -1;}
	double fallReward() const {return -100;}
	double bumpReward() const {return  stepReward();}
      };
      
      /**
       * Cliff walking simulator  
       * @author <a href="mailto:Herve.Frezza-Buet@supelec.fr">Herve.Frezza-Buet@supelec.fr</a>
       */
      template<typename CLIFF, typename CLIFF_PARAM>
      class Simulator {

      private:
	
	const CLIFF_PARAM& param;
	
      public:


	typedef typename CLIFF::phase_type  phase_type;
	typedef phase_type                  observation_type;
	typedef Action                      action_type;
	typedef double                      reward_type;

      private:

	phase_type current_state;
	double r;

      public:

	void restart(void) {
	  setPhase(CLIFF::start);
	}

	void setPhase(const phase_type& s) {
	  current_state = s;
	  if(s < CLIFF::start || s > CLIFF::goal) {
	    std::ostringstream ostr;

	    ostr << "Simulator::setPhase(" << s << ")";
	    throw BadState(ostr.str());
	  }
	}

	const observation_type& sense(void) const {
	  return current_state;
	}

	void timeStep(const action_type& a) {
	  
	  switch(current_state) {
	  case CLIFF::start:
	    stepStart(a);
	    break;
	  case CLIFF::goal:
	    stepGoal(a);
	    break;
	  default:
	    step(a);
	    break;
	  }
	}

      private:

	void stepStart(const action_type a) {
	  switch(a) {
	  case Action::actionNorth:
	    current_state = 1;
	    r = param.stepReward();
	    break;
	  case Action::actionSouth: 
	  case Action::actionWest:
	    r = param.bumpReward();
	    break;
	  case Action::actionEast:
	    r = param.fallReward();
	    break;
	  default:
	    std::ostringstream ostr;
	    ostr << "cliff_walking::Simulator::stepStart(" << static_cast<int>(a) << ")";
	    throw BadAction(ostr.str());
	  }
	}

	void stepGoal(const action_type a) {
	  r = param.goalReward();
	  throw rl::exception::Terminal("Transition from goal");
	}

	void step(const action_type a) {
	  phase_type s = current_state-1; // Easier index

	  switch(a) {
	  case Action::actionNorth:
	    if(s / CLIFF::length < CLIFF::width-1) { // not upper wall
	      s += CLIFF::length;
	      r = param.stepReward();
	    }
	    else
	      r = param.bumpReward();
	    break;
	  case Action::actionSouth: 
	    if(s / CLIFF::length > 0) { // not on the edge
	      s -= CLIFF::length;
	      r = param.stepReward();
	    }
	    else if(s == 0) {
	      s = CLIFF::start-1; // On start again
	      r = param.bumpReward();
	    }
	    else if(s == CLIFF::length-1) {
	      s = CLIFF::goal - 1; // remember that we will do s++...
	      r = param.stepReward();
	    }
	    else {
	      s = CLIFF::start-1;
	      r = param.fallReward();
	    }
	    break;
	  case Action::actionEast:
	    if(s % CLIFF::length < CLIFF::length-1) { // Not on right wall
	      r = param.stepReward();
	      s++;
	    }
	    else
	      r = param.bumpReward();
	    break;
	  case Action::actionWest:
	    if(s % CLIFF::length != 0) { // Not on left wall
	      r = param.stepReward();
	      s--;
	    }
	    else
	      r = param.bumpReward();
	    break;
	  default:
	    std::ostringstream ostr;
	    ostr << "cliff_walking::Simulator::timeStep(" << static_cast<int>(a) << ")";
	    throw BadAction(ostr.str());
	  }
	  
	  current_state = s+1;
	}
	
      public:

	reward_type reward(void) const {
	  return r;
	}

	Simulator(const CLIFF_PARAM& p) : param(p), current_state(CLIFF::start) {}
	Simulator(const Simulator& copy) : param(copy.param), current_state(copy.current_state) {}
	~Simulator(void) {}

	Simulator& operator=(const Simulator& copy) {
	  if(this != &copy) 
	    current_state = copy.current_state;
	  return *this;
	}
      };
      
    }
  }
}
