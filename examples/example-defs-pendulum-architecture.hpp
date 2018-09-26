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

#include <array>
#include <cmath>

// This file is a piece of code included by our examples. It gathers
// common definition of types.


// This feature justs transform (s,a) into a vector 
// (angle,speed,action_is_None,action_is_Left,action_is_Right).
#define PHI_DIRECT_DIMENSION 5
void phi_direct(gsl_vector *phi, const S& s, const A& a) {
    gsl_vector_set_zero(phi);
    gsl_vector_set(phi,0,s.angle);
    gsl_vector_set(phi,1,s.speed);
    switch(a) {
    case rl::problem::inverted_pendulum::Action::actionNone:
      gsl_vector_set(phi,2,1.0);
      break;
    case rl::problem::inverted_pendulum::Action::actionLeft:
      gsl_vector_set(phi,3,1.0);
      break;
    case rl::problem::inverted_pendulum::Action::actionRight:
      gsl_vector_set(phi,4,1.0);
      break;
    default:
      throw rl::problem::inverted_pendulum::BadAction(" in phi_direct()");
    }
}

#define PHI_RBF_DIMENSION 30
void phi_rbf(gsl_vector *phi, const S& s, const A& a) {
  std::array<double,3> angle = { {-M_PI_4,0,M_PI_4} };
  std::array<double,3> speed = { {-1,0,1} };

  int action_offset;
  int i,j,k;
  double dangle,dspeed;
  
  if(phi == (gsl_vector*)0)
    throw rl::exception::NullVectorPtr("in Feature::operator()");
  else if((int)(phi->size) != PHI_RBF_DIMENSION)
    throw rl::exception::BadVectorSize(phi->size,PHI_RBF_DIMENSION,"in Feature::operator()");
  
  switch(a) {
  case rl::problem::inverted_pendulum::Action::actionNone:
    action_offset=0;
    break;
  case rl::problem::inverted_pendulum::Action::actionLeft:
    action_offset=10;
    break;
  case rl::problem::inverted_pendulum::Action::actionRight:
    action_offset=20;
    break;
  default:
    throw rl::problem::inverted_pendulum::BadAction("in phi_gaussian()");
  }

  gsl_vector_set_zero(phi);
  for(i=0,k=action_offset+1;i<3;++i) {
    dangle  = s.angle - angle[i];
    dangle *= dangle;
    for(j=0;j<3;++j,++k) {
      dspeed  = s.speed - speed[j];
      dspeed *= dspeed;
      gsl_vector_set(phi,k,exp(-.5*(dangle+dspeed))); 
    }
    gsl_vector_set(phi,action_offset,1);
  }
}


