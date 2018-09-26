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

// See example-000-000-overview.cc

#define S_CARDINALITY         Cliff::size
#define A_CARDINALITY         rl::problem::cliff_walking::actionSize
#define TABULAR_Q_CARDINALITY S_CARDINALITY*A_CARDINALITY 
#define TABULAR_Q_RANK(s,a)   (static_cast<int>(a)*S_CARDINALITY+s)   

double q_parametrized(const gsl_vector* theta,
		      S s, A a) { 
  return gsl_vector_get(theta,TABULAR_Q_RANK(s,a));
}

void grad_q_parametrized(const gsl_vector* theta,   
			 gsl_vector* grad_theta_sa,
			 S s, A a) {
  gsl_vector_set_basis(grad_theta_sa,TABULAR_Q_RANK(s,a));
}
