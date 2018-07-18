/*   This file is part of rl-lib
 *
 *   Copyright (C) 2010,  Supelec
 *
 *   Author : Herve Frezza-Buet and Matthieu Geist
 *
 *   Contributor : Jeremy Fix
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

#include <type_traits>
#include <gsl/gsl_vector.h>

namespace rl {

  namespace traits {

    template <typename...>
    using void_t = void;

    template <typename F, typename S, typename=void>
    struct is_state_value_function : std::false_type {};

    template <typename F, typename S>
    struct is_state_value_function<F, S, 
				   void_t<decltype(std::declval<F>()(std::declval<const S>()))>> : std::true_type {};


    
    template <typename F, typename S, typename A, typename=void>
    struct is_state_action_value_function : std::false_type {};

    template <typename F, typename S, typename A>
    struct is_state_action_value_function<F, S, A,
					  void_t<decltype(std::declval<F>()(std::declval<const S>(), std::declval<const A>()))>> : std::true_type {};



    template<typename CRITIC, typename S, typename=void>
    struct is_srs_critic: std::false_type {};

    template<typename CRITIC, typename S>
    struct is_srs_critic<CRITIC, S,
			 void_t<decltype(std::declval<CRITIC>().learn(std::declval<const S>(), std::declval<double>(), std::declval<const S>()))>> : std::true_type {};


    template<typename CRITIC, typename S, typename A, typename=void>
    struct is_sars_critic: std::false_type {};

    template<typename CRITIC, typename S, typename A>
    struct is_sars_critic<CRITIC, S, A,
			  void_t<decltype(std::declval<CRITIC>().learn(std::declval<const S>(), std::declval<const A>(), std::declval<double>(), std::declval<const S>()))>> : std::true_type {};

    
    template<typename CRITIC, typename S, typename A, typename=void>
    struct is_sarsa_critic: std::false_type {};

    template<typename CRITIC, typename S, typename A>
    struct is_sarsa_critic<CRITIC, S, A,
			   void_t<decltype(std::declval<CRITIC>().learn(std::declval<const S>(), std::declval<const A>(), std::declval<double>(), std::declval<const S>(), std::declval<const A>()))>> : std::true_type {};    

    
    namespace gsl {
      
      template <typename F, typename S, typename=void>
      struct is_parametrized_state_value_function : std::false_type {};

      template <typename F, typename S>
      struct is_parametrized_state_value_function<F, S, 
						  void_t<decltype(std::declval<F>()(std::declval<const gsl_vector*>(), std::declval<const S>()))>> : std::true_type {};

    
      template <typename F, typename S, typename A, typename=void>
      struct is_parametrized_state_action_value_function : std::false_type {};

      template <typename F, typename S, typename A>
      struct is_parametrized_state_action_value_function<F, S, A,
							 void_t<decltype(std::declval<F>()(std::declval<const gsl_vector*>(), std::declval<const S>(), std::declval<const A>()))>> : std::true_type {};

    }
  }
}
