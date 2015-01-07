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

#include <functional>

namespace rl {    
  /**
   * This performs a policy iteration.<br>
   */
  template<typename BATCH_CRITIC, 
	   typename TRANSITION_ITERATOR,
	   typename ACTION_ITERATOR,
	   typename Q,
	   typename fctIS_TERMINAL,
	   typename fctGET_NEXT_STATE,
	   typename fctSET_NEXT_ACTION>
  void batch_policy_iteration_step(BATCH_CRITIC& critic,
				   const Q& q,
				   const TRANSITION_ITERATOR& begin,
				   const TRANSITION_ITERATOR& end,
				   const ACTION_ITERATOR& a_begin,
				   const ACTION_ITERATOR& a_end,
				   const fctIS_TERMINAL& is_terminal,
				   const fctGET_NEXT_STATE& get_next_state,
				   const fctSET_NEXT_ACTION& set_next_action) {
    critic(begin,end);
    for(TRANSITION_ITERATOR iter = begin;iter != end; ++iter) {
      auto& t = *iter;
      if(!is_terminal(t))
	set_next_action(t,
			rl::argmax(std::bind(q,get_next_state(t),std::placeholders::_1),
				   a_begin,a_end).first);
    }
  }
}
