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


#include <vector>

// This file is a piece of code included by our examples. It gathers
// common definition of types.

typedef Simulator::reward_type         Reward; 
typedef Simulator::observation_type    S;
typedef Simulator::action_type         A;

struct Transition {
  S      s;
  A      a;
  Reward r;
  S      s_; // read s_ as s'
  A      a_; // read a_ as a'
  bool   is_terminal;
};

typedef std::vector<Transition>        TransitionSet;


// Here are general purpose setting and reading functions.

Transition   make_transition(S s, A a, Reward r, S s_)    {return {s,a,r,s_,a /* unused */,false};}
Transition   make_terminal_transition(S s, A a, Reward r) {return {s,a,r,s /* unused */,a /* unused */,true};}
S            next_state_of(const Transition& t)           {return t.s_;}
Reward       reward_of(const Transition& t)               {return t.r;}
bool         is_terminal(const Transition& t)             {return t.is_terminal;}
void         set_next_action(Transition& t,A a)           {t.a_ = a;}    


rl::sa::Pair<S,A> current_of(const Transition& t)                         {return {t.s,t.a};}
rl::sa::Pair<S,A> next_of(const Transition& t)                            {return {t.s_,t.a_};}
Transition        make_transition_sa(const rl::sa::Pair<S,A>& z, 
				     Reward r, 
				     const rl::sa::Pair<S,A>& z_)         {return {z.s,z.a,r,z_.s,z_.a,false};}
Transition        make_terminal_transition_sa(const rl::sa::Pair<S,A>& z, 
					 Reward r)                        {return {z.s,z.a,r,z.s /* unused */,z.a /* unused */,true};}
