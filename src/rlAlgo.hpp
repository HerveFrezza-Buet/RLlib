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

#include <utility>
#include <cstdlib>
#include <vector>
#include <type_traits>
#include <iterator>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <map>

#include <gsl/gsl_vector.h>

namespace rl {

  template<typename ITERATOR,
	   typename fctEVAL>
  auto min(const fctEVAL& f,
	   const ITERATOR& begin, 
	   const ITERATOR& end)
    -> decltype(f(*begin))
  { 
    ITERATOR iter = begin;
    auto m     = f(*iter);
    for(++iter;iter!=end;++iter) {
      auto  v = f(*iter);
      if(v < m)
	m = v;
    }
    return m;
  }

  template<typename ITERATOR,
	   typename fctEVAL>
  auto max(const fctEVAL& f,
	   const ITERATOR& begin, 
	   const ITERATOR& end)
    -> decltype(f(*begin))
  { 
    ITERATOR iter = begin;
    auto m     = f(*iter);
    for(++iter;iter!=end;++iter) {
      auto  v = f(*iter);
      if(v > m)
	m = v;
    }
    return m;
  }

  template<typename ITERATOR,
	   typename fctEVAL>
  auto range(const fctEVAL& f,
	   const ITERATOR& begin, 
	   const ITERATOR& end)
    -> std::pair<decltype(f(*begin)),
		 decltype(f(*begin))>
  { 
    ITERATOR iter = begin;
    auto min   = f(*iter);
    auto max   = min;
    for(++iter;iter!=end;++iter) {
      auto  v = f(*iter);
      if(v > max)
	max = v;
      else if(v < min)
	min = v;
    }
    return {min,max};
  }

  template<typename ITERATOR,
	   typename fctEVAL>
  auto argmax(const fctEVAL& f,
	      const ITERATOR& begin, 
	      const ITERATOR& end)
    -> std::pair<decltype(*begin),
		 decltype(f(*begin))> 
{ 
    ITERATOR iter = begin;
    auto arg_max = *iter;
    auto max     = f(*iter);
    for(++iter;iter!=end;++iter) {
      auto a = *iter;
      auto v = f(a);
      if(v > max) {
	max = v;
	arg_max = a;
      }
    }
    return {arg_max,max};
  }

  template<typename ITERATOR,
	   typename fctEVAL>
  auto argmin(const fctEVAL& f,
	      const ITERATOR& begin, 
	      const ITERATOR& end)
    -> std::pair<decltype(*begin),
		 decltype(f(*begin))> 
{ 
    ITERATOR iter = begin;
    auto arg_min = *iter;
    auto min     = f(*iter);
    for(++iter;iter!=end;++iter) {
      auto  a = *iter;
      auto  v = f(a);
      if(v < min) {
	min = v;
	arg_min = a;
      }
    }
    return {arg_min,min};
  }
  
    /**
     * Builds an iterator of value type T. It requires the type T to be castable in int type,
     * the associated int values being contiguous.
     */


  template<typename A, bool>
  struct _enumeration_of {
    using value_type = A;
  };

  template<typename A>
  struct _enumeration_of<A, true> {
      using value_type = typename std::underlying_type<A>::type;
  };

  template<typename T>
  using enumeration_of = _enumeration_of<T, std::is_enum<T>::value>;


  template<typename T>
  class enumerator {
  private:

      using underlying_type = typename enumeration_of<T>::value_type;

      underlying_type j;

      static T bad_cast(underlying_type u) {
          T* ptr = reinterpret_cast<T*>(&u);
          return *ptr;
      }

  public:

    using difference_type   = int;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;
    using iterator_category = std::random_access_iterator_tag;

    enumerator() : j(0) {}
    enumerator(const enumerator& cp) : j(cp.j) {}
    enumerator(value_type i) : j(static_cast<underlying_type>(i)) {}
    enumerator& operator=(value_type i) {j=static_cast<underlying_type>(i); return *this;}
    enumerator& operator=(const enumerator& cp) {j=cp.j; return *this;}
    enumerator& operator++() {++j; return *this;}
    enumerator& operator--() {--j; return *this;}
    enumerator& operator+=(difference_type diff) {j+=diff; return *this;}
    enumerator& operator-=(difference_type diff) {j-=diff; return *this;}
    enumerator operator++(difference_type) {enumerator res = *this; ++*this; return res;}
    enumerator operator--(difference_type) {enumerator res = *this; --*this; return res;}
    difference_type operator-(const enumerator& i) const {return j - i.j;}
    enumerator operator+(underlying_type i) const {
        auto cpy =  *this;
        cpy.j += i;
        return cpy;
    }
    enumerator operator-(difference_type i) const {
        return (*this)+(-i);
    }
 
    T operator*() const {return bad_cast(j);}
    bool operator==(const enumerator& i) const {return j == i.j;}
    bool operator!=(const enumerator& i) const {return j != i.j;}
  };

  namespace random {

    
    /**
     * @return A random value according to the histogram represented
     * by f(x), for x in [begin,end[.
     */
    template<typename ITERATOR,
        typename fctEVAL,
        typename RANDOM_DEVICE>
            auto density(const fctEVAL& f,
                    const ITERATOR& begin, const ITERATOR& end,
                    RANDOM_DEVICE& rd) 
            -> decltype(*begin) {
                auto size = end-begin;
                std::vector<double> fvalues(size);
                auto iter = begin;
                auto fvaluesiter = fvalues.begin();
                for(; iter != end; ++iter, ++fvaluesiter)
                    *fvaluesiter = f(*iter);
                std::discrete_distribution<decltype(end - begin)> d(fvalues.begin(), fvalues.end());
                return *(begin + d(rd));

            }

    /**
     * @return true with a probability proba
     */

    template<typename ITERATOR,
	     typename fctEVAL,
         typename RANDOM_DEVICE>
    auto softmax(const fctEVAL& f,
		 double temperature,
		 const ITERATOR& begin, const ITERATOR& end, 
         RANDOM_DEVICE& rd) 
      -> decltype(*begin) {

          std::map<const decltype(*begin), double> f_values;
          double fmax = std::numeric_limits<double>::lowest();
          for(auto it = begin; it != end; ++it) {
              f_values[*it] = f(*it);
              fmax = std::max(fmax, f_values[*it]);
          }

          auto shifted_exp_values = [&temperature, &f_values, &fmax](const decltype(*begin)& a) -> double {
            return exp((f_values[a] - fmax)/temperature);
          };

      return rl::random::density(shifted_exp_values, begin,end, rd);
    }
  }


  namespace sa {
    
    template <typename S, typename A>
    struct Pair {
      S s;
      A a;
    };
    
    template <typename S, typename A>
    Pair<S,A> pair(const S& s, const A& a) {return {s,a};}
    

    namespace gsl {
      // This rewrites q(theta,s,a) as v(theta,(s,a)).
      template<typename S, typename A, typename REWARD, typename Q>
      auto vparam_of_qparam(const Q& q) -> std::function<REWARD (const gsl_vector*,const Pair<S,A>&)> {
	return [&q](const gsl_vector* theta, const Pair<S,A>& sa) -> REWARD {return q(theta,sa.s,sa.a);};
      }
      // This rewrites grad_q(theta,grad,s,a) as v(theta,grad,(s,a)).
      template<typename S, typename A, typename REWARD, typename Q>
      auto gradvparam_of_gradqparam(const Q& gq) -> std::function<void (const gsl_vector*,gsl_vector*,Pair<S,A>)> {
	return [&gq](const gsl_vector* theta, gsl_vector* grad, const Pair<S,A>& sa) -> void {gq(theta,grad,sa.s,sa.a);};
      }
    }
  }
}
