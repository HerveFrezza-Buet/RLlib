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

#include <sstream>
#include <type_traits>
#include <functional>
#include <gsl/gsl_vector.h>

#include <rlException.hpp>
#include <rlTD.hpp>

namespace rl {

    namespace gsl {
        /**
         * @short SARSA algorithm is just TD learning of Q
         */
        template<typename STATE,
            typename ACTION,
            typename fctQ_PARAMETRIZED,
            typename fctGRAD_Q_PARAMETRIZED>
                auto sarsa(gsl_vector* param,
                        double gamma_coef,
                        double alpha_coef,
                        const fctQ_PARAMETRIZED& fct_q,
                        const fctGRAD_Q_PARAMETRIZED& fct_grad_q) 
                -> TD<STATE,ACTION> {
                    return TD<STATE,ACTION>
                        (param,
                         gamma_coef,alpha_coef,
                         fct_q,fct_grad_q);
                }

    }
}

