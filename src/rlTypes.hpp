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

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

std::ostream& operator<<(std::ostream& os, const gsl_vector* v);
/**
 * @param v Is a pointer that will be freed and reallocated by the call. <b>avoid uninitialized v !</b> (use 0 at least)
 */
std::istream& operator>>(std::istream& is, gsl_vector*& v);
std::ostream& operator<<(std::ostream& os, const gsl_matrix* m);
/**
 * @param m Is a pointer that will be freed and reallocated by the call. <b>avoid uninitialized m !</b> (use 0 at least)
 */
std::istream& operator>>(std::istream& is, gsl_matrix*& m);




// GSL serialization

inline std::ostream& operator<<(std::ostream& os, const gsl_vector* v) {
  os << "[ " << v->size
     << " :";
  for(unsigned int i=0;i<v->size;++i)
    os << " " << gsl_vector_get(v,i);
  os << "]";
    
  return os;
}

inline std::istream& operator>>(std::istream& is, gsl_vector*& v) {
  char c;
  double value;
  unsigned int i,size;
  is >> c >> size >> c;
  gsl_vector_free(v);
  v = gsl_vector_alloc(size);
  for(i=0;i<size;++i) {
    is >> value;
    gsl_vector_set(v,i,value);
  }
  is >> c ;
  return is;
}

inline std::ostream& operator<<(std::ostream& os, const gsl_matrix* m) {
  unsigned int i,j;
  os << "[ " << m->size1 << 'x' <<  m->size2
     << " :";
  for(i=0;i<m->size1;++i)
    for(j=0;j<m->size2;++j)
      os << " " << gsl_matrix_get(m,i,j);
  os << "]";
  return os;
}

inline std::istream& operator>>(std::istream& is, gsl_matrix*& m) {
  char c;
  double value;
  unsigned int i,j,size1,size2;
  is >> c >> size1 >> c >> size2 >> c;
  gsl_matrix_free(m);
  m = gsl_matrix_alloc(size1,size2);
  for(i=0;i<size1;++i) 
    for(j=0;j<size2;++j) {
      is >> value;
      gsl_matrix_set(m,i,j,value);
  }
  is >> c ;
  return is;
}



