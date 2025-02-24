// Author : Romain E. Lacoste
// License : BSD-3-Clause

%module prox

%{
#define SWIG_FILE_WITH_INIT
#include "sparkle/prox/prox_zero.h"
#include "sparkle/prox/prox_l1.h"
#include "sparkle/prox/prox_l2.h"
#include "sparkle/prox/prox_elastic_net.h"
%}

%include sparkle/array/array_module.i

%include "sparkle/prox/prox_zero.h"   
%include "sparkle/prox/prox_l1.h"      
%include "sparkle/prox/prox_l2.h" 
%include "sparkle/prox/prox_elastic_net.h" 