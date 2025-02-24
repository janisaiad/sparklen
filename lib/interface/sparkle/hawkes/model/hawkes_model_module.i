// Author : Romain E. Lacoste
// License : BSD-3-Clause

%module hawkes_model

%{
#define SWIG_FILE_WITH_INIT
#include "sparkle/hawkes/model/model_hawkes_exp_least_squares_single.h"
#include "sparkle/hawkes/model/model_hawkes_exp_least_squares.h"
#include "sparkle/hawkes/model/model_hawkes_exp_log_likelihood_single.h"
#include "sparkle/hawkes/model/model_hawkes_exp_log_likelihood.h"
%}

%include sparkle/array/array_module.i

%include "sparkle/hawkes/model/model_hawkes_exp_least_squares_single.h"
%include "sparkle/hawkes/model/model_hawkes_exp_least_squares.h"
%include "sparkle/hawkes/model/model_hawkes_exp_log_likelihood_single.h"
%include "sparkle/hawkes/model/model_hawkes_exp_log_likelihood.h"         

  