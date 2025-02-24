// Author : Romain E. Lacoste
// License : BSD-3-Clause

#ifndef MODEL_HAWKES_EXP_LEAST_SQUARES_SINGLE_H_
#define MODEL_HAWKES_EXP_LEAST_SQUARES_SINGLE_H_

#include "sparkle/array/array.h"
#include "sparkle/array/array2D.h"
#include "sparkle/array/sharedarray.h"
#include "sparkle/array/sharedarray2D.h"

class ModelHawkesExpLeastSquaresSingle{

	private:

	size_t n_components;
	ArrayInt1D N;
	ArrayDouble1D H, H2;
	ArrayDouble2D D, C;
	bool weights_computed;

	void compute_weights(const ListSharedArrayDouble1D &jump_times, const double end_time, const double decay);

	double compute_loss_i(const size_t i, const ListSharedArrayDouble1D &jump_times, const double end_time, const double decay, const SharedArrayDouble2D &theta);

	void compute_grad_i(const size_t i, const ListSharedArrayDouble1D &jump_times, const double end_time, const double decay, const SharedArrayDouble2D &theta, SharedArrayDouble2D &grad);

	public:

	ModelHawkesExpLeastSquaresSingle();

	ModelHawkesExpLeastSquaresSingle(size_t n);

	size_t get_n_components();

	double compute_loss(const ListSharedArrayDouble1D &jump_times, const double end_time, const double decay, const SharedArrayDouble2D &theta);

	SharedArrayDouble2D compute_grad(const ListSharedArrayDouble1D &jump_times, const double end_time, const double decay, const SharedArrayDouble2D &theta);

	SharedArrayDouble2D compute_hessian(const ListSharedArrayDouble1D &jump_times, const double end_time, const double decay);

	friend class ModelHawkesExpLeastSquares;

};


#endif /* MODEL_HAWKES_EXP_LEAST_SQUARES_SINGLE_H_ */
