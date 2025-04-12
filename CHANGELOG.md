# Changelog

## [1.1.0] - 2025-04-12
### Added
- Implementation of a new learning rate scheduler: 
	two-way backtracking line search.
- Added in the C++ model classes: pre-calculation now leverage 
	the Markov property of the exponential kernel.
### Fixed
- The `LearnerHawkesExp` is now compatible with scikit-learn routines 
	for variable selection, enabling an easy-to-use pipeline for 
	selecting the decay hyperparameter when it is unknown.
- Providing `end_time` with data is now optional. 

## [1.0.0] - 2025-02-24
### Added
- Initial release of Sparklen with core features:

  * A efficient cluster-based simulation method for generating events.

  * A highly versatile and flexible framework for performing inference of 
    multivariate Hawkes process.

  * Novel approaches to address the challenge of multiclass 
    classification within the supervised learning framework.