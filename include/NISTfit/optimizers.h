#ifndef NISTFIT_OPTIMIZERS_
#define NISTFIT_OPTIMIZERS_

#include <vector>

namespace NISTfit{

    /**
     * @brief A struct for holding options to be passed to LevenbergMarquardt function
     */
    struct LevenbergMarquardtOptions {

        std::vector<double> c0; ///< The initial coefficients that are being fitted
        bool threading = false; ///< True to use threaded evaluation, false for serial evaluation
        short Nthreads = -1; ///< Number of threads to use; -1 for std::thread::hardware_concurrency(), positive number otherwise
        double omega; ///< The relaxation to be applied to the step, in the range (0,infinity). A value of 1.0 is a full step from Levenberg-Marquardt.  Making omega < 1 yields partial steps and makes the algorithm slower but more stable.

        LevenbergMarquardtOptions() : omega(1.0) {};
    };
    
    /**
     * @brief The Levenberg-Marquardt sum-of-squares minimizer
     * @param E The derived instance of AbstractEvaluator used to evaluate the terms in the sum-of-squares
     * @param options The options to be passed to this function
     */
    std::vector<double> LevenbergMarquardt(std::shared_ptr<AbstractEvaluator> &E, LevenbergMarquardtOptions &options);

} /* NISTfit */

#endif