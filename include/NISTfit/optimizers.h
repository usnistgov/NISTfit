#ifndef NISTFIT_OPTIMIZERS_
#define NISTFIT_OPTIMIZERS_

#include <vector>

namespace NISTfit{

    /**
     * 
     */
    struct LevenbergMarquadtOptions {
        std::vector<double> c0; ///< The initial coefficients that are being fitted
        bool threading = false; ///< True to use threaded evaluation, false for serial evaluation
        short Nthreads = -1; ///< Number of threads to use; -1 for std::thread::hardware_concurrency(), positive number otherwise
    };
    
    /**
     * /brief The Levenberg-Marquadt sum-of-squares minimizer
     * @param E The derived instance of AbstractEvaluator used to evaluate the terms in the sum-of-squares
     * @param options The options to be passed to this function
     */
    std::vector<double> LevenbergMarquadt(std::shared_ptr<AbstractEvaluator> &E, LevenbergMarquadtOptions &options);

} /* NISTfit */

#endif