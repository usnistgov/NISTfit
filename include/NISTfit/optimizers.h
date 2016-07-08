#ifndef NISTFIT_OPTIMIZERS_
#define NISTFIT_OPTIMIZERS_

#include <vector>

namespace NISTfit{
    
    /**
     * /brief The Levenberg-Marquadt sum-of-squares minimizer
     * @param E The derived instance of AbstractEvaluator used to evaluate the terms in the sum-of-squares
     * @param c0 The initial values of the coefficients that are to be fit
     * @param threading True to use multi-threading, serial otherwise
     * @param Nthreads -1 for std::thread::hardware_concurrency(), positive number otherwise
     */
    std::vector<double> LevenbergMarquadt(std::shared_ptr<AbstractEvaluator> &E,
                           std::vector<double> &c0,
                           bool threading = false,
                           short Nthreads = -1);

} /* NISTfit */

#endif