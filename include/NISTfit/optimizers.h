#ifndef NISTFIT_OPTIMIZERS_
#define NISTFIT_OPTIMIZERS_

#include <vector>

namespace NISTfit{
    
    /**
     * /brief The Levenberg-Marquadt sum-of-squares minimizer
     * /param E The derived instance of AbstractEvaluator used to evaluate the terms in the sum-of-squares
     * /param inputs The vector of inputs derived from AbstractInput
     * /param c0 The initial values of the coefficients that are to be fit
     */
    std::vector<double> LevenbergMarquadt(std::shared_ptr<AbstractEvaluator> &E,
                           std::vector<std::shared_ptr<AbstractInput> > &inputs,
                           std::vector<double> &c0,
                           bool threading = false);

} /* NISTfit */

#endif