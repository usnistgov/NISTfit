#ifndef NISTFIT_EXAMPLES_
#define NISTFIT_EXAMPLES_

#include "NISTfit/abc.h"

namespace NISTfit{

/*
 * \brief The factorial function
 * 
 * Not done efficiently, could certainly be improved, but the point is we want it to be slow!
 */
double factorial(int N){
    if (N == 0){ return 1; } // An identity
    double o = N; // output; as double to avoid overflow
    for (int i = N - 1; i > 0; --i) {
        o *= i;
    }
    return o;
}
/* 
 * \brief The exponential function exp(x) expressed as series expansion
 *
 * This is an intentionally slow implementation of the exponential function, in 
 * order to increase the amount of work per evaluation
 */
double exp_expansion(double x, int N) {
    if (N <= 0){ return exp(x); }
    double y = 0;
    for (int m=0; m < N; ++m){
        y += pow(x, m)/factorial(m);
    }
    return y;
}
double sin_expansion(double x, int N) {
    if (N <= 0){ return sin(x); }
    double y = 0;
    for (int m = 0; m < N; ++m) {
        y += pow(-1, m)*pow(x, 2*m+1)/factorial(2*m+1);
    }
    return y;
}
double cos_expansion(double x, int N) {
    if (N <= 0){ return cos(x); }
    double y = 0;
    for (int m = 0; m < N; ++m) {
        y += pow(-1, m)*pow(x, 2*m)/factorial(2*m);
    }
    return y;
}

class DecayingExponentialOutput : public NumericOutput {
protected:
    int N; ///< Order of Taylor series expansion
public:
    DecayingExponentialOutput(int N,  
                              const std::shared_ptr<NumericInput> &in)
        : NumericOutput(in), N(N) { resize(3); };
    /// In the highly unlikely case of an exception in this class, 
    /// (implementation of this method is required), set the calculated value 
    /// to something very large
    void exception_handler() { m_y_calc = 100000; }
    void evaluate_one() {
        // Get a reference to the coefficients
        const std::vector<double> &c = get_AbstractEvaluator().get_const_coefficients();
        // Do the calculation
        const double x = m_in->x(), e = exp_expansion(-c[0]*x, N), 
                     s1 = sin_expansion(c[1]*x, N), c2 = cos_expansion(c[2]*x, N);
        double y = e*s1*c2;
        Jacobian_row[0] = -x*y;
        Jacobian_row[1] = x*e*cos_expansion(c[1]*x,N)*c2;
        Jacobian_row[2] = -x*e*s1*sin_expansion(c[2]*x,N);
        m_y_calc = y;
    }
};

} /* namespace NISTfit */

#endif