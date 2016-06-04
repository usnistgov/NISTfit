#ifndef NISTFIT_NUMERIC_EVALUATORS_
#define NISTFIT_NUMERIC_EVALUATORS_

#include "NISTfit/abc.h"

namespace NISTfit{
    
    /// The class for the evaluation of a single output value for a single input value
    class PolynomialEvaluator : public AbstractNumericEvaluator {
    protected:
        std::size_t m_order; // The order of the polynomial (2: quadratic, 3: cubic, etc...)
    public:
        PolynomialEvaluator(std::size_t order) : m_order(order) {};
        std::shared_ptr<AbstractOutput> evaluate_one(const std::shared_ptr<AbstractInput> &pIn) const{
            // Cast to the derived type
            NumericInput *in = static_cast<NumericInput*>(pIn.get());
            // The row in the Jacobian for L-M
            std::vector<double> Jacobian_row(m_order+1);
            // Get the input
            double lhs = 0;
            // Do the calculation
            for (std::size_t i = 0; i < m_order+1; ++i) {
                double term = pow(in->x(), static_cast<int>(i));
                lhs += m_c[i] * term;
                Jacobian_row[i] = term;
            }
            // Return the output
            return std::shared_ptr<AbstractOutput>(new NumericOutput(*in, lhs, Jacobian_row));
        };
    };

}; /* namespace NISTfit */

#endif