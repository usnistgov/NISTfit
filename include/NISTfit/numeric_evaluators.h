#ifndef NISTFIT_NUMERIC_EVALUATORS_
#define NISTFIT_NUMERIC_EVALUATORS_

#include "NISTfit/abc.h"

namespace NISTfit{
    
    /// The class for the evaluation of a single output value for a single input value
    class PolynomialOutput : public NumericOutput {
    protected:
        std::size_t m_order; // The order of the polynomial (2: quadratic, 3: cubic, etc...)
        AbstractNumericEvaluator *m_evaluator; // The evaluator connected with this output
    public:
        PolynomialOutput(std::size_t order,
                         const std::shared_ptr<NumericInput> &in,
                         AbstractNumericEvaluator *peval
                         ) : NumericOutput(in), m_order(order), m_evaluator(peval) {};
        void evaluate_one(){
            // Get the input
            double lhs = 0;
            // Do the calculation
            const std::vector<double> &c = m_evaluator->get_const_coefficients();
            for (std::size_t i = 0; i < m_order+1; ++i) {
                double term = pow(m_in->m_x, static_cast<int>(i));
                lhs += c[i]*term;
                Jacobian_row[i] = term;
            }
            m_y_calc = lhs;
        };
    };
    
    /// The class for the evaluation of a single output value for a single input value
    class PolynomialEvaluator : public AbstractNumericEvaluator {
        std::vector<double> m_c;
    public:
        void set_coefficients(const std::vector<double> &c){ m_c = c; };
        const std::vector<double> & get_const_coefficients(){ return m_c; };
        PolynomialEvaluator(std::size_t order, const std::vector<std::shared_ptr<NumericInput> > &inputs)
        {
            for (auto &in: inputs){
                std::shared_ptr<NumericOutput> out(new PolynomialOutput(order, in, this));
                out->resize(order+1); // Set the size of the Jacobian row
                m_outputs.push_back(out);
            }
        };
    };
    
    
}; /* namespace NISTfit */

#endif