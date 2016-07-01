#ifndef NISTFIT_ABC_
#define NISTFIT_ABC_

#include <vector>
#include <future>
#include <memory>
#include <iterator>

#include "Eigen/Eigen/Dense"

namespace NISTfit{
    
    /// The abstract base class for the inputs
    class AbstractInput {
    };

    /// The abstract base class for the outputs
    class AbstractOutput{
    public:
        virtual double get_error() = 0;
        virtual std::vector<double> &get_Jacobian_row() { throw std::exception(); };
        /// Evaluate one input and cache the output variables internally
        /// This class should already be holding a pointer to the input to which it is connected
        virtual void evaluate_one() = 0;
        /// Return the linked input state
        virtual std::shared_ptr<AbstractInput> get_input() = 0;

    };

    /// The abstract base class for the evaluator
    class AbstractEvaluator
    {
    protected:
        // Collect the outputs
        std::vector<std::shared_ptr<AbstractOutput> > m_outputs;
        Eigen::MatrixXd J;
        Eigen::VectorXd r;
    public:
        virtual void set_coefficients(const std::vector<double> &) = 0;
        virtual const std::vector<double> & get_const_coefficients() = 0;
        std::size_t get_outputs_size(){ return m_outputs.size(); };
        
        /** Evaluate the residual function in serial operation for the input vector indices in the range [iInputStart, iInputStop)
         * @param iInputStart The starting index (included in the output)
         * @param iInputStop The stopping index (NOT included in the output) (a la Python)
         */
        void evaluate_serial(std::size_t iInputStart, std::size_t iInputStop, std::size_t iOutputStart) const {
            std::size_t j = iOutputStart;
            for (std::size_t i = iInputStart; i < iInputStop; ++i) {
                m_outputs[j]->evaluate_one();
                j++;
            }
        };
        
        void evaluate_parallel(short Nthreads) const{
            std::size_t Nmax = m_outputs.size();
            std::size_t Lchunk = Nmax / Nthreads;
            std::vector<std::future<void> > futures;
            futures.reserve(Nthreads);
            for (long i = 0; i < Nthreads; ++i) {
                std::size_t iStart = i*Lchunk;
                // The last thread gets the remainder, shorter than the others if Nmax mod Nthreads != 0
                std::size_t iEnd = ((i == Nthreads-1) ? m_outputs.size() : (i + 1)*Lchunk);
                futures.push_back(std::async(std::launch::async, &AbstractEvaluator::evaluate_serial, this,
                                             iStart, iEnd, iStart
                                  ));
            }
            // Wait for all the outputs
            for (auto &e : futures) {
                e.get();
            }
        };
        /** Construct the Jacobian matrix, where each entry in Jacobian matrix is given by
         * \f[ J_{ij} = \frac{\partial (y_{\rm fit} - y_{\rm given})_i}{\partial c_i} \f]
         * It is constructed by taking the rows of the Jacobian matrix stored in instances of AbstractOutput
         */
        const Eigen::MatrixXd &get_Jacobian_matrix() {
            std::size_t ncol = m_outputs[0]->get_Jacobian_row().size();
            J.resize(m_outputs.size(), ncol);
            int i = 0;
            for (auto &o : m_outputs) {
                const std::vector<double> &Jrow = o->get_Jacobian_row();
                Eigen::Map<const Eigen::VectorXd> Jrow_wrap(&Jrow[0], Jrow.size());
                J.row(i) = Jrow_wrap;
                i++;
            }
            return J;
        };
        /** /brief Construct the residual vector of residuals for each data point
         * \f[ r_i = (y_{\rm fit} - y_{\rm given})_i\f]
         */
        const Eigen::VectorXd &get_error_vector() {
            r.resize(m_outputs.size());
            int i = 0;
            for (auto &o : m_outputs) {
                r(i) = o->get_error(); i++;
            }
            return r;
        }
    };

    /// The data inputs
    class NumericInput : public AbstractInput{
        public:
            double m_x, m_y;
            NumericInput(double x, double y) : m_x(x), m_y(y) {};
            double x() const { return m_x; };
            double y() const { return m_y; };
    };
    
    /// The class for the evaluation of a single output value for a single input value
    class AbstractNumericEvaluator : public AbstractEvaluator {
        
        
    };

    /// The data structure for an output for the single y output variable
    class NumericOutput : public AbstractOutput{
        protected:
            const std::shared_ptr<NumericInput> &m_in;
            double m_y_calc;
            std::vector<double> Jacobian_row; // Partial derivative of calculated value with respect to each independent variable
            std::vector<double> m_c; // Coefficients that are being fit
            std::shared_ptr<AbstractNumericEvaluator> m_evaluator; // The evaluator connected with this output
        public:
            NumericOutput(const std::shared_ptr<NumericInput> &in) : m_in(in) {};
            double get_error(){ return m_y_calc - m_in->y(); };
            std::vector<double> & get_Jacobian_row() { return Jacobian_row; }
            void resize(std::size_t N){ Jacobian_row.resize(N); };
            std::shared_ptr<AbstractInput> get_input(){ return m_in; };
    };

    
}; /* namespace NISTfit */

#endif