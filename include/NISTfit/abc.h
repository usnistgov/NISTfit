#ifndef NISTFIT_ABC_
#define NISTFIT_ABC_

#include <vector>
#include <future>
#include <memory>
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
    };

    /// The abstract base class for the evaluator
    class AbstractEvaluator
    {
    public:
        virtual std::shared_ptr<AbstractOutput> evaluate_one(const std::shared_ptr<AbstractInput> &in) const = 0;
        virtual void set_coefficients(const std::vector<double> &) = 0;
        
        /** Evaluate the residual function in serial operation for the indices in the range [iStart, iStop)
         * @param iStart The starting index (included in the output)
         * @param iStop The stopping index (NOT included in the output) (a la Python)
         */
        std::vector<std::shared_ptr<AbstractOutput> > evaluate_serial(std::vector<std::shared_ptr<AbstractInput> > & q,
                                                                      std::size_t iStart, std::size_t iStop) const {
            std::vector<std::shared_ptr<AbstractOutput> > outs;
            outs.reserve(q.size());
            for (std::size_t i = iStart; i < iStop; ++i) {
                outs.emplace_back(evaluate_one(q[i]));
            }
            return outs;
        };
        
        std::vector<std::shared_ptr<AbstractOutput> > evaluate_parallel(std::vector<std::shared_ptr<AbstractInput> > & q, short Nthreads) const{
            std::size_t Nmax = q.size();
            std::size_t Lchunk = Nmax / Nthreads;
            std::vector<std::future<std::vector<std::shared_ptr<AbstractOutput> > > > futures;
            for (long i = 0; i < Nthreads; ++i) {
                std::size_t iStart = i*Lchunk;
                std::size_t iEnd = (i + 1)*Lchunk;
                futures.push_back(std::async(std::launch::async, &AbstractEvaluator::evaluate_serial, this, std::ref(q), iStart, iEnd));
            }
            // Wait for the threads to terminate
            for (; ; ) // Infinite loop
            {
                bool keep_going = true;
                for (auto &e : futures) {
                    keep_going = keep_going && !e.valid();
                }
                if (!keep_going) { break; }
            }
            // Collect the outputs
            std::vector<std::shared_ptr<AbstractOutput> > outs;
            outs.reserve(Nmax); // Pre-allocate the appropriate size
            for (auto &e : futures) {
                std::vector<std::shared_ptr<AbstractOutput> > thread_results = e.get(); // :( Copy
                outs.insert(outs.end(), thread_results.begin(), thread_results.end());
            }
            return outs;
        };
        /** Construct the Jacobian matrix, where each entry in Jacobian matrix is given by
         * \f[ J_{ij} = \frac{\partial (y_{\rm fit} - y_{\rm given})_i}{\partial c_i} \f]
         * It is constructed by taking the rows of the Jacobian matrix stored in instances of AbstractOutput
         */
        Eigen::MatrixXd get_Jacobian_matrix(const std::vector<std::shared_ptr<AbstractOutput> > &outs) {
            std::size_t ncol = outs[0]->get_Jacobian_row().size();
            Eigen::MatrixXd J(outs.size(), ncol);
            int i = 0;
            for (auto &o : outs) {
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
        Eigen::VectorXd get_error_vector(const std::vector<std::shared_ptr<AbstractOutput> > &outs) {
            Eigen::VectorXd r(outs.size());
            int i = 0;
            for (auto &o : outs) {
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

    /// The data structure for an output for the single y output variable
    class NumericOutput : public AbstractOutput{
    private:
        NumericInput m_in;
        double m_y_calc;
        std::vector<double> Jacobian_row; // Partial derivative of calculated value with respect to each independent variable
    public:
        NumericOutput(NumericInput &in, double y_calc) : m_in(in), m_y_calc(y_calc) {};
        NumericOutput(NumericInput &in, double y_calc, const std::vector<double> &Jacobian_row) : m_in(in), m_y_calc(y_calc), Jacobian_row(Jacobian_row) {};
        NumericOutput(NumericInput &in, double y_calc, std::vector<double> &&Jacobian_row) : m_in(in), m_y_calc(y_calc), Jacobian_row(std::move(Jacobian_row)) {};
        double get_error(){ return m_y_calc - m_in.y(); };
        std::vector<double> & get_Jacobian_row() { return Jacobian_row; }
    };

    /// The class for the evaluation of a single output value for a single input value
    class AbstractNumericEvaluator : public AbstractEvaluator {
    protected:
        std::vector<double> m_c;
    public:
        void set_coefficients(const std::vector<double> &c) { this->m_c = c; };
    };
}; /* namespace NISTfit */

#endif