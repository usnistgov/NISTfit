#ifndef NISTFIT_ABC_
#define NISTFIT_ABC_

#include <vector>
#include <future>
#include <memory>
#include <iterator>
#include <iostream>
#include <numeric>      // std::accumulate
#include <queue>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#undef WIN32_LEAN_AND_MEAN
#endif

#include "Eigen/Dense"

#include "ThreadPool.h"

namespace NISTfit{

    // Forward definitions
    struct ThreadData;
    class AbstractEvaluator;
    
    /// The abstract base class for the inputs
    class AbstractInput {
    };

    /// The abstract base class for the outputs
    class AbstractOutput{
    private:
        /// The AbstractEvaluator that this output is linked with.  This pointer *MUST* be set 
        /// when the AbstractOutput (or derived class thereof) is added to the AbstractEvaluator.
        /// The AbstractEvaluator::add_output() function takes case of this automatically.
        AbstractEvaluator *m_evaluator;
    public:
        virtual double get_error() const = 0;
        virtual const std::vector<double> &get_Jacobian_row() const { throw std::exception(); };
        /// Evaluate one input and cache the output variables internally
        /// This class should already be holding a pointer to the input to which it is connected
        virtual void evaluate_one() = 0;
        /// Return the linked input state
        virtual AbstractInput & get_input() const = 0;
        /// A pure-virtual function that is used to handle ANY exception that is caught in the
        /// evaluate_one function.  You might want to consider re-throwing the exception in the function
        /// and then setting an error flag/message, etc.
        virtual void exception_handler() = 0;
        /// Get the pointer to the AbstractEvaluator linked with this output
        virtual const AbstractEvaluator & get_AbstractEvaluator() const { return *m_evaluator; }
        /// Set the pointer to the AbstractEvaluator linked with this output
        virtual void set_AbstractEvaluator(AbstractEvaluator *evaluator) { m_evaluator = evaluator; }
    };

    /// The abstract base class for the evaluator
    class AbstractEvaluator
    {
    private:
        std::vector<std::shared_ptr<AbstractOutput> > m_outputs;
        std::vector<int> m_affinity_scheme; ///< A vector of processor indices that shall be used for each thread spun up, 0-based
        std::unique_ptr<ThreadPool> m_pool; ///< A ThreadPool of threads
        std::vector<double> m_times; ///< Elapsed times for each parallel payload
    protected:
        Eigen::MatrixXd J;
        Eigen::VectorXd r;

        /**
        * @brief Setup the threads that will be used to do the evaluations
        *
        */
        void setup_threads(short Nthreads) {
            
            if (!m_pool || Nthreads != m_pool->get_threads().size()){
                // Make a thread pool for the workers
                m_pool = std::unique_ptr<ThreadPool>(new ThreadPool(Nthreads));
                m_times.resize(Nthreads);
      
                // Set the thread affinity if desired
#if defined(WIN32)
                auto &threads = m_pool->get_threads();
                for (long i = 0; i < Nthreads; ++i) {
                    std::thread &td = threads[i];
                    if (!m_affinity_scheme.empty() && i <= m_affinity_scheme.size()) {
                        // See http://stackoverflow.com/a/41574964/1360263
                        auto affinity_mask = (static_cast<DWORD_PTR>(1) << m_affinity_scheme[i]); //core number starts from 0
                        SetThreadAffinityMask(td.native_handle(), affinity_mask);
                    }
                }
#endif
            }
        };
        /**
        * @brief Kill the threads that have been spun up to do the evaluations
        *
        * This function is called when AbstractEvaluator is destroyed
        */
        void kill_threads() {
            if (m_pool){
                m_pool->JoinAll();
                m_pool.release();
            }
        };
        /// Add a single output to the list of outputs and connect pointer to AbstractEvaluator
        void add_output(const std::shared_ptr<AbstractOutput> &out) {
            m_outputs.push_back(out);
            m_outputs.back()->set_AbstractEvaluator(this);
        }
    public:
        virtual void set_coefficients(const std::vector<double> &) = 0;
        virtual const std::vector<double> & get_const_coefficients() const = 0;
        /// Get the size of the outputs
        std::size_t get_outputs_size() { return m_outputs.size(); };
        /// Add a vector of instances derived from AbstractOutput to this evaluator
        void add_outputs(const std::vector<std::shared_ptr<AbstractOutput> > &outputs) {
            for (auto &out : outputs) {
                add_output(out);
            }
        }
        /// Get a reference to the vector of outputs
        std::vector<std::shared_ptr<AbstractOutput> > & get_outputs() { return m_outputs; };
        /// Destructor
        ~AbstractEvaluator() {
            // auto startTime = std::chrono::system_clock::now();
            kill_threads();
            // auto endTime = std::chrono::system_clock::now();
            // double thread_kill_elap = std::chrono::duration<double>(endTime - startTime).count();
            //std::cout << "thread teardown:" << thread_kill_elap << " s\n";
        }
            std::vector<double> get_times(){ return m_times; }

        /**
         * @brief Evaluate the residual function in serial operation for the input vector indices in the range [iInputStart, iInputStop)
         * @param iInputStart The starting index (included in the output)
         * @param iInputStop The stopping index (NOT included in the output) (a la Python)
         * @param iOutputStart The starting index for where the outputs should be placed (probably equal to iInputStart)
         */
        void evaluate_serial(std::size_t iInputStart, std::size_t iInputStop, std::size_t iOutputStart) const {
            int Nrepeat = 1;
            for (int rep = 0; rep < Nrepeat; ++rep) {
                std::size_t j = iOutputStart;
                for (std::size_t i = iInputStart; i < iInputStop; ++i) {
                    try{
                        m_outputs[j]->evaluate_one();
                    }
                    catch(...){
                        m_outputs[j]->exception_handler();
                    }
                    j++;
                }
            }
        };
        
        // Return a vector of times for each repeat of calling evaluate_parallel
        std::vector<double> time_evaluate_parallel(short Nthreads, short Nrepeats){
            std::vector<double> times;
            for (auto i = 0; i < Nrepeats; ++i){
                auto startTime = std::chrono::high_resolution_clock::now();
                evaluate_parallel(Nthreads);
                auto endTime = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration<double>(endTime - startTime).count());
            }
            return times;
        }
        // Return a vector of times for each repeat of calling evaluate_serial
        std::vector<double> time_evaluate_serial(short Nrepeats){
            std::vector<double> times;
            for (auto i = 0; i < Nrepeats; ++i){
                auto startTime = std::chrono::high_resolution_clock::now();
                evaluate_serial(0,get_outputs_size(),0);
                auto endTime = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration<double>(endTime - startTime).count());
            }
            return times;
        }

        /**
         * @brief Evaluate all the outputs in parallel
         * @param Nthreads The number of threads over which the calculations should be distributed
         */
        void evaluate_parallel(short Nthreads){

            // Set up threads but put them in holding pattern
            // no-op if threads are already initialized
            setup_threads(Nthreads);

            std::size_t Nmax = m_outputs.size();
            std::size_t Lchunk = Nmax / Nthreads;
            
            for (auto i = 0; i < Nthreads; ++i)
            {
                auto itStart = m_outputs.begin() + i*Lchunk;
                // The last thread gets the remainder, shorter than the others if N mod Nthreads != 0
                // iEnd is NON-INCLUSIVE !!!!!!!!!!
                auto itEnd = m_outputs.begin() + ((i == Nthreads - 1) ? Nmax : (i + 1)*Lchunk);
                double &elapsed = m_times[i];
                std::function<void(void)> f = [itStart, itEnd, &elapsed]() {
                    auto startTime = std::chrono::high_resolution_clock::now();
                    for (auto it = itStart; it != itEnd; ++it) {
                        try {
                            (*it)->evaluate_one();
                        }
                        catch (...) {
                            (*it)->exception_handler();
                        }
                    }
                    auto endTime = std::chrono::high_resolution_clock::now();
                    elapsed = std::chrono::duration<double>(endTime - startTime).count();
                };
                m_pool->AddJob(f);
            }
            // Now we wait for all threads to finish
            m_pool->WaitAll();
        };

        /** @brief Construct the Jacobian matrix \f$\mathbf{J}\f$
         *
         * Each entry in the Jacobian matrix is given by
         * \f[ J_{ij} = \frac{\partial r_i}{\partial c_j} \f]
         * where \f$r_i\f$ is the \f$i\f$-th residue and \f$c_j\f$ is the \f$j\f$-th coefficient
         *
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
        Eigen::MatrixXd build_Jacobian_matrix_numerically(double dx) {
            std::size_t ncol = m_outputs[0]->get_Jacobian_row().size();
            Eigen::MatrixXd Jfd(m_outputs.size(), ncol);

            // Initial values
            auto c0 = get_const_coefficients();
            set_coefficients(c0);
            evaluate_serial(0,m_outputs.size(),0);
            Eigen::VectorXd r0 = get_error_vector();

            // Iterate over the columns
            for (int icol = 0; icol < ncol; ++icol) {
                std::vector<double> c = c0;
                double dc = dx*c[icol];
                c[icol] += dc;
                set_coefficients(c);
                evaluate_serial(0, m_outputs.size(), 0);
                r = get_error_vector();
                Jfd.col(icol) = (r-r0)/dc;
            }
            return Jfd;
        };

        /** @brief Construct the residual vector of residuals for each data point
         *
         * \f[ r_i = (y_{\rm model} - y_{\rm given})_i\f]
         *
         * Internally, the AbstractOutput::get_error() function is called on each AbstractOutput managed by this evaluator.  One 
         * of AbstractOutput::evaluate_serial() or AbstractOutput::evaluate_threaded() should have already been called before calling this function
         */
        const Eigen::VectorXd &get_error_vector() {
            r.resize(m_outputs.size());
            int i = 0;
            for (auto &o : m_outputs) {
                r(i) = o->get_error(); i++;
            }
            return r;
        }
        /**
         * @brief Set affinity scheme that is to be used to determine which thread is connected to which processor
         *
         * The indices in the vector are the indices for the first, second, third, etc. thread, 0-based.  For instance if you want
         * the affinity to go as the first thread on core 1, the first core on thread 2, etc., you might have: [0,2,4,6,1,3,5,7]
         */
        void set_affinity_scheme(const std::vector<int> &affinity_scheme){ m_affinity_scheme = affinity_scheme; };
        /**
         * @brief Get the affinity scheme that is in use
         */
        const std::vector<int> & get_affinity_scheme() { return m_affinity_scheme; };
    };

    /// The data inputs
    class NumericInput : public AbstractInput{
        protected:
            double m_x, m_y;
        public:
            NumericInput(double x, double y) : m_x(x), m_y(y) {};
            const double x() const { return m_x; };
            const double y() const { return m_y; };
    };
    
    /// The class for the evaluation of a single output value for a single input value
    class NumericEvaluator : public AbstractEvaluator {
    protected:
        std::vector<double> m_c;
    public:
        void set_coefficients(const std::vector<double> &c){ m_c = c; };
        const std::vector<double> & get_const_coefficients() const { return m_c; };
    };

    /// The data structure for an output for the single y output variable
    class NumericOutput : public AbstractOutput{
        protected:
            const std::shared_ptr<NumericInput> m_in;
            double m_y_calc;
            std::vector<double> Jacobian_row; // Partial derivative of calculated value with respect to each independent variable
        public:
            /// Copy constructor
            NumericOutput(const std::shared_ptr<NumericInput> &in) : m_in(in) {};
            /// Move constructor
            NumericOutput(const std::shared_ptr<NumericInput> &&in) : m_in(in) {};
            virtual double get_error() const override { return m_y_calc - m_in->y(); };
            virtual const std::vector<double> & get_const_coefficients() const { return get_AbstractEvaluator().get_const_coefficients(); }
            const std::vector<double> & get_Jacobian_row() const override  { return Jacobian_row; }
            AbstractInput & get_input() const override { return *static_cast<AbstractInput*>(m_in.get()); };
            void resize(std::size_t N){ Jacobian_row.resize(N); };
    };
    
    /// The data structure for an output for the single y output variable
    class FiniteDiffOutput : public NumericOutput{
        
    protected:
        std::function<double(const std::vector<double> &)> m_f;
        std::vector<double> m_dc;
    public:
        /// Copy constructor w/ passed in model function
        FiniteDiffOutput(const std::shared_ptr<NumericInput> &in,
                         const std::function<double(const std::vector<double> &)> &f,
                         const std::vector<double> &dc)
            : NumericOutput(in), m_f(f), m_dc(dc) {
                resize(dc.size());};
        /// Move constructor w/ passed in model function
        FiniteDiffOutput(const std::shared_ptr<NumericInput> &&in,
                         const std::function<double(const std::vector<double> &)> &f,
                         const std::vector<double> &dc)
            : NumericOutput(in), m_f(f), m_dc(dc) {
                resize(dc.size());};
        virtual double call_func(const std::vector<double> &c){
            return m_f(c);
        }
        /// Evaluate the function, and the Jacobian row by numerical differentiation
        void evaluate_one() override{
            // Do the calculation
            const std::vector<double> &c = get_const_coefficients();
            m_y_calc = call_func(c);
            for (std::size_t i = 0; i < c.size(); ++i) {
                std::vector<double> cp = c, cm = c;
                cp[i] += m_dc[i]; cm[i] -= m_dc[i];
                Jacobian_row[i] = (call_func(cp) - call_func(cm))/(2*m_dc[i]);
            }
        };
        void exception_handler() override{ m_y_calc = 100000; }
    };
    
}; /* namespace NISTfit */

#endif
