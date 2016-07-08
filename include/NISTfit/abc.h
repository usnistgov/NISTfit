#ifndef NISTFIT_ABC_
#define NISTFIT_ABC_

#include <vector>
#include <future>
#include <memory>
#include <iterator>
#include <iostream>
#include <numeric>      // std::accumulate
#include <queue>

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
        std::vector<std::thread> threads; ///< The threads used to evaluate the threads
        std::deque<std::atomic<bool> > thread_run; ///< A flag for thread status
        bool quit; 
        std::mutex quit_mutex;
    public:
        virtual void set_coefficients(const std::vector<double> &) = 0;
        virtual const std::vector<double> & get_const_coefficients() = 0;
        std::size_t get_outputs_size() { return m_outputs.size(); };

        ~AbstractEvaluator() {
            if (!threads.empty()) {
                auto startTime = std::chrono::system_clock::now();
                kill_threads();
                auto endTime = std::chrono::system_clock::now();
                double thread_kill_elap = std::chrono::duration<double>(endTime - startTime).count();
                //std::cout << "thread teardown:" << thread_kill_elap << " s\n";
            }
        }

        /** Evaluate the residual function in serial operation for the input vector indices in the range [iInputStart, iInputStop)
         * @param iInputStart The starting index (included in the output)
         * @param iInputStop The stopping index (NOT included in the output) (a la Python)
         */
        void evaluate_serial(std::size_t iInputStart, std::size_t iInputStop, std::size_t iOutputStart) const {
            int Nrepeat = 1;
            for (int rep = 0; rep < Nrepeat; ++rep) {
                std::size_t j = iOutputStart;
                for (std::size_t i = iInputStart; i < iInputStop; ++i) {
                    m_outputs[j]->evaluate_one();
                    j++;
                }
            }
        };

        // In an infinite loop, either run the evaluation, or don't do anything.
        void evaluate_threaded(std::size_t iInputStart, std::size_t iInputStop, std::size_t iOutputStart, std::atomic<bool> &run) {
            while(!quit){
                if (run.load()){
                    // Do the evaluation - setting values are thread-safe so long as ranges set do not overlap
                    evaluate_serial(iInputStart, iInputStop, iInputStart);

                    // Set the completion flag - I'm done for now
                    run.store(false, std::memory_order::memory_order_relaxed);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        };

        void setup_threads(short Nthreads) {
            std::size_t Nmax = m_outputs.size();
            std::size_t Lchunk = Nmax / Nthreads;

            // Set up the atomics for communicating with the threads
            thread_run.resize(Nthreads);
            for (std::size_t i = 0; i < Nthreads; ++i) {
                thread_run[i] = false;
            }
            
            // Set the quit flag to false (no mutex needed since only main thread active at this point)
            quit = false;

            for (long i = 0; i < Nthreads; ++i) {
                std::size_t iStart = i*Lchunk;
                // The last thread gets the remainder, shorter than the others if Nmax mod Nthreads != 0
                std::size_t iEnd = ((i == Nthreads - 1) ? m_outputs.size() : (i + 1)*Lchunk);

                threads.push_back(
                    std::thread(&AbstractEvaluator::evaluate_threaded, this, iStart, iEnd, iStart, std::ref(thread_run[i]))
                );
                
            }
        }
        void kill_threads() {
            
            // Signal the threads to stop
            quit_mutex.lock();
            quit = true;
            quit_mutex.unlock();
            
            // Join the threads (wait for them to terminate)
            for (auto &e : threads) {  e.join(); }

            threads.clear();
        }
        
        void evaluate_parallel(short Nthreads){
            
            if (threads.empty()) {
                auto startTime = std::chrono::system_clock::now();
                    // Set up threads but put them in holding pattern
                    setup_threads(Nthreads);
                auto endTime = std::chrono::system_clock::now();
                double thread_setup_elap = std::chrono::duration<double>(endTime - startTime).count();
                //std::cout << "thread setup:" << thread_setup_elap << " s\n";
            }

            // Set the thread completion flag to fire the threads
            // and allow them to evaluate in parallel
            for (std::size_t i = 0; i < thread_run.size(); ++i) {
                thread_run[i].store(true, std::memory_order::memory_order_relaxed);
            }
            
            // Wait for all the threads
            bool done;
            do{ 
                done = true;
                for (const auto &running : thread_run) {
                    if (running) { done = false; break; }
                }
            } while (!done);
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
    protected:
        std::vector<double> m_c;
    public:
        void set_coefficients(const std::vector<double> &c){ m_c = c; };
        const std::vector<double> & get_const_coefficients(){ return m_c; };
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