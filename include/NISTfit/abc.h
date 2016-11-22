#ifndef NISTFIT_ABC_
#define NISTFIT_ABC_

#include <vector>
#include <future>
#include <memory>
#include <iterator>
#include <iostream>
#include <numeric>      // std::accumulate
#include <queue>

#include "Eigen/Dense"

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
        /// A pure-virtual function that is used to handle ANY exception that is caught in the
        /// evaluate_one function.  You might want to consider re-throwing the exception in the function
        /// and then setting an error flag/message, etc.
        virtual void exception_handler() = 0;

    };

    // Forward definitions
    struct ThreadData;
    class AbstractEvaluator;

    // Convenience type definition
    using job = std::packaged_task<void(ThreadData *)>;

    // Some data associated to each thread.
    struct ThreadData
    {
        int id; // Could use thread::id, but this is filled before the thread is started
        std::thread t; // The thread object
        std::queue<job> jobs; // The job queue
        std::condition_variable cv; // The condition variable to wait for threads
        std::mutex m; // Mutex used for avoiding data races
        AbstractEvaluator *eval; // Evaluator
        std::size_t iStart, iEnd; 
        bool stop = false; // When set, this flag tells the thread that it should exit
    };

    /// The abstract base class for the evaluator
    class AbstractEvaluator
    {
    protected:
        // Collect the outputs
        std::vector<std::shared_ptr<AbstractOutput> > m_outputs;
        Eigen::MatrixXd J;
        Eigen::VectorXd r;
        std::vector<ThreadData> thread_data; ///< The thread data for the threads
        std::vector<std::future<void>> futures;
        bool quit; 
        std::mutex quit_mutex;
    public:
        virtual void set_coefficients(const std::vector<double> &) = 0;
        virtual const std::vector<double> & get_const_coefficients() = 0;
        /// Get the size of the outputs
        std::size_t get_outputs_size() { return m_outputs.size(); };

        ~AbstractEvaluator() {
            if (!thread_data.empty()) {
                // auto startTime = std::chrono::system_clock::now();
                kill_threads();
                // auto endTime = std::chrono::system_clock::now();
                // double thread_kill_elap = std::chrono::duration<double>(endTime - startTime).count();
                //std::cout << "thread teardown:" << thread_kill_elap << " s\n";
            }
        }

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

        /**
         * @brief In an infinite loop, either run the evaluation, or wait
         */
        void evaluate_threaded(ThreadData *pData) {

            std::unique_lock<std::mutex> l(pData->m, std::defer_lock);
            while (true)
            {
                l.lock();

                // Wait until the queue isn't empty or stop is signaled
                pData->cv.wait(l, [pData]() {
                    return (pData->stop || !pData->jobs.empty());
                });

                // Stop was signaled, let's exit the thread
                if (pData->stop) { return; }

                // Pop one task from the queue...
                job j = std::move(pData->jobs.front());
                pData->jobs.pop();

                l.unlock();

                // Execute the task!
                j(pData);
            }
        };
        /**
         * @brief Setup the threads that will be used to do the evaluations
         * 
         */
        void setup_threads(short Nthreads) {
            if (thread_data.empty()) {
                // auto startTime = std::chrono::system_clock::now();

                std::size_t Nmax = m_outputs.size();
                std::size_t Lchunk = Nmax / Nthreads;

                // Have to reconstruct because we cannot resize vector<ThreadData> because of non-copyconstructable members
                thread_data = std::vector<ThreadData>(Nthreads);

                for (long i = 0; i < Nthreads; ++i) {
                    auto &td = thread_data[i];
                    td.iStart = i*Lchunk;
                    // The last thread gets the remainder, shorter than the others if Nmax mod Nthreads != 0
                    td.iEnd = ((i == Nthreads - 1) ? m_outputs.size() : (i + 1)*Lchunk);
                    td.eval = this;
                    // Construct the thread that will actually do the evaluation
                    td.t = std::thread(&AbstractEvaluator::evaluate_threaded, this, &td);
                }
                // auto endTime = std::chrono::system_clock::now();
                // double thread_setup_elap = std::chrono::duration<double>(endTime - startTime).count();
                //std::cout << "thread setup:" << thread_setup_elap << " s\n";
            }
        };
        /**
         * @brief Kill the threads that have been spun up to do the evaluations
         * 
         * This function is called when AbstractEvaluator is destroyed
         */
        void kill_threads() {

            // Tell all threads to stop
            for (auto& td : thread_data)
            {
                std::unique_lock<std::mutex> l(td.m);
                td.stop = true;
                td.cv.notify_one();
                
            }
            // Join all the threads
            for (auto& td : thread_data) { td.t.join(); }
        };
        void evaluate_parallel(short Nthreads){
            
            // Set up threads but put them in holding pattern
            //  no-op if threads are already initialized
            setup_threads(Nthreads);

            // Now fire!
            for (auto &td : thread_data){

                // The payload that will execute when the job is run
                job j([](ThreadData *pData)
                    {
                        for (std::size_t j = pData->iStart; j < pData->iEnd; ++j) {
                            try{
                                pData->eval->m_outputs[j]->evaluate_one();
                            }
                            catch(...){
                                pData->eval->m_outputs[j]->exception_handler();
                            }
                        }
                    }
                );
                futures.push_back(j.get_future());

                // Add the job to the queue
                std::unique_lock<std::mutex> l(td.m);
                td.jobs.push(std::move(j));

                // Notify the thread that there is work do to...
                td.cv.notify_one();
            }
            
            // Wait for all the tasks to be completed...
            for (auto& f : futures) { f.wait(); }
            futures.clear();
        };

        /** 
         * @brief Construct the Jacobian matrix \f$\mathbf{J}\f$
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

        /** @brief Construct the residual vector of residuals for each data point
         * \f[ r_i = (y_{\rm fit} - y_{\rm given})_i\f]
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
            const std::shared_ptr<NumericInput> m_in;
            double m_y_calc;
            std::vector<double> Jacobian_row; // Partial derivative of calculated value with respect to each independent variable
            std::shared_ptr<AbstractNumericEvaluator> m_evaluator; // The evaluator connected with this output
        public:
            /// Copy constructor
			NumericOutput(const std::shared_ptr<NumericInput> &in) : m_in(in) {};
			/// Move constructor
			NumericOutput(const std::shared_ptr<NumericInput> &&in) : m_in(in) {};
            virtual double get_error(){ return m_y_calc - m_in->y(); };
            std::vector<double> & get_Jacobian_row() { return Jacobian_row; }
            void resize(std::size_t N){ Jacobian_row.resize(N); };
            std::shared_ptr<AbstractInput> get_input(){ return m_in; };
    };
    
}; /* namespace NISTfit */

#endif
