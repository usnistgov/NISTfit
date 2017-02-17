//
//  Inspired by http://www.drdobbs.com/cpp/c11s-async-template/240001196
//  See also http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf
//

#include "NISTfit/abc.h"
#include "NISTfit/optimizers.h"
#include "NISTfit/numeric_evaluators.h"
#include "NISTfit/examples.h"

#include <iostream>
#include <chrono>

using namespace NISTfit;

class SaturationPressureOutput : public NumericOutput {
protected:
    const std::vector<double> m_e; // Exponents for terms in saturation pressure equation
public:
    SaturationPressureOutput(const std::vector<double> &e,
                             const std::shared_ptr<NumericInput> &in)
        : NumericOutput(in), m_e(e) { };
    /// In the highly unlikely case of an exception in this class (implementation of this method is required), 
    /// set the calculated value to something very large
    void exception_handler(){ m_y_calc = 100000; }
    void evaluate_one(){
        // Retrieve references to required data structures
        auto &AE = get_AbstractEvaluator();
        const std::vector<double> &c = AE.get_const_coefficients();
        Eigen::MatrixXd::RowXpr &Jacobian_row = AE.get_Jacobian_row(get_index());
        
        // Do the calculation
        double y = 0;
        for (int repeat = 0; repeat < 1; ++repeat){
        for (std::size_t i = 0; i < m_e.size(); ++i) {
            double term = pow(m_in->x(), m_e[i]);
            y += c[i] * term;
            Jacobian_row[i] = term;
        }
        }
        m_y_calc = y;
    }
};

double fit_waterpanc(bool threading, std::size_t N, short Nthreads) {
    
    std::vector<double> theta(N), LHS(N);

    // Generate some artificial data from the ancillary curve of Wagner and Pruss for water
    std::vector<double> c0 = { -7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502},
                        e = { 1, 1.5, 3, 3.5, 4, 7.5 };
    for (std::size_t i = 0; i < N; ++i) {
        theta[i] = (1 - 0.5778678897721512) / (N - 1)*i + 0.5778678897721512;
        LHS[i] = 0;
        // LHS = ln(p/pc)*T/Tc
        for (std::size_t k = 0; k < c0.size(); ++k) {
            LHS[i] += c0[k] * pow(theta[i], e[k]);
        }
    }

    // Create the outputs
    std::vector<std::shared_ptr<AbstractOutput> > outputs;
    for (int i = 0; i < theta.size(); ++i) {
        auto in = std::shared_ptr<NumericInput>(new NumericInput(theta[i], LHS[i]));
        outputs.push_back(std::shared_ptr<AbstractOutput>(new SaturationPressureOutput(e, in)));
    }
    
    // Instantiate the evaluator
    std::shared_ptr<AbstractEvaluator> eval(new NumericEvaluator());
    eval->add_outputs(outputs);
    
    // Run and time
    std::vector<double> c = { 1,1,1,1,1,1 };
    std::vector<std::shared_ptr<AbstractOutput> > outs;
    auto startTime = std::chrono::system_clock::now();
    auto opts = LevenbergMarquardtOptions();
        opts.c0 = c; opts.threading = threading; opts.Nthreads = Nthreads;
        auto cc = LevenbergMarquardt(eval, opts);
    return std::chrono::duration<double>(std::chrono::system_clock::now() - startTime).count();
}

double fit_polynomial(bool threading, std::size_t Nmax, short Nthreads)
{
    std::vector<std::shared_ptr<AbstractOutput> > outputs;
    for (double i = 0; i < Nmax; ++i) {
        double x = i / ((double)Nmax);
        double y = 1 + 2*x + 3*x*x;
        auto in = std::shared_ptr<NumericInput>(new NumericInput(x, y));
        outputs.push_back(std::shared_ptr<AbstractOutput>(new PolynomialOutput(2, in)));
    }
    std::shared_ptr<AbstractEvaluator> eval(new NumericEvaluator());
    eval->add_outputs(outputs);
    
    std::vector<double> c = { -10, -2, 2.5 };
    auto startTime = std::chrono::system_clock::now();
        auto opts = LevenbergMarquardtOptions();
        opts.c0 = c; opts.threading = threading; opts.Nthreads = Nthreads;
        auto cc = LevenbergMarquardt(eval, opts);
    auto endTime = std::chrono::system_clock::now();
    return std::chrono::duration<double>(endTime - startTime).count();
}

double fit_decaying_exponential(bool threading, std::size_t Nmax, short Nthreads, long N)
{
    double a = 0.2, b = 3, c = 1.3;
    std::vector<std::shared_ptr<AbstractOutput> > outputs;
    for (double i = 0; i < Nmax; ++i) {
        double x = i / ((double)Nmax);
        double y = exp(-a*x)*sin(b*x)*cos(c*x);
        auto in = std::shared_ptr<NumericInput>(new NumericInput(x, y));
        outputs.push_back(std::shared_ptr<AbstractOutput>(new DecayingExponentialOutput(N, in)));
    }
    std::shared_ptr<AbstractEvaluator> eval(new NumericEvaluator());
    eval->add_outputs(outputs);

    std::vector<double> c0 = { 1, 1, 1 };
    auto startTime = std::chrono::system_clock::now();
    auto opts = LevenbergMarquardtOptions();
    opts.c0 = c0; opts.threading = threading; opts.Nthreads = Nthreads;
    auto cc = LevenbergMarquardt(eval, opts);
    auto endTime = std::chrono::system_clock::now();
    return std::chrono::duration<double>(endTime - startTime).count();
}

void speedtest_fit_polynomial(short Nthread_max)
{
    std::cout << "XXXXXXXXXX POLYNOMIAL XXXXXXXXXX" << std::endl;
    for (std::size_t N = 10000; N < 10000000; N *= 10) {
        auto time_serial = fit_polynomial(false, N, 1);
        for (short Nthreads = 2; Nthreads <= Nthread_max; ++Nthreads) {
            const bool threading = true;
            auto time_parallel = fit_polynomial(threading, N, Nthreads);
            printf("%10d %10d %10.7f %10.7f(nothread) %10.7f(thread)\n", Nthreads, static_cast<int>(N), time_serial/time_parallel, time_serial, time_parallel);
        }
    }
}

void speedtest_fit_water_ancillary(short Nthread_max)
{
    std::cout << "XXXXXXXXXX WATER ANCILLARY XXXXXXXXXX" << std::endl;
    for (std::size_t N = 10000; N < 10000000; N *= 10) {
        auto time_serial = fit_waterpanc(false, N, 1);
        for (short Nthreads = 2; Nthreads <= Nthread_max; ++Nthreads) {
            const bool threading = true;
            auto time_parallel = fit_waterpanc(threading, N, Nthreads);
            printf("%10d %10d %10.7f %10.7f(nothread) %10.7f(thread)\n", Nthreads, static_cast<int>(N), time_serial / time_parallel, time_serial, time_parallel);
        }
    }
}

void speedtest_decaying_exponential(short Nthread_max)
{   
    std::cout << "XXXXXXXXXX DECAYING EXPONENTIAL with N-term expansions XXXXXXXXXX" << std::endl;
    for (long N = 10; N <= 50; N += 20) {
        auto time_serial = fit_decaying_exponential(false, 10000, 1, N);
        for (short Nthreads = 2; Nthreads <= Nthread_max; ++Nthreads) {
            const bool threading = true;
            auto time_parallel = fit_decaying_exponential(threading, 10000, Nthreads, N);
            printf("%10d %10d %10.7f %10.7f(nothread) %10.7f(thread)\n", Nthreads, static_cast<int>(N), time_serial/time_parallel, time_serial, time_parallel);
        }
    }
}

int main(){
    short Nthread_max = std::min(static_cast<short>(10), static_cast<short>(std::thread::hardware_concurrency()));
#ifdef NTHREAD_MAX
    Nthread_max = NTHREAD_MAX;
#endif
    std::cout << "Max # of threads: " << Nthread_max << std::endl;
    speedtest_decaying_exponential(Nthread_max);
    speedtest_fit_polynomial(Nthread_max);
    speedtest_fit_water_ancillary(Nthread_max);
}
