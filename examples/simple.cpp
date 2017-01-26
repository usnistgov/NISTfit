//
//  Inspired by http://www.drdobbs.com/cpp/c11s-async-template/240001196
//  See also http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf
//

#include "NISTfit/abc.h"
#include "NISTfit/optimizers.h"
#include "NISTfit/numeric_evaluators.h"

#include <iostream>
#include <chrono>

using namespace NISTfit;

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
    double y = 0;
    for (int m=0; m < N; ++m){
        y += pow(x, m)/factorial(m);
    }
    return y;
}
double sin_expansion(double x, int N) {
    double y = 0;
    for (int m = 0; m < N; ++m) {
        y += pow(-1, m)*pow(x, 2*m+1)/factorial(2*m+1);
    }
    return y;
}
double cos_expansion(double x, int N) {
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
        : NumericOutput(in), N(N) {};
    /// In the highly unlikely case of an exception in this class (implementation of this method is required), 
    /// set the calculated value to something very large
    void exception_handler() { m_y_calc = 100000; }
    void evaluate_one() {
        // Do the calculation
        const std::vector<double> &c = get_AbstractEvaluator()->get_const_coefficients();
        const double x = m_in->x(), e = exp_expansion(-c[0] * x, N), s1 = sin_expansion(c[1] * x, N), c2 = cos_expansion(c[2] * x, N);
        double y = e*s1*c2;
        Jacobian_row[0] = -x*y;
        Jacobian_row[1] = x*e*cos_expansion(c[1]*x,N)*c2;
        Jacobian_row[2] = -x*e*s1*sin_expansion(c[2]*x,N);
        m_y_calc = y;
    }
};

/// The class for the evaluation of a single output value for a single input value
class DecayingExponentialEvaluator : public NumericEvaluator {
public:
    DecayingExponentialEvaluator(int N, const std::vector<std::shared_ptr<NumericInput> > &inputs)
    {
        for (auto &in : inputs) {
            std::shared_ptr<NumericOutput> out(new DecayingExponentialOutput(N, in));
            out->resize(3); // Set the size of the Jacobian row
            add_output(out);
        }
    };
}; 

class SaturationPressureOutput : public NumericOutput {
protected:
    const std::vector<double> m_e; // Exponents for terms in saturation pressure equation
public:
    SaturationPressureOutput(const std::vector<double> &e,
                             const std::shared_ptr<NumericInput> &in)
        : NumericOutput(in), m_e(e) {};
    /// In the highly unlikely case of an exception in this class (implementation of this method is required), 
    /// set the calculated value to something very large
    void exception_handler(){ m_y_calc = 100000; }
    void evaluate_one(){
        // Do the calculation
        double y = 0;
        const std::vector<double> &c = get_AbstractEvaluator()->get_const_coefficients();
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

/// The class for the evaluation of a single output value for a single input value
class SaturationPressureEvaluator : public NumericEvaluator {
public:
    SaturationPressureEvaluator(const std::vector<double> &e,
                                const std::vector<std::shared_ptr<NumericInput> > &inputs)
    {
        for (auto &in : inputs) {
            std::shared_ptr<NumericOutput> out(new SaturationPressureOutput(e, in));
            out->resize(e.size()); // Set the size of the Jacobian row
            add_output(out);
        }
    };
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

    // Create the inputs
    std::vector<std::shared_ptr<NumericInput> > inputs; 
    for (int i = 0; i < theta.size(); ++i) {
        inputs.push_back(std::shared_ptr<NumericInput>(new NumericInput(theta[i], LHS[i])));
    }
    
    // Instantiate the evaluator
    std::shared_ptr<AbstractEvaluator> eval(new SaturationPressureEvaluator({ 1, 1.5, 3, 3.5, 4, 7.5 }, inputs));
    
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
    std::vector<std::shared_ptr<NumericInput> > inputs;
    for (double i = 0; i < Nmax; ++i) {
        double x = i / ((double)Nmax);
        double y = 1 + 2*x + 3*x*x;
        inputs.push_back(std::shared_ptr<NumericInput>(new NumericInput(x, y)));
    }
    std::shared_ptr<AbstractEvaluator> eval(new PolynomialEvaluator(2, inputs));
    
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
    std::vector<std::shared_ptr<NumericInput> > inputs;
    for (double i = 0; i < Nmax; ++i) {
        double x = i / ((double)Nmax);
        double y = exp(-a*x)*sin(b*x)*cos(c*x);
        inputs.push_back(std::shared_ptr<NumericInput>(new NumericInput(x, y)));
    }
    std::shared_ptr<AbstractEvaluator> eval(new DecayingExponentialEvaluator(N, inputs));

    std::vector<double> c0 = { 0.05, 1, 1.5 };
    auto startTime = std::chrono::system_clock::now();
    auto opts = LevenbergMarquardtOptions();
    opts.c0 = c0; opts.threading = threading; opts.Nthreads = Nthreads;
    auto cc = LevenbergMarquardt(eval, opts);
    auto endTime = std::chrono::system_clock::now();
    return std::chrono::duration<double>(endTime - startTime).count();
}


void speedtest_fit_polynomial()
{
    short Nthread_max = std::min(static_cast<short>(10), static_cast<short>(std::thread::hardware_concurrency()));
    std::cout << "XXXXXXXXXX POLYNOMIAL XXXXXXXXXX" << std::endl;
    for (short Nthreads = 1; Nthreads <= Nthread_max; ++Nthreads) {
        for (std::size_t N = 100; N < 10000000; N *= 10) {
            std::vector<double> times;
            for (auto &threading : { true, false }) {
                auto t = fit_polynomial (threading, N, Nthreads);
                times.push_back(t);
            }
            printf("%10d %10d %10.7f %10.7f(nothread) %10.7f(thread)\n", Nthreads, static_cast<int>(N), times[1] / times[0], times[1], times[0]);
        }
    }
}

void speedtest_fit_water_ancillary()
{
    short Nthread_max = std::min(static_cast<short>(10), static_cast<short>(std::thread::hardware_concurrency()));
    std::cout << "XXXXXXXXXX WATER ANCILLARY XXXXXXXXXX" << std::endl;
    for (short Nthreads = 1; Nthreads <= Nthread_max; ++Nthreads) {
        for (std::size_t N = 100; N < 10000000; N *= 10) {
            std::vector<double> times;
            for (auto &threading : { true, false }) {
                auto t = fit_waterpanc(threading, N, Nthreads);
                times.push_back(t);
            }
            printf("%10d %10d %10.7f %10.7f(nothread) %10.7f(thread)\n", Nthreads, static_cast<int>(N), times[1] / times[0], times[1], times[0]);
        }
    }
}

void speedtest_decaying_exponential()
{
    short Nthread_max = std::min(static_cast<short>(10), static_cast<short>(std::thread::hardware_concurrency()));
    std::cout << "XXXXXXXXXX DECAYING EXPONENTIAL with 50-term expansions XXXXXXXXXX" << std::endl;
    for (short Nthreads = 1; Nthreads <= Nthread_max; ++Nthreads) {
        for (long N = 50; N < 100; N *= 10) {
            std::vector<double> times;
            for (auto &threading : { true, false }) {
                auto t = fit_decaying_exponential(threading, 10000, Nthreads, N);
                times.push_back(t);
            }
            printf("%10d %10d %10.7f %10.7f(nothread) %10.7f(thread)\n", Nthreads, static_cast<int>(N), times[1] / times[0], times[1], times[0]);
        }
    }
}

int main(){
    std::cout << "hardware_concurrency:" << std::thread::hardware_concurrency() << std::endl;
    speedtest_decaying_exponential();
    speedtest_fit_polynomial();
    speedtest_fit_water_ancillary();
}
