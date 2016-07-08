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

class SaturationPressureOutput : public NumericOutput {
protected:
    const std::vector<double> m_e; // Exponents for terms in saturation pressure equation
    AbstractNumericEvaluator *m_evaluator; // The evaluator connected with this output
public:
    SaturationPressureOutput(const std::vector<double> &e,
                             const std::shared_ptr<NumericInput> &in,
                             AbstractNumericEvaluator *eval)
        : NumericOutput(in), m_e(e), m_evaluator(eval) {};
    
    void evaluate_one(){
        // Do the calculation
        double y = 0;
        const std::vector<double> &c = m_evaluator->get_const_coefficients();
        for (std::size_t i = 0; i < m_e.size(); ++i) {
            double term = pow(m_in->m_x, m_e[i]);
            y += c[i] * term;
            Jacobian_row[i] = term;
        }
        m_y_calc = y;
    }
};

/// The class for the evaluation of a single output value for a single input value
class SaturationPressureEvaluator : public AbstractNumericEvaluator {
public:
    SaturationPressureEvaluator(const std::vector<double> &e,
                                const std::vector<std::shared_ptr<NumericInput> > &inputs)
    {
        for (auto &in : inputs) {
            std::shared_ptr<NumericOutput> out(new SaturationPressureOutput(e, in, this));
            out->resize(e.size()); // Set the size of the Jacobian row
            m_outputs.push_back(out);
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
        auto cc = LevenbergMarquadt(eval, c, threading, Nthreads);
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
        auto cc = LevenbergMarquadt(eval, c, threading, Nthreads);
    auto endTime = std::chrono::system_clock::now();
    return std::chrono::duration<double>(endTime - startTime).count();
}


void speedtest_fit_polynomial()
{
    std::cout << "XXXXXXXXXX POLYNOMIAL XXXXXXXXXX" << std::endl;
    for (short Nthreads = 1; Nthreads <= static_cast<short>(std::thread::hardware_concurrency()); ++Nthreads) {
        for (std::size_t N = 100; N < 10000000; N *= 10) {
            std::vector<double> times;
            for (auto &threading : { true, false }) {
                auto t = fit_polynomial (threading, N, Nthreads);
                times.push_back(t);
            }
            printf("%10d %10d %10.7f %10.7f %10.7f\n", Nthreads, static_cast<int>(N), times[1] / times[0], times[1], times[0]);
        }
    }
}

void speedtest_fit_water_ancillary()
{
    std::cout << "XXXXXXXXXX WATER ANCILLARY XXXXXXXXXX" << std::endl;
    for (short Nthreads = 1; Nthreads <= static_cast<short>(std::thread::hardware_concurrency()); ++Nthreads) {
        for (std::size_t N = 100; N < 10000000; N *= 10) {
            std::vector<double> times;
            for (auto &threading : { true, false }) {
                auto t = fit_waterpanc(threading, N, Nthreads);
                times.push_back(t);
            }
            printf("%10d %10d %10.7f %10.7f %10.7f\n", Nthreads, static_cast<int>(N), times[1] / times[0], times[1], times[0]);
        }
    }
}

int main(){
    speedtest_fit_polynomial();
    speedtest_fit_water_ancillary();
}
