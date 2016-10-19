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

class AncillaryOutput : public NumericOutput {
protected:
    AbstractNumericEvaluator *m_evaluator; // The evaluator connected with this output
public:
    AncillaryOutput(int Ncoeffs,
        const std::shared_ptr<NumericInput> &in,
        AbstractNumericEvaluator *eval)
        : NumericOutput(in), m_evaluator(eval)
    {
        resize(Ncoeffs);
    };

    // Do the calculation
    void evaluate_one() {
        double y = 0;
        const std::vector<double> &c = m_evaluator->get_const_coefficients();
        assert(Jacobian_row.size() == c.size());
        // y = sum_i a_i*x^(e_i)
        for (std::size_t i = 0; i < c.size(); i += 2) {
            // c[i]: coefficient, c[i+1]: exponent
            double ai = c[i], ei = c[i+1];
            double term = pow(m_in->m_x, ei);
            assert(m_in->m_x > 0);
            y += ai*term;
            Jacobian_row[i] = term; // x^e_i
            Jacobian_row[i+1] = 0*ai*term*log(m_in->m_x);
        }
        m_y_calc = y;
    }
    double evaluate(const std::vector<double> &c) {
        double y = 0;
        for (std::size_t i = 0; i < c.size(); i += 2) {
            // c[i]: coefficient, c[i+1]: exponent
            double ai = c[i], ei = c[i + 1];
            y += ai*pow(m_in->m_x, ei);
        }
        return y;
    }
    double der(std::size_t i, double dc) {
        const std::vector<double> &c0 = m_evaluator->get_const_coefficients();
        std::vector<double> cp = c0, cm = c0;
        cp[i] += dc; cm[i] -= dc;
        return (evaluate(cp) - evaluate(cm))/(2*dc);
    }
};

/// The class for the evaluation of a single output value for a single input value
class AncillaryEvaluator : public AbstractNumericEvaluator {
public:
    AncillaryEvaluator(int Ncoeffs, const std::vector<std::shared_ptr<NumericInput> > &inputs)
    {
        for (auto &in : inputs) {
            m_outputs.push_back(
                std::shared_ptr<NumericOutput>(new AncillaryOutput(Ncoeffs, in, this))
            );
        }
    };
};


double fit_waterpanc(bool threading, std::size_t N, short Nthreads) 
{    
    std::vector<double> LHS = { -3344.55, -2654.68, -2160.29, -1795.21, -1518.5, -1303.96, -1134.31, -997.77, -886.19, -793.73, -716.17, -650.41, -594.08, -545.41, -503.01, -465.81, -432.94, -403.73, -377.63, -354.18, -333.01, -313.82, -296.35, -280.39, -265.76, -252.31, -239.89, -228.41, -217.75, -207.84, -198.61, -189.98, -181.9, -174.33, -167.21, -160.51, -154.19, -148.23, -142.58, -137.24, -132.17, -127.36, -122.79, -118.43, -114.28, -110.32, -106.55, -102.93, -99.48, -96.17, -92.99, -89.95, -87.03, -84.22, -81.53, -78.93, -76.43, -74.02, -71.7, -69.46, -67.3, -65.22, -63.2, -61.25, -59.36, -57.54, -55.77, -54.06, -52.4, -50.79, -49.23, -47.72, -46.24, -44.82, -43.43, -42.08, -40.77, -39.49, -38.25, -37.04, -35.87, -34.72, -33.61, -32.52, -31.46, -30.42, -29.41, -28.43, -27.47, -26.53, -25.61, -24.72, -23.85, -22.99, -22.16, -21.34, -20.54, -19.76, -19, -18.25, -17.52, -16.8, -16.1, -15.42, -14.74, -14.08, -13.44, -12.81, -12.19, -11.58, -10.98, -10.4, -9.82, -9.26, -8.71, -8.17, -7.63, -7.11, -6.6, -6.1, -5.6, -5.12, -4.64, -4.17, -3.71, -3.26, -2.81, -2.38, -1.95, -1.52, -1.11, -0.7, -0.3, 0.1, 0.49, 0.87, 1.25, 1.62, 1.98, 2.34, 2.7, 3.04, 3.39, 3.72, 4.06, 4.38, 4.71, 5.03, 5.34, 5.65, 5.95, 6.25, 6.55, 6.84, 7.12, 7.41, 7.68, 7.96, 8.23, 8.5, 8.76, 9.02, 9.28, 9.53, 9.78, 10.03, 10.27, 10.51, 10.75, 10.98, 11.21, 11.44, 11.66, 11.88, 12.1, 12.32, 12.53, 12.74, 12.95, 13.15, 13.36, 13.75, 14.14, 14.52, 14.89, 15.25, 15.61, 15.95, 16.29, 16.61, 16.94, 17.25, 17.56, 17.86, 18.15, 18.44, 18.72, 18.99, 19.26, 19.53, 19.78, 20.04, 20.28, 20.53, 20.76, 21, 21.22, 21.45, 21.67, 21.88, 22.09, 22.3, 22.5, 22.7, 22.89, 23.08, 23.27, 23.46, 23.64, 23.81, 23.99, 24.16, 24.33, 24.49, 24.65, 24.81, 24.97, 25.12, 25.27, 25.42, 25.56, 25.71, 25.85, 25.99, 26.12, 26.25, 26.39, 26.51, 26.64, 26.77, 26.89, 27.01, 27.13, 27.24, 27.36, 27.47, 27.58, 27.69, 27.8, 27.91, 28.01, 28.11, 28.21, 28.31, 28.41, 28.51, 28.6, 28.7, 28.79, 28.88, 28.97, 29.06, 29.14, 29.23, 29.31, 29.39, 29.48, 29.56, 29.63, 29.71, 29.79, 29.86, 29.94, 30.01, 30.09, 30.16, 30.23, 30.3, 30.36, 30.43, 30.5, 32.88, 34.12 };
    std::vector<double> T_over_1000 = { 0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.2, 0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245, 0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.285, 0.29, 0.295, 0.3, 0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345, 0.35, 0.355, 0.36, 0.365, 0.37, 0.375, 0.38, 0.385, 0.39, 0.395, 0.4, 0.405, 0.41, 0.415, 0.42, 0.425, 0.43, 0.435, 0.44, 0.445, 0.45, 0.455, 0.46, 0.465, 0.47, 0.475, 0.48, 0.485, 0.49, 0.495, 0.5, 0.505, 0.51, 0.515, 0.52, 0.525, 0.53, 0.535, 0.54, 0.545, 0.55, 0.555, 0.56, 0.565, 0.57, 0.575, 0.58, 0.585, 0.59, 0.595, 0.6, 0.605, 0.61, 0.615, 0.62, 0.625, 0.63, 0.635, 0.64, 0.645, 0.65, 0.655, 0.66, 0.665, 0.67, 0.675, 0.68, 0.685, 0.69, 0.695, 0.7, 0.705, 0.71, 0.715, 0.72, 0.725, 0.73, 0.735, 0.74, 0.745, 0.75, 0.755, 0.76, 0.765, 0.77, 0.775, 0.78, 0.785, 0.79, 0.795, 0.8, 0.805, 0.81, 0.815, 0.82, 0.825, 0.83, 0.835, 0.84, 0.845, 0.85, 0.855, 0.86, 0.865, 0.87, 0.875, 0.88, 0.885, 0.89, 0.895, 0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.4, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5, 1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.6, 1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.67, 1.68, 1.69, 1.7, 1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79, 1.8, 1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89, 1.9, 1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99, 2, 2.5, 3};

    std::vector<double> c0 = { 103.56, -526.25, -1107.0, -1932.0 },
        e = { 0.05081, 1.395, 4.133, 8.554 };
        //e = { 0.216, 1.04, 3.0, 7.0 };

    //// Generate some artificial data from the ancillary curve of Wagner and Pruss for water
    //std::vector<double> c0 = { -7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502 },
    //    e = { 1, 1.5, 3, 3.5, 4, 7.5 }, theta(N), LHS(N);
    //for (std::size_t i = 0; i < N; ++i) {
    //    theta[i] = (1 - 0.5778678897721512) / (N - 1)*i + 0.5778678897721512;
    //    LHS[i] = 0;
    //    // LHS = ln(p/pc)*T/Tc
    //    for (std::size_t k = 0; k < c0.size(); ++k) {
    //        LHS[i] += c0[k] * pow(theta[i], e[k]);
    //    }
    //}

    // Create the inputs
    std::vector<std::shared_ptr<NumericInput> > inputs;
    for (int i = 0; i < LHS.size(); ++i) {
        inputs.push_back(std::shared_ptr<NumericInput>(new NumericInput(100.0/(T_over_1000[i]*1000), LHS[i])));
    }

    // Instantiate the evaluator
    std::shared_ptr<AbstractEvaluator> eval(new AncillaryEvaluator(c0.size()*2, inputs));

    // Run and time
    std::vector<double> c;
    for (int i = 0; i < c0.size(); ++i) {
        c.push_back(c0[i]);
        c.push_back(e[i]);
    }
    std::vector<std::shared_ptr<AbstractOutput> > outs;
    auto startTime = std::chrono::system_clock::now();
    auto cc = LevenbergMarquadt(eval, c, threading, Nthreads);
    for (int i = 0; i < cc.size(); i += 2) {
        std::cout << cc[i] << " " << cc[i+1] << std::endl;
    }
    return std::chrono::duration<double>(std::chrono::system_clock::now() - startTime).count();
}

int main() {
    fit_waterpanc(false, 1000, 4);    
}