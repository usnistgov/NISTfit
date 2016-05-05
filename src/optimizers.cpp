#include "NISTfit/abc.h"
#include "NISTfit/optimizers.h"
#include "Eigen/Eigen/Dense"
#include <cfloat>

void NISTfit::LevenbergMarquadt(std::shared_ptr<AbstractEvaluator> &E,
                       std::vector<std::shared_ptr<AbstractInput> > &inputs,
                       std::vector<double> &c0)
{
    double F_previous = 8888888;
    double lambda = 1;
    double nu = 2;
    bool threading = true;

    std::vector<double> c = c0;
    
    Eigen::Map<Eigen::VectorXd> c_wrap(&c[0], c.size());
    
    for (int counter = 0; counter < 40; ++counter) {
        
        E->set_coefficients(c);
        
        auto outs = (
                     threading // Check if threading
                     ? E->evaluate_parallel(inputs, std::thread::hardware_concurrency()) // Using threading
                     : E->evaluate_serial(inputs, 0, inputs.size()) // Not using threading
                     );
        Eigen::MatrixXd J = E->get_Jacobian_matrix(outs);
        Eigen::VectorXd r = E->get_error_vector(outs);
        
        double F = r.squaredNorm();
        if (counter == 0) {
            double tau0 = 0.00001;
            double maxDiag = (J.transpose()*J).diagonal().maxCoeff();
            lambda = maxDiag*tau0;
        }
        
        // Levenberg-Marquadt with A*DELTAc = RHS
        Eigen::MatrixXd t2 = lambda*(J.transpose()*J).diagonal().asDiagonal();
        Eigen::MatrixXd A = J.transpose()*J + t2;
        Eigen::MatrixXd RHS = -J.transpose()*r;
        
        // Calculate the step
        Eigen::VectorXd DELTAc = A.colPivHouseholderQr().solve(RHS);
        // Take the step
        c_wrap += DELTAc;
        
        // Resize the step (See Madsen document)
        if (counter > 0) {
            double DELTAL = 0.5*DELTAc.transpose()*(lambda*DELTAc - J.transpose()*r);
            double rho = (F_previous - F) / DELTAL;
            
            if (rho > 0) {
                lambda *= std::max(1.0 / 3.0, 1 - pow(2 * rho - 1, 3));
                nu = 2;
            }
            else {
                lambda *= nu;
                nu *= 2;
            }
        }
        F_previous = F;
        if (F < DBL_EPSILON) { break; }
    }
}