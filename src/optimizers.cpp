#include "NISTfit/abc.h"
#include "NISTfit/optimizers.h"
#include "Eigen/Dense"
#include <cfloat>

std::vector<double> NISTfit::LevenbergMarquardt(std::shared_ptr<AbstractEvaluator> &E,
                                                LevenbergMarquardtOptions &options)
{
    double F_previous = 8888888;
    double lambda = 1;
    double nu = 2;

    std::vector<double> c =options.c0;
    
    Eigen::Map<Eigen::VectorXd> c_wrap(&c[0], c.size());
    
    for (int counter = 0; counter < 100; ++counter) {
        
        E->set_coefficients(c);
        
        (
         options.threading // Check if threading
         ? E->evaluate_parallel( (options.Nthreads < 0) ? std::thread::hardware_concurrency() : options.Nthreads) // Using threading
         : E->evaluate_serial(0, E->get_outputs_size(), 0) // Not using threading
        );
        const Eigen::MatrixXd &J = E->get_Jacobian_matrix();
        const Eigen::VectorXd &r = E->get_error_vector();
        //printf("r(min,max,mean): %g %g %g\n", r.minCoeff(), r.maxCoeff(), r.mean());
        
        double F = r.squaredNorm(); // This is actually the sum of squares of the entries in the error vector
        if (counter == 0) {
            // Madsen recommends setting tau to 1e-6 if the initial guess 
            // is believed to be a good estimate of the final solution, or 
            // larger values like 1e-3 or 1e0 if the guess value is less certain
            double tau0 = 1;
            double maxDiag = (J.transpose()*J).diagonal().maxCoeff();
            lambda = maxDiag*tau0;
        }
        
        // Levenberg-Marquardt with A*DELTAc = RHS
        Eigen::MatrixXd t2 = lambda*(J.transpose()*J).diagonal().asDiagonal();
        Eigen::MatrixXd A = J.transpose()*J + t2;
        Eigen::MatrixXd RHS = -J.transpose()*r;
        
        // Calculate the step
        Eigen::VectorXd DELTAc = options.omega*A.colPivHouseholderQr().solve(RHS);
        // Take the step
        c_wrap += DELTAc;
        
        // Resize the step (See Madsen document)
        if (counter > 0) {
            double DELTAL = 0.5*DELTAc.transpose()*(lambda*DELTAc - J.transpose()*r);
            double rho = (F_previous - F) / DELTAL;
            
            // Madsen Eq. 2.21
            if (rho > 0) {
                lambda *= std::max(1.0/3.0, 1 - pow(2 * rho - 1, 3));
                nu = 2;
            }
            else {
                lambda *= nu;
                nu *= 2;
            }
        }

        // If the residual has stopped changing, stop, no sense to keep evaluating 
        // with the same coefficients
        if (counter > 1 && std::abs(F / F_previous - 1) < 1e-10) { 
            break; 
        }

        // Copy the residual
        F_previous = F;

        // Check whether to stop
        if (F < DBL_EPSILON || std::abs(lambda) < DBL_EPSILON) { 
            break; 
        }
    }
    return c;
}