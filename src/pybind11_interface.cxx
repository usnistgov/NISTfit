#ifdef PYBIND11


#include "NISTfit/abc.h"
#include "NISTfit/optimizers.h"
#include "NISTfit/numeric_evaluators.h"
#include "NISTfit/examples.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

using namespace NISTfit;

double fit_decaying_exponential(bool threading, std::size_t Nmax, short Nthreads, long N, long Nrepeat)
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
    eval->set_coefficients(opts.c0);
    if(threading){
        for (auto jj  = 0; jj < Nrepeat; ++jj){
            eval->evaluate_parallel(Nthreads);
       }
    } 
    else{
        for (auto jj  = 0; jj < Nrepeat; ++jj){
            eval->evaluate_serial(0,eval->get_outputs_size(),0);
       }
    }
    auto endTime = std::chrono::system_clock::now();
    return std::chrono::duration<double>(endTime - startTime).count();
}

void init_fitter(py::module &m){
    

    class PyFiniteDiffOutput : public FiniteDiffOutput {
    public:
        /* Inherit the constructors */
        using FiniteDiffOutput::FiniteDiffOutput;

        /* Trampoline (need one for each virtual function) */
        double call_func(const std::vector<double> &c) override {
            /* Release the GIL */
            py::gil_scoped_release release;
            {
                /* Acquire GIL before calling Python code */
                py::gil_scoped_acquire acquire;
                
                PYBIND11_OVERLOAD(
                    double,                      /* Return type */
                    FiniteDiffOutput,            /* Parent class */
                    call_func,                   /* Name of function in C++ (must match Python name) */
                    c                            /* Argument(s) */
                );
            }
        }
    };

    py::class_<AbstractOutput, std::shared_ptr<AbstractOutput> >(m, "AbstractOutput")
        .def("get_error", &AbstractOutput::get_error)
        ;

    py::class_<PolynomialOutput, AbstractOutput, std::shared_ptr<PolynomialOutput> >(m, "PolynomialOutput")
        .def(py::init<std::size_t, const std::shared_ptr<NumericInput> &>())
        ;

    py::class_<DecayingExponentialOutput, AbstractOutput, std::shared_ptr<DecayingExponentialOutput> >(m, "DecayingExponentialOutput")
        .def(py::init<int, const std::shared_ptr<NumericInput> &>())
        ;

    py::class_<FiniteDiffOutput, AbstractOutput, PyFiniteDiffOutput /* trampoline */, std::shared_ptr<FiniteDiffOutput> >(m, "FiniteDiffOutput")
        .def(py::init<const std::shared_ptr<NumericInput> &, 
                      const std::function<double(const std::vector<double> &)>,
                      const std::vector<double> &
                      >())
        ;

    py::class_<AbstractEvaluator, std::shared_ptr<AbstractEvaluator>>(m, "AbstractEvaluator")
        .def("evaluate_serial", &AbstractEvaluator::evaluate_serial)
        .def("evaluate_parallel", &AbstractEvaluator::evaluate_parallel)
        .def("get_outputs_size", &AbstractEvaluator::get_outputs_size)
        .def("add_outputs", &AbstractEvaluator::add_outputs)
        .def("get_error_vector", &AbstractEvaluator::get_error_vector, py::return_value_policy::copy)
        .def("get_affinity_scheme", &AbstractEvaluator::get_affinity_scheme)
        .def("set_affinity_scheme", &AbstractEvaluator::set_affinity_scheme)
        ;

    py::class_<NumericEvaluator, AbstractEvaluator, std::shared_ptr<NumericEvaluator> >(m, "NumericEvaluator")
        .def(py::init<>())
        .def("set_coefficients", &NumericEvaluator::set_coefficients)
        ;

    py::class_<NumericInput, std::shared_ptr<NumericInput> >(m, "NumericInput")
        .def(py::init<double, double>())
        .def("x", &NumericInput::x)
        .def("y", &NumericInput::y);

    py::class_<LevenbergMarquardtOptions>(m, "LevenbergMarquardtOptions")
        .def(py::init<>())
        .def_readwrite("c0", &LevenbergMarquardtOptions::c0)
        .def_readwrite("threading", &LevenbergMarquardtOptions::threading)
        .def_readwrite("Nthreads", &LevenbergMarquardtOptions::Nthreads)
        .def_readwrite("omega", &LevenbergMarquardtOptions::omega)
        .def_readwrite("tau0", &LevenbergMarquardtOptions::tau0)
        ;

    m.def("LevenbergMarquardt", &LevenbergMarquardt, "Fit");
    m.def("fit_decaying_exponential", &fit_decaying_exponential);

    m.def("Eigen_nbThreads", [](){ return Eigen::nbThreads(); });
    m.def("Eigen_setNbThreads", [](int Nthreads) { return Eigen::setNbThreads(Nthreads); });
}

PYBIND11_PLUGIN(NISTfit) {
    py::module m("NISTfit", "NISTfit module");

    init_fitter(m);

    return m.ptr();
}

#endif
