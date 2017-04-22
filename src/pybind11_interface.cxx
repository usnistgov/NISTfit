#ifdef PYBIND11


#include "NISTfit/abc.h"
#include "NISTfit/optimizers.h"
#include "NISTfit/numeric_evaluators.h"
#include "NISTfit/examples.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

void init_fitter(py::module &m){
    using namespace NISTfit;

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
        .def("add_outputs", &AbstractEvaluator::add_outputs)
        ;

    py::class_<NumericEvaluator, AbstractEvaluator, std::shared_ptr<NumericEvaluator> >(m, "NumericEvaluator")
        .def(py::init<>())
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

    m.def("Eigen_nbThreads", [](){ return Eigen::nbThreads(); });
    m.def("Eigen_setNbThreads", [](int Nthreads) { return Eigen::setNbThreads(Nthreads); });
}

PYBIND11_PLUGIN(PolyFitter) {
    py::module m("PolyFitter", "PolyFitter module");

    init_fitter(m);

    return m.ptr();
}

#endif
