#ifdef PYBIND11


#include "NISTfit/abc.h"
#include "NISTfit/optimizers.h"
#include "NISTfit/numeric_evaluators.h"
#include "NISTfit/examples.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void init_fitter(py::module &m){
    using namespace NISTfit;

    py::class_<AbstractOutput>(m, "AbstractOutput")
        .def("get_error", &AbstractOutput::get_error)
        ;

    py::class_<PolynomialOutput, AbstractOutput, std::shared_ptr<PolynomialOutput> >(m, "PolynomialOutput")
        .def(py::init<std::size_t, const std::shared_ptr<NumericInput> &>())
        ;

    py::class_<DecayingExponentialOutput, AbstractOutput, std::shared_ptr<DecayingExponentialOutput> >(m, "DecayingExponentialOutput")
        .def(py::init<int, const std::shared_ptr<NumericInput> &>())
        ;

    py::class_<AbstractEvaluator>(m, "AbstractEvaluator")
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
    
}

PYBIND11_PLUGIN(PolyFitter) {
    py::module m("PolyFitter", "PolyFitter module");

    init_fitter(m);

    return m.ptr();
}

#endif
