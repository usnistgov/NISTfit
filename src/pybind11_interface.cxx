#ifdef PYBIND11


#include "NISTfit/abc.h"
#include "NISTfit/optimizers.h"
#include "NISTfit/numeric_evaluators.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void init_fitter(py::module &m){
    using namespace NISTfit;

    py::class_<NumericInput, std::shared_ptr<NumericInput> >(m, "NumericInput")
        .def(py::init<double, double>())
        .def("x", &NumericInput::x)
        .def("y", &NumericInput::y);

    py::class_<PolynomialOutput>(m, "PolynomialOutput")
        .def(py::init<std::size_t, const std::shared_ptr<NumericInput> &>())
        ;

    py::class_<AbstractEvaluator>(m, "AbstractEvaluator")
        .def("evaluate_serial", &AbstractEvaluator::evaluate_serial)
        ;

    py::class_<PolynomialEvaluator, AbstractEvaluator, std::shared_ptr<PolynomialEvaluator> >(m, "PolynomialEvaluator")
        .def(py::init<std::size_t, const std::vector<std::shared_ptr<NumericInput> > & >())

        ;

    py::class_<LevenbergMarquardtOptions>(m, "LevenbergMarquardtOptions")
        .def(py::init<>())
        .def_readwrite("c0", &LevenbergMarquardtOptions::c0)
        .def_readwrite("threading", &LevenbergMarquardtOptions::threading)
        .def_readwrite("Nthreads", &LevenbergMarquardtOptions::Nthreads)
        .def_readwrite("omega", &LevenbergMarquardtOptions::omega)
        ;

    m.def("LevenbergMarquardt", &LevenbergMarquardt, "Fit");
    
}

PYBIND11_PLUGIN(PolyFitter) {
    py::module m("PolyFitter", "PolyFitter module");

    init_fitter(m);

    return m.ptr();
}

#endif
