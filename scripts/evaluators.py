import NISTfit
import numpy as np

def get_eval_poly(Npoints):
    x = np.linspace(0,1,Npoints)
    y = 1 + 2*x + 3*x**2 + 4*x**6
    order = 6
    outputs = [NISTfit.PolynomialOutput(order, NISTfit.NumericInput(_x, _y)) 
               for _x,_y in zip(x, y)]
    eva = NISTfit.NumericEvaluator()
    eva.add_outputs(outputs)
    return eva, [1.5]*(order+1)

def get_eval_decaying_exponential(Norder):
    a = 0.2; b = 3; c = 1.3;
    x = np.linspace(0, 2, 1200)
    y = np.exp(-a*x)*np.sin(b*x)*np.cos(c*x)
    outputs = [NISTfit.DecayingExponentialOutput(Norder, NISTfit.NumericInput(_x, _y)) 
               for _x,_y in zip(x, y)]
    eva = NISTfit.NumericEvaluator()
    eva.add_outputs(outputs)
    return eva, [0.5, 2, 0.8]

def get_eval_decaying_exponential_finite_diff(Norder):
    a = 0.2; b = 3; c = 1.3;
    x = np.linspace(0, 2, 1000)
    y = np.exp(-a*x)*np.sin(b*x)*np.cos(c*x)
    dc = [0.01]*3 # epsilon used for each coefficient in the finite difference

    outputs = []

    for _x, _y in zip(x, y):
        # def f(c, x):
        #     np.exp(-c[0]*self.x)*np.sin(c[1]*self.x)*np.cos(c[2]*self.x)
        # o = pf.FiniteDiffOutput(pf.NumericInput(_x, _y), functools.partial(f, x=_x),dc)

        class Output(pf.FiniteDiffOutput):
            def __init__(self, input, x):
                super(Output, self).__init__(input,lambda c: c[0]+c[1], dc)
                self.x = x

            def call_func(self, c):
                return np.exp(-c[0]*self.x)*np.sin(c[1]*self.x)*np.cos(c[2]*self.x)
        
        o = Output(pf.NumericInput(_x, _y), _x)
        outputs.append(o)

    eva = pf.NumericEvaluator()
    eva.add_outputs(outputs)
    return eva, [0.5, 2, 0.8]    