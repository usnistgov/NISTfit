import sys; sys.path.append('build/pybind11/Debug')
import PolyFitter as pf, numpy as np, time

tic = time.clock()
inputs = []
for x in np.linspace(0,1,1000):
    y = 1 + 2*x + 3*x**2 + 4*x**6
    in1 = pf.NumericInput(x, y)
    inputs.append(in1)

order = 6
eva = pf.PolynomialEvaluator(order, inputs)

o = pf.LevenbergMarquardtOptions()
o.c0 = [1.5]*(order+1)
tic = time.clock()
assert(len(o.c0) == order+1)
cfinal = pf.LevenbergMarquardt(eva, o)
print cfinal
toc = time.clock()
print toc-tic