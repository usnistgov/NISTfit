import sys; sys.path.append('build/pybind11/Debug')
import PolyFitter as pf, numpy as np, time
import matplotlib.pyplot as plt

def polynomial():
    x = np.linspace(0,1,10000)
    y = 1 + 2*x + 3*x**2 + 4*x**6
    inputs = [pf.NumericInput(_x, _y) for _x,_y in zip(x, y)]
    order = 6
    eva = pf.PolynomialEvaluator(order, inputs)

    tic = time.clock()
    o = pf.LevenbergMarquardtOptions()
    o.c0 = [1.5]*(order+1)
    o.threading = True
    o.Nthreads = 4

    tic = time.clock()
    assert(len(o.c0) == order+1)
    cfinal = pf.LevenbergMarquardt(eva, o)
    toc = time.clock()
    print cfinal
    print toc-tic

def decaying_exponential():

    a = 2; b = 3; c = 1.2;
    x = np.linspace(0,2,500)
    y = np.exp(-a*x)*np.sin(b*x)*np.cos(c*x)
    inputs = [pf.NumericInput(_x, _y) for _x,_y in zip(x, y)]
    N = 10 # order of Taylor series
    eva = pf.DecayingExponentialEvaluator(N, inputs)

    tic = time.clock()
    o = pf.LevenbergMarquardtOptions()
    o.c0 = [0.1,0.1,0.1]
    o.threading = True
    o.Nthreads = 4

    tic = time.clock()
    cfinal = pf.LevenbergMarquardt(eva, o)
    toc = time.clock()
    print cfinal
    print toc-tic

    plt.plot(x,y)
    a,b,c = cfinal
    yfit = np.exp(-a*x)*np.sin(b*x)*np.cos(c*x)
    plt.plot(x,yfit)
    plt.show()

if __name__=='__main__':
    polynomial()
    decaying_exponential()