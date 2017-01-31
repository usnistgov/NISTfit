import sys; sys.path.append('build/pybind11/Release')
import PolyFitter as pf, numpy as np, time
import matplotlib.pyplot as plt

def get_eval_poly(Npoints):
    x = np.linspace(0,1,Npoints)
    y = 1 + 2*x + 3*x**2 + 4*x**6
    order = 6
    outputs = [pf.PolynomialOutput(order, pf.NumericInput(_x, _y)) for _x,_y in zip(x, y)]
    eva = pf.NumericEvaluator()
    eva.add_outputs(outputs)
    return eva, [1.5]*(order+1)

def get_eval_decaying_exponential(Norder):
    a = 0.2; b = 3; c = 1.3;
    x = np.linspace(0, 2, 1000)
    y = np.exp(-a*x)*np.sin(b*x)*np.cos(c*x)
    outputs = [pf.DecayingExponentialOutput(Norder, pf.NumericInput(_x, _y)) for _x,_y in zip(x, y)]
    eva = pf.NumericEvaluator()
    eva.add_outputs(outputs)
    return eva, [0.5, 2, 0.8]

def speedtest(get_eva, args, ofname):

    o = pf.LevenbergMarquardtOptions()
    o.tau0 = 1
    
    for arg in args: # order of Taylor series expansion
        
        # Serial evaluation
        eva, o.c0 = get_eva(arg)
        tic = time.clock()
        o.threading = False
        cfinal = pf.LevenbergMarquardt(eva, o)
        toc = time.clock()
        time_serial = toc-tic

        # Parallel evaluation
        o.threading = True
        times = []
        for Nthreads in [1,2,3,4,5,6,7,8]:
            eva, o.c0 = get_eva(arg)
            o.Nthreads = Nthreads
            tic = time.clock()
            cfinal = pf.LevenbergMarquardt(eva, o)
            toc = time.clock()
            times.append(toc-tic)
        plt.plot(range(1, len(times)+1),time_serial/np.array(times),label='N: '+str(arg))

    plt.legend(loc='best')
    plt.plot([1,8],[1,8],lw=3)
    plt.xlabel(r'$N_{\rm threads}$ (-)')
    plt.ylabel(r'Speedup $t_{\rm serial}/t_{\rm parallel}$ (-)')
    plt.savefig(ofname)
    plt.show()

if __name__=='__main__':
    speedtest(get_eval_poly, [100,1000,10000],'speedup_polynomial.pdf')
    speedtest(get_eval_decaying_exponential, [10,20,30,40,50], 'speedup_decaying_exponential.pdf')