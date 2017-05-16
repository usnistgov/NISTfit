import NISTfit
import numpy as np
import timeit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import functools

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

def speedtest(get_eva, args, ofname):

    o = NISTfit.LevenbergMarquardtOptions()
    o.tau0 = 1

    fig, ax1 = plt.subplots(1,1,figsize=(4,3))
    fig2, ax2 = plt.subplots(1,1,figsize=(4,3))
    
    for arg,c in zip(args,['b','r','c']):
        for affinity, dashes in [(True,()),(False,[2,2])]:
            print(arg,affinity)
            # Serial evaluation
            eva, o.c0 = get_eva(arg)
            Nrepeat = 20
            elap = 0
            tic = timeit.default_timer()
            o.threading = False
            for i in range(Nrepeat):
                cfinal = NISTfit.LevenbergMarquardt(eva, o)
            elap = timeit.default_timer() - tic
            time_serial = elap/Nrepeat

            # Parallel evaluation
            o.threading = True
            times = []
            for Nthreads in [1,2,3,4,5,6,7,8]:
                NISTfit.Eigen_setNbThreads(Nthreads)
                eva, o.c0 = get_eva(arg)
                o.Nthreads = Nthreads
                elap = 0
                tic = timeit.default_timer()
                for i in range(Nrepeat):
                    cfinal = NISTfit.LevenbergMarquardt(eva, o)
                elap = timeit.default_timer() - tic
                times.append(elap/Nrepeat)
            
            ax1.plot(range(1, len(times)+1),time_serial/np.array(times), color = c, dashes = dashes)
            if arg < 0:
                lbl = 'native'
            else:
                lbl = 'N: '+str(arg)
            ax2.plot(range(1, len(times)+1), np.array(times)/times[0], color = c, dashes = dashes, label = lbl)
            if affinity:
                ax1.text(len(times)-0.25, (time_serial/np.array(times))[-1], lbl, 
                     ha='right', va='center',
                     color=c,
                     bbox = dict(facecolor='w',
                                 edgecolor=c,
                                 boxstyle='round')
                     )

    ax1.plot([2,2.9],[7,7],lw=1,color='grey')
    ax1.plot([2,2.9],[6,6],lw=1,color='grey',dashes = [2,2])
    ax1.text(3,7,'Affinity',ha='left',va='center')
    ax1.text(3,6,'No affinity',ha='left',va='center')
    ax1.plot([1,8],[1,8],'k',lw=3,label='linear speedup')
    ax1.set_xlabel(r'$N_{\rm threads}$ (-)')
    ax1.set_ylabel(r'Speedup $t_{\rm serial}/t_{\rm parallel}$ (-)')
    fig.tight_layout(pad=0.3)
    fig.savefig(ofname)

    ax2.set_xlabel(r'$N_{\rm threads}$ (-)')
    ax2.set_ylabel(r'Speedup $t_{\rm serial}/t_{\rm parallel}$ (-)')
    ax2.legend(loc='best',ncol=2)
    fig2.tight_layout(pad=0.3)
    fig2.savefig('total')

    plt.close('all')

if __name__=='__main__':
    speedtest(get_eval_poly, [120,12000],'LM_speedup_polynomial.pdf')
    speedtest(get_eval_decaying_exponential, [50,5,-1], 
              'LM_speedup_decaying_exponential.pdf')
    # speedtest(get_eval_decaying_exponential_finite_diff, [1], 
    #           'LM_speedup_decaying_exponential_finite_diff.pdf')