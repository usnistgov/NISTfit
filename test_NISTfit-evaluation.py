import NISTfit
import numpy as np, timeit
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

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

def speedtest(get_eva, args, ofname, affinity = False):

    o = NISTfit.LevenbergMarquardtOptions()
    o.tau0 = 1

    fig1, ax1 = plt.subplots(1,1,figsize=(4,3))
    fig2, ax2 = plt.subplots(1,1,figsize=(4,3))

    if affinity:
        affinity_options = [(True,()),(False,[2,2])]
    else:
        affinity_options = [(False,())]
    
    for arg,c in zip(args,['b','r','c']):
        for affinity, dashes in affinity_options:
            print(arg,affinity)
            # Serial evaluation
            eva, o.c0 = get_eva(arg)
            if affinity:
                eva.set_affinity_scheme([0,2,4,6,1,3,5,7])
            Nrepeats = 100
            eva.set_coefficients(o.c0)
            N = eva.get_outputs_size()
            tic = timeit.default_timer()
            for i in range(Nrepeats):
                eva.evaluate_serial(0, N, 0)
            toc = timeit.default_timer()
            elap = toc-tic
            time_serial = elap/Nrepeats

            # Parallel evaluation
            o.threading = True
            times = []
            for Nthreads in [1,2,3,4,5,6,7,8]:
                #NISTfit.Eigen_setNbThreads(Nthreads)
                eva, o.c0 = get_eva(arg)
                if affinity:
                    eva.set_affinity_scheme([0,2,4,6,1,3,5,7])
                eva.set_coefficients(o.c0)
                elap = 0
                cfinal = eva.evaluate_parallel(Nthreads)
                tic = timeit.default_timer()
                for i in range(Nrepeats):
                    cfinal = eva.evaluate_parallel(Nthreads)
                toc = timeit.default_timer()
                elap = toc-tic
                times.append(elap/Nrepeats)
            
            line, = ax1.plot(range(1, len(times)+1),time_serial/np.array(times),color=c,dashes=dashes)
            if arg < 0:
                lbl = 'native'
            else:
                lbl = 'N: '+str(arg)

            ax2.plot(range(1, len(times)+1),np.array(times)/times[0],label = lbl,color=c,dashes=dashes)
            if affinity or len(affinity_options) == 1:
                ax1.text(len(times)-0.25, (time_serial/np.array(times))[-1], lbl, 
                         ha='right', va='center',
                         color=c,
                         bbox = dict(facecolor='w',
                                     edgecolor=line.get_color(),
                                     boxstyle='round')
                         )

    if affinity or len(affinity_options) > 1:
        ax1.plot([2,2.9],[7,7],lw=1,color='grey')
        ax1.plot([2,2.9],[6,6],lw=1,color='grey',dashes = [2,2])
        ax1.text(3,7,'Affinity',ha='left',va='center')
        ax1.text(3,6,'No affinity',ha='left',va='center')
    ax1.plot([1,8],[1,8],'k',lw=3,label='linear speedup')
    ax1.set_xlabel(r'$N_{\rm threads}$ (-)')
    ax1.set_ylabel(r'Speedup $t_{\rm serial}/t_{\rm parallel}$ (-)')
    fig1.tight_layout(pad=0.3)
    fig1.savefig(ofname)

    NN = np.linspace(1,8)
    ax2.plot(NN,1/NN,'k',lw=3,label='linear speedup')
    ax2.set_xlabel(r'$N_{\rm threads}$ (-)')
    ax2.set_ylabel(r'Total time $t_{\rm parallel}/t_{\rm 1 thread}$ (-)')
    ax2.legend(loc='best',ncol=2)
    fig2.tight_layout(pad=0.3)
    fig2.savefig('abs-'+ofname)

    plt.close('all')

if __name__=='__main__':
    speedtest(get_eval_poly, [120,12000],'speedup_polynomial.pdf')
    speedtest(get_eval_decaying_exponential, [50,5,-1], 'speedup_decaying_exponential.pdf')