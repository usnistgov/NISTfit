from __future__ import division, print_function

import json
import sys
import timeit

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Common module with the generators
import evaluators
import NISTfit

def speedtest(get_eva, args, ofname, Nthreads_max = 8, affinity = False):

    o = NISTfit.LevenbergMarquardtOptions()
    o.tau0 = 1

    fig, ax1 = plt.subplots(1,1,figsize=(4,3))
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
            Nrepeat = 40
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
            Nthreads_list = range(1, Nthreads_max+1)
            for Nthreads in Nthreads_list:
                NISTfit.Eigen_setNbThreads(Nthreads)
                eva, o.c0 = get_eva(arg)
                o.Nthreads = Nthreads
                elap = 0
                tic = timeit.default_timer()
                for i in range(Nrepeat):
                    cfinal = NISTfit.LevenbergMarquardt(eva, o)
                elap = timeit.default_timer() - tic
                times.append(elap/Nrepeat)
            
            ax1.plot(Nthreads_list,time_serial/np.array(times), 
                     color = c, dashes = dashes)
            if arg < 0:
                lbl = 'native'
            else:
                lbl = 'N: '+str(arg)
            ax2.plot(Nthreads_list, np.array(times)/times[0], color = c, 
                     dashes = dashes, label = lbl)
            if affinity or len(affinity_options) == 1:
                ax1.text(len(times)-0.25, (time_serial/np.array(times))[-1], lbl, 
                     ha='right', va='center',
                     color=c,
                     bbox = dict(facecolor='w',
                                 edgecolor=c,
                                 boxstyle='round')
                     )

    if affinity or len(affinity_options) > 1:
        ax1.plot([2,2.9],[7,7],lw=1,color='grey')
        ax1.plot([2,2.9],[6,6],lw=1,color='grey',dashes = [2,2])
        ax1.text(3,7,'Affinity',ha='left',va='center')
        ax1.text(3,6,'No affinity',ha='left',va='center')
    ax1.plot([1,Nthreads_max],[1,Nthreads_max],'k',lw=3,label='linear speedup')
    ax1.set_xlabel(r'$N_{\rm threads}$ (-)')
    ax1.set_ylabel(r'Speedup $t_{\rm serial}/t_{\rm parallel}$ (-)')
    fig.tight_layout(pad=0.3)
    fig.savefig(ofname)

    NN = np.linspace(1,8)
    ax2.plot(NN,1/NN,'k',lw=3,label='linear speedup')
    ax2.set_xlabel(r'$N_{\rm threads}$ (-)')
    ax2.set_ylabel(r'Total time $t_{\rm parallel}/t_{\rm 1 thread}$ (-)')
    ax2.legend(loc='best',ncol=2)
    fig2.tight_layout(pad=0.3)
    fig2.savefig('abs-'+ofname)

    plt.close('all')

if __name__=='__main__':
    # Allow for the number of threads to be provided at the command line as the argument to this script
    Nthreads_max = 8
    if len(sys.argv) == 2:
        Nthreads_max = int(sys.argv[-1])

    speedtest(evaluators.get_eval_poly, [120,12000],'LM_speedup_polynomial.pdf',
              Nthreads_max = Nthreads_max)
    speedtest(evaluators.get_eval_decaying_exponential, [50,5,-1], 
              'LM_speedup_decaying_exponential.pdf',
              Nthreads_max = Nthreads_max)
    # speedtest(get_eval_decaying_exponential_finite_diff, [1], 
    #           'LM_speedup_decaying_exponential_finite_diff.pdf')