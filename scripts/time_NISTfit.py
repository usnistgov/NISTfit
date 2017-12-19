from __future__ import division, print_function

# Standard libraries (always available)
import json
import sys
import timeit

# Conda-installable packages
import numpy as np
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Common module with the generators
import evaluators
import NISTfit

def generate_results(get_eva, args, ofname, method = 'evaluate', Nthreads_max = 8, 
                     affinity = True, Nrepeats = 100, Eigen_threads = True):

    o = NISTfit.LevenbergMarquardtOptions()
    o.tau0 = 1

    windows_scheme = list(range(0,Nthreads_max,2)) + list(range(1,Nthreads_max,2))
    linux_scheme = list(range(0,Nthreads_max))
    if sys.platform.startswith('win'):
        scheme = windows_scheme 
    else:
        scheme = linux_scheme

    if affinity:
        affinity_options = [(True,()),(False,[2,2])]
    else:
        affinity_options = [(False,())]

    data = {'filename': ofname, 'times':[], 'Nthreads_max': Nthreads_max, 
            'affinity': affinity, 'affinity_options': affinity_options,
            'args' : args}
    
    for arg in args:
        for affinity, affinity_dashes in affinity_options:
            print(arg,affinity)
            
            reject = int(0.0*Nrepeats)
            
            # Serial evaluation
            eva, o.c0 = get_eva(arg)
            if affinity:
                eva.set_affinity_scheme(scheme)
            eva.set_coefficients(o.c0)
            N = eva.get_outputs_size()
            if method == 'evaluate':
                times = eva.time_evaluate_serial(Nrepeats)
            elif method == 'LM':
                o.threading = False
                times = NISTfit.time_LevenbergMarquardt(eva, o, Nrepeats)
            else:
                raise ValueError("Bad method")
            elap = np.min(np.sort(times)[reject:Nrepeats-reject])
            
            data['times'].append(dict(arg=arg, type='serial', Nthreads = 0,
                                      time = elap, affinity = affinity))

            # Parallel evaluation
            o.threading = True
            Nthreads_list = range(1, Nthreads_max+1)
            for Nthreads in Nthreads_list:
                if Eigen_threads:
                    NISTfit.Eigen_setNbThreads(Nthreads)
                eva, o.c0 = get_eva(arg)
                o.Nthreads = Nthreads
                eva.set_coefficients(o.c0)
                if affinity:
                    eva.set_affinity_scheme(scheme)
                elap = None
                cfinal = eva.evaluate_parallel(Nthreads)
                if method == 'evaluate':
                    times = eva.time_evaluate_parallel(Nthreads, Nrepeats)
                elif method == 'LM':
                    times = NISTfit.time_LevenbergMarquardt(eva, o, Nrepeats)
                else:
                    raise ValueError("Bad method")
                elap = np.min(np.sort(times)[reject:Nrepeats-reject])
                
                data['times'].append(dict(arg=arg, type='parallel', 
                                          Nthreads=Nthreads, time=elap,
                                          affinity=affinity))

    with open('timing-'+ofname+'.json','w') as fp:
        fp.write(json.dumps(data, indent =2))

def plot_results(ofname):

    with open('timing-'+ofname+'.json') as fp:
        _data = json.load(fp)
        Nthreads_max = _data['Nthreads_max']
        Nthreads_list = range(1, Nthreads_max+1)
        affinity_options = _data['affinity_options']
        args = _data['args']
        df = pandas.DataFrame(_data['times'])

    fig1, ax1 = plt.subplots(1,1,figsize=(4,3))
    fig2, ax2 = plt.subplots(1,1,figsize=(4,3))

    if np.sum(df.affinity)>0 or len(affinity_options) > 1:
        ax1.plot([2,2.9],[7,7],lw=1,color='grey')
        ax1.plot([2,2.9],[6,6],lw=1,color='grey',dashes = [2,2])
        ax1.text(3,7,'Affinity',ha='left',va='center')
        ax1.text(3,6,'No affinity',ha='left',va='center')

    for arg, c in zip(args,['b','r','c','k']):
        for affinity, dashes in affinity_options:

            # Extract data for this arg from the pandas DataFrame
            time_serial = float(df[(df.arg == arg) & (df.Nthreads == 0) & (df.affinity == affinity)].time)
            times = np.array(df[(df.arg == arg) & (df.Nthreads >= 1) & (df.affinity == affinity)].time)

            line, = ax1.plot(Nthreads_list,time_serial/np.array(times),
                             color=c,dashes=dashes)

            if arg < 0:
                lbl = 'native'
            else:
                lbl = 'N: '+str(arg)

            ax2.plot(Nthreads_list, np.array(times)/times[0],label = lbl,
                     color=c,dashes=dashes)
            if affinity or len(affinity_options) == 1:
                ax1.text(len(times)-0.25, (time_serial/np.array(times))[-1], lbl, 
                         ha='right', va='center',
                         color=c,
                         bbox = dict(facecolor='w',
                                     edgecolor=line.get_color(),
                                     boxstyle='round')
                         )

    ax1.plot([1,Nthreads_max],[1,Nthreads_max],'k',lw=3,label='linear speedup')
    ax1.set_xlabel(r'$N_{\rm threads}$ (-)')
    ax1.set_ylabel(r'Speedup $t_{\rm serial}/t_{\rm parallel}$ (-)')
    fig1.tight_layout(pad=0.3)
    fig1.savefig(ofname + '.pdf', transparent = True)

    # NN = np.linspace(1,Nthreads_max)
    # ax2.plot(NN,1/NN,'k',lw=3,label='linear speedup')
    # ax2.set_xlabel(r'$N_{\rm threads}$ (-)')
    # ax2.set_ylabel(r'Total time $t_{\rm parallel}/t_{\rm 1 thread}$ (-)')
    # ax2.legend(loc='best',ncol=2)
    # fig2.tight_layout(pad=0.3)
    # fig2.savefig('abs-'+ofname + '.pdf', transparent = True)

    plt.close('all')

if __name__=='__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Run the timing tests for NISTfit')
    parser.add_argument('Nthreads_max', metavar='Nthreads_max', type=int, nargs=1, help="The maximum number of threads")
    parser.add_argument('--affinity-too', nargs='?', const=True, default=False, help="If defined, the affinity tests will also be run (windows only)")
    args = parser.parse_args()
    
    for method in ['evaluate','LM']:
        # Many fewer evaluations for LM than for normal evaluation
        divisor = 1
        if method == 'LM':
            divisor = 10
        ofname = method+'-speedup_polynomial'
        generate_results(evaluators.get_eval_poly, [120,12000], ofname,
                         Nthreads_max = args.Nthreads_max[0], method = method, 
                         Nrepeats = 500//divisor, affinity = args.affinity_too)
        plot_results(ofname)

        ofname = method+'-speedup_decaying_exponential'
        generate_results(evaluators.get_eval_decaying_exponential, [50,5,-1], 
                         ofname, Nthreads_max = args.Nthreads_max[0], method = method,
                         Nrepeats = 200//divisor, affinity= args.affinity_too)
        plot_results(ofname)