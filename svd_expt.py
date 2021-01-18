import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from scipy.stats import zscore
import importlib
import zipfile
import math
import utils
import scipy as sp
from scipy import io
import scipy.signal
from scipy.sparse.linalg import eigsh
import csv
import timeit
import power_law

# algorithm for randomized SVD
def randomized_SVD(R,K):
    N = R.shape[1]
    start = timeit.default_timer()
    Z = np.random.standard_normal((N,K))
    Y = R @ Z
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ R
    U,S,V = np.linalg.svd(B)
    U = Q @ U
    end = timeit.default_timer()
    T = end - start
    inds = np.argsort(S)[::-1]
    S = S[inds]
    U = U[:,inds]
    V = V[inds,:]
    return U,S,V, T

# standard SVD
def standard_SVD(R):
    start = timeit.default_timer()
    U, S, V = np.linalg.svd(R)
    end = timeit.default_timer()
    T = end-start
    inds = np.argsort(S)[::-1]
    U = U[:,inds]
    V = V[inds,:]
    S = S[inds]
    return U,S,V,T

# comparison of alignment of normalized eigenvectors
def compare_orthogonal_matrices(U,V):
    num = min(U.shape[1],V.shape[1])
    dots = np.zeros(num)
    for i in range(num):
        dots[i] = (np.dot(U[:,i], V[:,i]))**2
    return dots

plt.rcParams.update({'font.size': 12})


# data downsloaded from Pachitariu, Michelos, Stringer 2019.
# at  https://janelia.figshare.com/articles/Recordings_of_20_000_neurons_from_V1_in_response_to_oriented_stimuli/8279387
dataroot = 'grating_data'
db = np.load(os.path.join(dataroot, 'database.npy'), allow_pickle=True)
fs = []

# iterate over all datasets
all_mouse_names = []
mouse_dict = {}
for di in db:
    mname = di['mouse_name']
    if mname not in mouse_dict:
        mouse_dict[mname] = []

    datexp = di['date']
    blk = di['block']
    stype = di['expt']

    fname = '%s_%s_%s_%s.npy'%(stype, mname, datexp, blk)
    fs.append(os.path.join(dataroot, fname))

count = 0
maxcount = 5


npc = 20
fs_all = fs
#fs = [fs[0]]
#fs = fs[0:1]
all_spectra = []
all_alphas = []


count = 0


# size of the grid of orientations on [0,2 pi]
num_stim = 200

# iterate over the datasets
for t,f in enumerate(fs):

    if t > 0:
        break

    F1_indices = []
    F2_indices = []

    if count > 5:
        break
    count += 1

    dat = np.load(f, allow_pickle=True).item()
    sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)


    # perform trial averaging to smooth out the responses

    stim_vals = np.linspace(0,2*math.pi, num_stim)
    resp_avg = np.zeros( (sresp.shape[0], num_stim) )
    density = np.zeros( len(stim_vals))
    for i in range(num_stim-1):
        stim_inds = [j for j in range(len(istim)) if istim[j] < stim_vals[i+1] and istim[j] > stim_vals[i]]
        resp_avg[:,i] = np.mean( sresp[:,stim_inds] , axis = 1)
        density[i] = len(stim_inds)
    #plt.plot(stim_vals, density)
    #plt.show()
    #plt.plot(stim_vals, resp_avg[0,:])
    #plt.show()

    resp_avg = resp_avg[:,0:resp_avg.shape[1]-1]
    stim_vals = stim_vals[0:stim_vals.shape[0]-1]
    resp_avg = 1/np.sqrt(resp_avg.shape[0]*resp_avg.shape[1]) * resp_avg

    resp_avg = resp_avg

    # number of dimensions for random projection
    K = 50

    # compare random and standard SVD
    U, S,V, T = standard_SVD(resp_avg)
    Uk, Sk, Vk, Tk = randomized_SVD(resp_avg, K)
    print("T, Tk = (%0.4f, %0.4f)" % (T,Tk))
    plt.loglog(S, label = 'Full SVD')
    plt.loglog(Sk, '--', color = 'black', label = 'Randomized SVD')
    plt.xlabel(r'$k$',fontsize=20)
    plt.ylabel(r'$\sigma_k$',fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('svd_expt_compare.pdf')
    plt.show()

    # plot top kernel eigenfunctions
    for i in range(6):
        v = V[i,:]**2
        v = v/np.mean(v)
        plt.plot(stim_vals, v + 3*i)
        v = V[i,:]**2
        v = v/np.mean(v)
        if i==0:
            plt.plot(stim_vals, v + 3*i, '--', color = 'black', label = 'Random SVD')
        else:
            plt.plot(stim_vals, v + 3*i, '--', color = 'black')

    plt.xlabel(r'$\theta$', fontsize=20)
    plt.ylabel(r'$\phi_k(\theta)^2$', fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('eigenfunction_comparison_experimental_data.pdf')
    plt.show()


    # vary the embedding dimension
    Kvals = np.logspace(1, np.log10(0.95*num_stim), 10).astype('int')

    errs = []
    times = []
    num_repeat = 10
    for i,K in enumerate(Kvals):
        err_avg = 0
        times_avg = 0
        for t in range(num_repeat):
            Uk,Sk,Vk, Tk = randomized_SVD(resp_avg, K)
            R_rec = Uk @ np.diag(Sk) @ Vk
            rec_error = np.linalg.norm(R_rec - resp_avg, 'fro') / np.linalg.norm(resp_avg,'fro')
            err_avg += 1/num_repeat * rec_error
            times_avg += 1/num_repeat * Tk
        times += [err_avg]
        errs += [times_avg]
        print("reconstruction error: %0.4f" % rec_error)

    plt.loglog(Kvals, errs)
    plt.xlabel(r'$K$',fontsize=20)
    plt.ylabel(r'$E_R$',fontsize=20)
    plt.tight_layout()
    plt.savefig('reconstruction_vs_K_expt.pdf')
    plt.show()

    plt.loglog(Kvals, times)
    plt.xlabel(r'$K$',fontsize=20)
    plt.ylabel(r'SVD Time',fontsize=20)
    plt.tight_layout()
    plt.savefig('times_svd_expt.pdf')
    plt.show()
