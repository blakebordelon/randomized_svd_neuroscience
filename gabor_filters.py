import numpy as np
import scipy as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import math
import timeit

# sparse Gabor feature map feature map; sparsity depends on threshold
def gabor_features(N, P, sigma, thresh):
    theta_stim = np.linspace(-math.pi, math.pi, P)
    theta_pr = 2*math.pi * np.random.random_sample(N)
    th_diff = np.outer(theta_stim, np.ones(len(theta_pr))) - np.outer(np.ones(len(theta_stim)), theta_pr)
    cos_th = np.cos(th_diff)
    R = np.cosh(sigma*cos_th)
    R = R * np.exp(-sigma)
    R = (R > thresh) * (R-thresh)
    return 1/np.sqrt(N) * R

# the traditional randomized SVD algorithm
def randomized_SVD(R,K):
    N = R.shape[1]
    Z = np.random.standard_normal((N,K))
    start = timeit.default_timer()
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

# Hadamard transform; In practice we did not see an improvement.
def hadamard_SVD(R, K):
    start = timeit.default_timer()
    N = R.shape[1]
    #H = sp.linalg.hadamard(N)
    D = np.diag(np.sign(np.random.random_sample(N) - 0.5))
    rand_inds = np.random.choice(N, K)
    C = np.eye(N)[:,rand_inds]
    time_trans = timeit.default_timer()
    Z = fast_hadamard_transform(D @ R.T, N).T
    end_trans = timeit.default_timer()
    print("hadamard transform time %0.5f" % (end_trans-time_trans))
    Y = Z @ C
    #Y = R @ D @ H @ C
    Q,_ = np.linalg.qr(Y)
    B = Q.T @ R
    U,S,V = np.linalg.svd(B)
    U = Q @ U
    end = timeit.default_timer()
    T = end - start
    inds = np.argsort(S)[::-1]
    S = S[inds]
    U = U[:,inds]
    V = V[inds,:]
    return U, S, V, T


# compute alignment of orthogonal matrices
def compare_orthogonal_matrices(U,V):
    num = min(U.shape[1],V.shape[1])
    dots = np.zeros(num)
    for i in range(num):
        dots[i] = (np.dot(U[:,i], V[:,i]))**2
    return dots

# compute SVD on full matrix
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


# random projections
def random_projection_JL_SVD(R, K):
    N = R.shape[1]
    Rh = 1/np.sqrt(K) * R @  np.random.standard_normal((N,K))
    return standard_SVD(Rh)


# Sparse SVD solver
def sparse_svd(R,K):
    start = timeit.default_timer()
    U,S,V = sp.sparse.linalg.svds(R,K)
    end = timeit.default_timer()
    T = end - start
    inds = np.argsort(S)[::-1]
    S = S[inds]
    U = U[:,inds]
    V = V[inds,:]
    return U,S,V,T

# implementation of Hadamard transform
def fast_hadamard_transform(A,n):
    if n % 2 != 0 or A.shape[0] != n:
        print("error!")
        return
    if n == 2:
        H = np.array([[1,1],[1,-1]])
        return H @ A
    else:
        n2 = int(n/2)
        Z1 = fast_hadamard_transform(A[0:n2,:], n2)
        Z2 = fast_hadamard_transform(A[n2:n], n2)
        Z_tot = np.zeros((n,A.shape[1]))
        Z_tot[0:n2] = Z1 + Z2
        Z_tot[n2:n] = Z1 - Z2
        return Z_tot

    return


N = 2**(13)
P = 4096
thresh = 0.2
sigma = 6



# vary K, compute reconstruction error and time
Kvals = np.logspace(1,2,20).astype('int')
R = gabor_features(N,P,sigma,thresh)
R_lin = gabor_features(N,P,sigma, 0)
U,S,V,T = standard_SVD(R)
print("T total % 0.3f" % T)
num_repeat=3
errors = np.zeros(len(Kvals))
errors_lin = np.zeros(len(Kvals))
times = np.zeros(len(Kvals))
times_lin = np.zeros(len(Kvals))

for i,K in enumerate(Kvals):
    Ti = 0
    ei = 0
    Tlin = 0
    elin = 0
    for t in range(num_repeat):
        Ui,Si,Vi,Tt = randomized_SVD(R,K)
        print(Ui.shape)
        print(Si.shape)
        print(Vi.shape)
        ei += 1/num_repeat * np.linalg.norm( Ui @ np.diag(Si) @ Vi - R, 'fro')
        Ti += 1/num_repeat * Tt
        Ui,Si,Vi,Tt = randomized_SVD(R_lin,K)
        Tlin += 1/num_repeat *T
        elin += 1/num_repeat * np.linalg.norm(Ui @ np.diag(Si) @ Vi - R_lin,'fro')


    errors[i] = ei
    errors_lin[i] = elin
    times[i] = Ti
    times_lin[i] = Tlin

plt.loglog(Kvals, errors_lin, label = 'linear neurons')
plt.loglog(Kvals, errors, label = 'rectified neurons')
#plt.loglog(Kvals, errors_jl)
plt.xlabel(r'$K$',fontsize=20)
plt.ylabel(r'$||R - \hat{R}||_F$',fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('reconstruction_vs_K_nonlinear.pdf')
plt.show()


plt.loglog(Kvals, times_lin, label = 'linear neurons')
plt.loglog(Kvals, times, label = 'rectified neurons')
#plt.loglog(Kvals, errors_jl)
plt.xlabel(r'$K$',fontsize=20)
plt.ylabel(r'SVD time',fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('times_vs_K_nonlinear.pdf')
plt.show()


# vary the sparsity level
thresh_vals = np.logspace(0,0.45,10).astype('int')
K = 100
times = []
errors = []
errors_lin = []
times_lin = []
for i,thresh in enumerate(thresh_vals):
    R = gabor_features(N,P,sigma,thresh)
    #R_lin = gabor_features(N,P,sigma, 0)
    Ui,Si,Vi,Ti = randomized_SVD(R,K)
    print(Ui.shape)
    print(Si.shape)
    print(Vi.shape)
    ei = np.linalg.norm( Ui @ np.diag(Si) @ Vi - R, 'fro')
    times += [Ti]
    errors+=[ei]

    Ui,Si,Vi,Ti = randomized_SVD(R_lin,K)
    errors_lin += [np.linalg.norm(Ui @ np.diag(Si) @ Vi - R_lin,'fro')]
    times_lin += [Ti]
    #Ui,Si,Vi,Ti = random_projection_JL_SVD(R,K)
    #ej = np.linalg.norm(Ui @ np.diag(Si) @ Vi - R, 'fro')
    #errors_jl += [ej]



R = gabor_features(N, P, sigma, thresh).T
theta_stim = np.linspace(-math.pi, math.pi, P)


# time for standard SVD
U,S,V,T = standard_SVD(R)
print("Standard SVD time: %0.4f" % T)



K = 100



# compare random SVD reconstruction
Uh,Sh,Vh,Th = randomized_SVD(R,K)
Uhad, Shad, Vhad, Thad = hadamard_SVD(R,K)

plt.loglog(S, label = 'Full SVD Spectrum')
plt.loglog(Sh, '--', color = 'black', label = 'Random Projected Spectrum')
plt.xlabel(r'$k$', fontsize = 20)
plt.ylabel(r'$\sigma_k$', fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('gabor_full_vs_random_svd_10.pdf')
plt.show()


R_rand = Uh @ np.diag(Sh) @ Vh
R_had = Uhad @ np.diag(Shad) @ Vhad
E_rand = np.linalg.norm(R_rand-R,'fro')/np.linalg.norm(R,'fro')
E_had = np.linalg.norm(R_had - R,'fro') / np.linalg.norm(R,'fro')

print("Time Standard: %0.5f" % T)
print("time hadamard: %0.5f" % Thad)
print("time gauss: %0.5f" % Th)
print("E_rand: %0.5f" % E_rand)
print("E_had: %0.5f" % E_had)



K = 10
plt.loglog(S, label = 'Full SVD Spectrum')
plt.loglog(Sh, '--', color = 'black', label = 'Random Projected Spectrum')
plt.xlabel(r'$k$', fontsize = 20)
plt.ylabel(r'$\sigma_k$', fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('gabor_full_vs_random_svd_10.pdf')
plt.show()


Uh,Sh,Vh,Th = randomized_SVD(R,K)
Uhad, Shad, Vhad, Thad = hadamard_SVD(R,K)

R_rand = Uh @ np.diag(Sh) @ Vh
R_had = Uhad @ np.diag(Shad) @ Vhad
E_rand = np.linalg.norm(R_rand-R,'fro')/np.linalg.norm(R,'fro')
E_had = np.linalg.norm(R_had - R,'fro') / np.linalg.norm(R,'fro')
print("K = 10")
print("Time Standard: %0.5f" % T)
print("time hadamard: %0.5f" % Thad)
print("time gauss: %0.5f" % Th)
print("E_rand: %0.5f" % E_rand)
print("E_had: %0.5f" % E_had)



# compare alignment of top eigenvectors
dots = compare_orthogonal_matrices(Uh,U)
dotv = compare_orthogonal_matrices(Vh.T,V.T)

plt.loglog(dots, label = 'Kernel Eigenfunctions')
plt.loglog(dotv, '--', color = 'black', label = 'Neural PCs')
plt.xlabel(r'$k$',fontsize=20)
plt.ylabel(r'$(u_k \cdot \hat{u}_k)^2$', fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('alignment_10.pdf')
plt.show()
print(dots)
print(dotv)
print("time for QR version")
print(Th)


# plot top eigenvectors
for i in range(6):
    v = U[:,i]**2
    v = v/np.mean(v)

    plt.plot(np.linspace(-math.pi, math.pi, P), v + 3*i)
    v = Uh[:,i]**2
    v = v/np.mean(v)
    if i==0:
        plt.plot(np.linspace(-math.pi, math.pi, P), v + 3*i, '--', color = 'black', label = 'Random SVD')
    else:
        plt.plot(np.linspace(-math.pi, math.pi, P), v + 3*i, '--', color = 'black')

plt.xlabel(r'$\theta$', fontsize=20)
plt.ylabel(r'$\phi_k(\theta)^2$', fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('eigenfunction_comparison_10.pdf')
plt.show()


Kvals = np.logspace(np.log10(0.1*P), np.log10(10*P), 10).astype('int')
times = []
for i,K in enumerate(Kvals):
    Uh,Sh,Vh,Th = random_projection_JL_SVD(R, K)
    times += [Th]

plt.loglog(Kvals / P, times, label = 'experimental time')
plt.loglog(Kvals/P , (Kvals / P)**2, '--', color = 'black', label = 'T = K')
plt.legend()
plt.xlabel(r'$K/P$', fontsize=20)
plt.ylabel(r'SVD Time', fontsize=20)
plt.tight_layout()
plt.savefig('times_random_projection.pdf')
plt.show()



#thresh_vals = [0.0,0.1,0.2, 0.5, 1]
thresh_vals = np.linspace(0.1,0.5,20)
num = []
sparse_time = []
K = 100
for i, t in enumerate(thresh_vals):
    R = gabor_features(N,P,sigma,t)
    Rs = sp.sparse.csc_matrix(R)
    print("converted")
    all_T = np.zeros(10)
    for j in range(10):
        U,S,V,T = sparse_svd(Rs,K)
        all_T[j] = T
    sparse_time += [np.mean(all_T)]

#plt.plot(thresh_vals, num)
#plt.xlabel(r'$a$',fontsize=20)
#plt.ylabel(r'$f$', fontsize=20)
#plt.tight_layout()
#plt.savefig('coding_level_vs_a.pdf')
#plt.show()


plt.plot(thresh_vals, sparse_time)
plt.xlabel(r'$a$',fontsize=20)
plt.ylabel(r'Sparse SVD time', fontsize=20)
plt.tight_layout()
plt.savefig('time_vs_threshold.pdf')
plt.show()
