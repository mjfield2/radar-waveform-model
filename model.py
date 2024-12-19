import numpy as np
from scipy.interpolate import interp2d, RectBivariateSpline, RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import matlib
from numpy import ix_
from tqdm.auto import tqdm
import multiprocessing as mp


def finddx(epmax, mumax, srcpulse, t, thres=0.02):
    # convert relative permittivity and permeability to true values
    mu0 = 1.2566370614e-6
    ep0 = 8.8541878176e-12
    epmax *= ep0
    mumax *= mu0

    # compute amplitude spectrum of source pulse and corresponding frequency vector
    n = 2 ** np.ceil(np.log2(len(srcpulse)))
    W = np.abs(np.fft.fftshift(np.fft.fft(srcpulse, int(n))))
    W /= np.max(W)
    fn = 0.5 / (t[1] - t[0])
    df = 2 * fn / n
    f = np.arange(-fn, fn, df)[int(n / 2):]
    W = W[int(n / 2):]

    # determine the maximum allowable spatial discretization
    # (5 grid points per minimum wavelength are needed to avoid dispersion)
    fmax = f[np.max(np.where(W >= thres))]
    wlmin = 1 / (fmax * np.sqrt(epmax * mumax))
    dxmax = wlmin / 5

    return dxmax, wlmin, fmax


def finddt(epmin, mumin, dx, dz):
    # Physical constants
    mu0 = 1.2566370614e-6
    ep0 = 8.8541878176e-12
    
    # Convert relative permittivity and permeability to true values
    epmin = epmin * ep0
    mumin = mumin * mu0
    
    # Determine maximum allowable time step for numerical stability
    dtmax = 6 / 7 * np.sqrt(epmin * mumin / (1 / dx**2 + 1 / dz**2))
    
    return dtmax

def blackharrispulse(fr, t):
    # Compute the Blackman-Harris window as specified in Chen et al. (1997)
    a = [0.35322222, -0.488, 0.145, -0.010222222]
    T = 1.14 / fr
    window = np.zeros_like(t)
    for n in range(4):
        window += a[n] * np.cos(2 * n * np.pi * t / T)
    window[t >= T] = 0

    # For the pulse, approximate the window's derivative and normalize
    p = np.concatenate(([0], np.diff(window)))  # Approximate derivative
    p /= np.max(np.abs(p))  # Normalize

    return p

def padgrid(A, x, z, n):
    # Determine the new position vectors
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    x2 = np.arange(x[0] - n * dx, x[-1] + (n) * dx+1e-10, dx)
    z2 = np.arange(z[0] - n * dz, z[-1] + (n) * dz+1e-10, dz)

    # Pad the grid
    A2 = np.hstack((np.tile(A[:, [0]], (1, n)), A, np.tile(A[:, [-1]], (1, n))))
    A2 = np.vstack((np.tile(A2[[0], :], (n, 1)), A2, np.tile(A2[[-1], :], (n, 1))))

    A2 = np.pad(A, ((n, n), (n, n)), mode='edge')

    return A2, x2, z2


def gridinterp(A, x, z, x2, z2, method='nearest'):
    # Transpose for interpolation
    #A = A.T

    # Perform the interpolation
    #interp_func = interp2d(x, z, A, kind=method)
    xx2, zz2 = np.meshgrid(x2, z2, indexing='ij')
    interp_func = RegularGridInterpolator((x, z), A, method=method, bounds_error=False, fill_value=np.mean(A))
    A2 = interp_func(np.array([xx2.flatten(), zz2.flatten()]).T)

    # Transpose back for output
    # A2 = A2.T

    return A2.reshape(xx2.shape)





def TM_model2d(ep, mu, sig, xprop, zprop, srcloc, recloc, srcpulse, t, npml, outstep=1, plotopt=[1, 50, 0.05]):
    if len(plotopt) == 2:
        plotopt.append(0.05)

    if ep.shape != mu.shape or ep.shape != sig.shape:
        print('ep, mu, and sig matrices must be the same size')
        return
    if (len(xprop), len(zprop)) != ep.shape:
        print('xprop and zprop are inconsistent with ep, mu, and sig')
        return
    if ep.shape[0] % 2 != 1 or ep.shape[1] % 2 != 1:
        print('ep, mu, and sig must have an odd # of rows and columns')
        return
    if srcloc.shape[1] != 2 or recloc.shape[1] != 2:
        print('srcloc and recloc matrices must have 2 columns')
        return
    if (np.max(srcloc[:, 0]) > np.max(xprop) or np.min(srcloc[:, 0]) < np.min(xprop) or
            np.max(srcloc[:, 1]) > np.max(zprop) or np.min(srcloc[:, 1]) < np.min(zprop)):
        print('source vector out of range of modeling grid')
        return
    if (np.max(recloc[:, 0]) > np.max(xprop) or np.min(recloc[:, 0]) < np.min(xprop) or
            np.max(recloc[:, 1]) > np.max(zprop) or np.min(recloc[:, 1]) < np.min(zprop)):
        print('receiver vector out of range of modeling grid')
        return
    if len(srcpulse) != len(t):
        print('srcpulse and t vectors must have same # of points')
        return
    if npml >= ep.shape[0] / 2 or npml >= ep.shape[1] / 2:
        print('too many PML boundary layers for grid')
        return


    ep0 = 8.8541878176e-12
    mu0 = 1.2566370614e-6
    ep = ep * ep0
    mu = mu * mu0
    nx = (len(xprop)+1) // 2
    nz = (len(zprop)+1) // 2
    dx = 2 * (xprop[1] - xprop[0])
    dz = 2 * (zprop[1] - zprop[0])


    xHx = np.arange(xprop[1], xprop[-2], dx)
    zHx = np.arange(zprop[0], zprop[-1], dz)
    xHz = np.arange(xprop[0], xprop[-1], dx)
    zHz = np.arange(zprop[1], zprop[-2], dz)
    xEy = xHx
    zEy = zHz

    nsrc = srcloc.shape[0]
    nrec = recloc.shape[0]
    srci = np.zeros(nsrc,dtype=np.int32)
    srcj = np.zeros(nsrc,dtype=np.int32)
    reci = np.zeros(nrec,dtype=np.int32)
    recj = np.zeros(nrec,dtype=np.int32)
    srcx = np.zeros(nsrc)
    srcz = np.zeros(nsrc)
    recx = np.zeros(nrec)
    recz = np.zeros(nrec)

    dt = t[1] - t[0]
    numit = len(t)
    gather = np.zeros((int((numit - 1) / outstep), nrec, nsrc))
    tout = np.zeros(int((numit - 1) / outstep))



    for s in range(nsrc):
        print(s)
        print(srcloc[s,0])
        print("xEy - srcloc[s, 0]")
        print(np.min(np.abs(xEy - srcloc[s, 0])))
        srci[s] = np.argmin(np.abs(xEy - srcloc[s, 0]))
        srcj[s] = np.argmin(np.abs(zEy - srcloc[s, 1]))
        xEy[s]

        srcx[s] = xEy[srci[s]]
        srcz[s] = zEy[srcj[s]]

    print("print srci")

    print(srci)


    for r in range(nrec):
        reci[r] = np.argmin(np.abs(xEy - recloc[r, 0]))
        recj[r] = np.argmin(np.abs(zEy - recloc[r, 1]))
        recx[r] = xEy[reci[r]]
        recz[r] = zEy[recj[r]]

    m = 4
    Kxmax = 5
    Kzmax = 5
    sigxmax = (m + 1) / (150 * np.pi * np.sqrt(ep / ep0) * dx)
    sigzmax = (m + 1) / (150 * np.pi * np.sqrt(ep / ep0) * dz)
    alpha = 0

    kpmlLout = 0
    kpmlLin = 2 * npml + 1
    kpmlRin = len(xprop) - (2 * npml + 2)
    kpmlRout = len(xprop) -1
    lpmlTout = 0
    lpmlTin = 2 * npml + 1
    lpmlBin = len(zprop) - (2 * npml + 2)
    lpmlBout = len(zprop) -1


    xdel = np.zeros((len(xprop),len(zprop)))
    k = np.arange(kpmlLout, kpmlLin)

    xdel[k, :] = matlib.repmat((kpmlLin - k) / (2 * npml),len(zprop),1).transpose()
    k = np.arange(kpmlRin, kpmlRout + 1)
    xdel[k, :] = matlib.repmat((k - kpmlRin) / (2 * npml),len(zprop),1).transpose()
    zdel = np.zeros((len(xprop),len(zprop)))
    l = np.arange(lpmlTout, lpmlTin + 1)
    zdel[:,l] = matlib.repmat((lpmlTin - l) / (2 * npml),len(xprop),1)
    l = np.arange(lpmlBin, lpmlBout + 1)
    zdel[:,l] = matlib.repmat((l - lpmlBin) / (2 * npml),len(xprop),1)


    sigx = sigxmax * np.power(xdel, m)
    sigz = sigzmax * np.power(zdel, m)
    Kx = 1 + (Kxmax - 1) * np.power(xdel, m)
    Kz = 1 + (Kzmax - 1) * np.power(zdel, m)


    Ca = (1 - dt * sig / (2 * ep)) / (1 + dt * sig / (2 * ep))
    Cbx = (dt / ep) / ((1 + dt * sig / (2 * ep)) * 24 * dx * Kx)
    Cbz = (dt / ep) / ((1 + dt * sig / (2 * ep)) * 24 * dz * Kz)
    Cc = (dt / ep) / (1 + dt * sig / (2 * ep))
    Dbx = (dt / (mu * Kx * 24 * dx))
    Dbz = (dt / (mu * Kz * 24 * dz))
    Dc = dt / mu
    Bx = np.exp(-((sigx / Kx + alpha) * (dt / ep0)))
    Bz = np.exp(-((sigz / Kz + alpha) * (dt / ep0)))
    Ax = (sigx / (sigx * Kx + Kx**2 * alpha + 1e-20) * (Bx - 1)) / (24 * dx)
    Az = (sigz / (sigz * Kz + Kz**2 * alpha + 1e-20) * (Bz - 1)) / (24 * dz)


    for s in range(nsrc):
        iii = srci[s]
        jjj = srcj[s]
        print("index:")
        print(iii)
        print(jjj)
        print("source position:")
        print(srcx[s])
        print(srcz[s])
        # zero all field matrices
        Ey = np.zeros((nx - 1, nz - 1))  # Ey component of electric field
        Hx = np.zeros((nx - 1, nz))  # Hx component of magnetic field
        Hz = np.zeros((nx, nz - 1))  # Hz component of magnetic field
        Eydiffx = np.zeros((nx, nz - 1))  # difference for dEy/dx
        Eydiffz = np.zeros((nx - 1, nz))  # difference for dEy/dz
        Hxdiffz = np.zeros((nx - 1, nz - 1))  # difference for dHx/dz
        Hzdiffx = np.zeros((nx - 1, nz - 1))  # difference for dHz/dx
        PEyx = np.zeros((nx - 1, nz - 1))  # psi_Eyx (for PML)
        PEyz = np.zeros((nx - 1, nz - 1))  # psi_Eyz (for PML)
        PHx = np.zeros((nx - 1, nz))  # psi_Hx (for PML)
        PHz = np.zeros((nx, nz - 1))  # psi_Hz (for PML)

        # time stepping loop
        for it in range(numit):
            # update Hx component...


            # determine indices for entire, PML, and interior regions in Hx and property grids

            i = np.arange(1, nx - 2)  # indices for all components in Hx matrix to update
            j = np.arange(2, nz - 2)

            k = 2 * i + 1  # corresponding indices in property grids
            l = 2 * j
            kp = k[(k <= kpmlLin) | (k >= kpmlRin)]  # corresponding property indices in PML region
            lp = l[(l <= lpmlTin) | (l >= lpmlBin)]
            ki = k[(k > kpmlLin) & (k < kpmlRin)]  # corresponding property indices in interior (non-PML) region
            li = l[(l > lpmlTin) & (l < lpmlBin)]
            ip = (kp - 1) // 2
            jp = lp // 2  # Hx indices in PML region
            ii = (ki - 1) // 2
            ji = li // 2  # Hx indices in interior (non-PML) region



            # update to be applied to the whole Hx grid

            
            Eydiffz[ix_(i,j)] = -Ey[ix_(i,j+1)] + 27 * Ey[ix_(i,j)] - 27 * Ey[ix_(i,j - 1)] + Ey[ix_(i,j - 2)]
            Hx[ix_(i,j)] = Hx[ix_(i,j)] - Dbz[ix_(k,l)] * Eydiffz[ix_(i,j)]



            # update to be applied only to the PML region
            PHx[ix_(ip,j)] = Bz[ix_(kp, l)] * PHx[ix_(ip, j)] + Az[ix_(kp, l)] * Eydiffz[ix_(ip, j)]
            PHx[ix_(ii,jp)] = Bz[ix_(ki, lp)] * PHx[ix_(ii, jp)] + Az[ix_(ki, lp)] * Eydiffz[ix_(ii, jp)]
            Hx[ix_(ip,j)] = Hx[ix_(ip,j)] - Dc[ix_(kp, l)] * PHx[ix_(ip, j)]
            Hx[ix_(ii,jp)] =  Hx[ix_(ii,jp)] - Dc[ix_(ki, lp)] * PHx[ix_(ii, jp)]

            # update Hz component...

            # determine indices for entire, PML, and interior regions in Hz and property grids
            i = np.arange(2, nx - 2)  # indices for all components in Hz matrix to update
            j = np.arange(1, nz - 2)
            k = 2 * i   # corresponding indices in property grids
            l = 2 * j + 1
            kp = k[(k <= kpmlLin) | (k >= kpmlRin)]  # corresponding property indices in PML region
            lp = l[(l <= lpmlTin) | (l >= lpmlBin)]
            ki = k[(k > kpmlLin) & (k < kpmlRin)]  # corresponding property indices in interior (non-PML) region
            li = l[(l > lpmlTin) & (l < lpmlBin)]
            ip = (kp ) // 2
            jp = (lp -1 )// 2  # Hz indices in PML region
            ii = (ki ) // 2
            ji = (li -1) // 2  # Hz indices in interior (non-PML) region



            # update to be applied to the whole Hz grid
            Eydiffx[ix_(i, j)] = -Ey[ix_(i+1, j)] + 27 * Ey[ix_(i, j)] - 27 * Ey[ix_(i - 1, j)] + Ey[ix_(i - 2, j)]
            Hz[ix_(i, j)] = Hz[ix_(i, j)] + Dbx[ix_(k, l)] * Eydiffx[ix_(i, j)]




            # update to be applied only to the PML region
            PHz[ix_(ip, j)] = Bx[ix_(kp, l)] * PHz[ix_(ip, j)] + Ax[ix_(kp, l)] * Eydiffx[ix_(ip, j)]
            PHz[ix_(ii, jp)] = Bx[ix_(ki, lp)] * PHz[ix_(ii, jp)] + Ax[ix_(ki, lp)] * Eydiffx[ix_(ii, jp)]
            Hz[ix_(ip, j)] = Hz[ix_(ip, j)] + Dc[kp][:, l] * PHz[ix_(ip, j)]
            Hz[ix_(ii, jp)] = Hz[ix_(ii, jp)] + Dc[ix_(ki, lp)] * PHz[ix_(ii, jp)]



            # update Ey component...

            # determine indices for entire, PML, and interior regions in Ey and property grids
            i = np.arange(1, nx - 2)  # indices for all components in Ey matrix to update
            j = np.arange(1, nz - 2)
            k = 2 * i + 1 # corresponding indices in property grids
            l = 2 * j + 1
            kp = k[(k <= kpmlLin) | (k >= kpmlRin)]  # corresponding property indices in PML region
            lp = l[(l <= lpmlTin) | (l >= lpmlBin)]
            ki = k[(k > kpmlLin) & (k < kpmlRin)]  # corresponding property indices in interior (non-PML) region
            li = l[(l > lpmlTin) & (l < lpmlBin)]
            ip = kp // 2
            jp = lp // 2  # Ey indices in PML region
            ii = ki // 2
            ji = li // 2  # Ey indices in interior (non-PML) region

            # update to be applied to the whole Ey grid
            Hxdiffz[ix_(i, j)] = -Hx[ix_(i, j + 2)] + 27 * Hx[ix_(i, j + 1)] - 27 * Hx[ix_(i, j)] + Hx[ix_(i, j - 1)]
            Hzdiffx[ix_(i, j)] = -Hz[ix_(i+ 2, j)] + 27 * Hz[ix_(i + 1, j)] - 27 * Hz[ix_(i, j)] + Hz[ix_(i - 1, j)]
            Ey[ix_(i, j)] = Ca[ix_(k, l)] * Ey[ix_(i, j)] + Cbx[ix_(k, l)] * Hzdiffx[ix_(i, j)] - Cbz[ix_(k, l)] * Hxdiffz[ix_(i, j)]


            # update to be applied only to the PML region
            PEyx[ix_(ip, j)] = Bx[ix_(kp, l)] * PEyx[ix_(ip, j)] + Ax[ix_(kp, l)] * Hzdiffx[ix_(ip, j)]
            PEyx[ix_(ii, jp)] = Bx[ix_(ki, lp)] * PEyx[ix_(ii, jp)] + Ax[ix_(ki, lp)] * Hzdiffx[ix_(ii, jp)]
            PEyz[ix_(ip, j)] = Bz[ix_(kp, l)] * PEyz[ix_(ip, j)] + Az[ix_(kp, l)] * Hxdiffz[ix_(ip, j)]
            PEyz[ix_(ii, jp)] = Bz[ix_(ki, lp)] * PEyz[ix_(ii, jp)] + Az[ix_(ki, lp)] * Hxdiffz[ix_(ii, jp)]
            Ey[ix_(ip,j)] = Ey[ix_(ip, j)]+ Cc[ix_(kp, l)] * (PEyx[ix_(ip, j)] - PEyz[ix_(ip, j)])
            Ey[ix_(ii, jp)] = Ey[ix_(ii, jp)]  + Cc[ix_(ki, lp)] * (PEyx[ix_(ii, jp)] - PEyz[ix_(ii, jp)])

            # add source pulse to Ey at source location
            # (emulates infinitesimal Ey directed line source with current = srcpulse)

            iii = srci[s]
            jjj = srcj[s]
            
            Ey[iii,jjj] = Ey[iii,jjj] + srcpulse[it]


            # plot the Ey wavefield if necessary
            if plotopt[0] == 1:
                if (it - 1) % plotopt[1] == 0:
                    print(f"Source {s}/{nsrc}, Iteration {it}/{numit}, t = {t[it-1]*1e9} ns")
                    plt.figure(figsize=(8,3))
                    img=plt.imshow(Ey.transpose(),vmin=-plotopt[2], vmax=plotopt[2],cmap=cm.coolwarm,extent=(xEy[0], xEy[-1],zEy[-1], zEy[0]))
                    plt.gca().set_aspect('equal', adjustable='box')
                    #plt.title(f"Source {s}/{nsrc}, Iteration {it}/{numit}, Ey wavefield at t = {t[it-1]*1e9} ns")
                    plt.xlabel('Position (m)')
                    plt.ylabel('Depth (m)')
                    plt.colorbar(img,label='Amplitude')
                    plt.tight_layout()
                    plt.pause(.001)
                    #plt.close()
                    #fig.savefig(f"figures/source_{s}-{nsrc}_iteratate{it}-{numit}.png",dpi=200)


            # record the results in gather matrix if necessary

            if (it - 1) % outstep == 0:
                tout[(it - 1) // outstep] = t[it - 1]
                for r in range(nrec):
                    gather[(it - 1) // outstep, r, s - 1] = Ey[reci[r], recj[r]]
    return gather, tout, srcx, srcz, recx, recz

def TM_model2d_custom(ep, mu, sig, xprop, zprop, srcloc, recloc, srcpulse, t, npml, outstep=1, plotopt=[1, 50, 0.05], quiet=False, save_wav=False):
    if len(plotopt) == 2:
        plotopt.append(0.05)

    if ep.shape != mu.shape or ep.shape != sig.shape:
        print('ep, mu, and sig matrices must be the same size')
        return
    if (len(xprop), len(zprop)) != ep.shape:
        print('xprop and zprop are inconsistent with ep, mu, and sig')
        return
    if ep.shape[0] % 2 != 1 or ep.shape[1] % 2 != 1:
        print('ep, mu, and sig must have an odd # of rows and columns')
        return
    if srcloc.shape[1] != 2 or recloc.shape[1] != 2:
        print('srcloc and recloc matrices must have 2 columns')
        return
    if (np.max(srcloc[:, 0]) > np.max(xprop) or np.min(srcloc[:, 0]) < np.min(xprop) or
            np.max(srcloc[:, 1]) > np.max(zprop) or np.min(srcloc[:, 1]) < np.min(zprop)):
        print('source vector out of range of modeling grid')
        return
    if (np.max(recloc[:, 0]) > np.max(xprop) or np.min(recloc[:, 0]) < np.min(xprop) or
            np.max(recloc[:, 1]) > np.max(zprop) or np.min(recloc[:, 1]) < np.min(zprop)):
        print('receiver vector out of range of modeling grid')
        return
    if len(srcpulse) != len(t):
        print('srcpulse and t vectors must have same # of points')
        return
    if npml >= ep.shape[0] / 2 or npml >= ep.shape[1] / 2:
        print('too many PML boundary layers for grid')
        return


    ep0 = 8.8541878176e-12
    mu0 = 1.2566370614e-6
    ep = ep * ep0
    mu = mu * mu0
    nx = (len(xprop)+1) // 2
    nz = (len(zprop)+1) // 2
    dx = 2 * (xprop[1] - xprop[0])
    dz = 2 * (zprop[1] - zprop[0])


    xHx = np.arange(xprop[1], xprop[-2], dx)
    zHx = np.arange(zprop[0], zprop[-1], dz)
    xHz = np.arange(xprop[0], xprop[-1], dx)
    zHz = np.arange(zprop[1], zprop[-2], dz)
    xEy = xHx
    zEy = zHz

    nsrc = srcloc.shape[0]
    nrec = recloc.shape[0]
    srci = np.zeros(nsrc,dtype=np.int32)
    srcj = np.zeros(nsrc,dtype=np.int32)
    reci = np.zeros(nrec,dtype=np.int32)
    recj = np.zeros(nrec,dtype=np.int32)
    srcx = np.zeros(nsrc)
    srcz = np.zeros(nsrc)
    recx = np.zeros(nrec)
    recz = np.zeros(nrec)

    dt = t[1] - t[0]
    numit = len(t)
    gather = np.zeros((int((numit - 1) / outstep), nrec, nsrc))
    tout = np.zeros(int((numit - 1) / outstep))

    # make cache for wavefield
    if sav_wav==True:
        n_out = 0
        for it in range(numit):
            if (it - 1) % plotopt[1] == 0:
                n_out += 1
        wav_cache = np.zeros((nsrc, n_out, nz-1, nx-1))

    for s in range(nsrc):
        srci[s] = np.argmin(np.abs(xEy - srcloc[s, 0]))
        srcj[s] = np.argmin(np.abs(zEy - srcloc[s, 1]))
        xEy[s]

        srcx[s] = xEy[srci[s]]
        srcz[s] = zEy[srcj[s]]

    for r in range(nrec):
        reci[r] = np.argmin(np.abs(xEy - recloc[r, 0]))
        recj[r] = np.argmin(np.abs(zEy - recloc[r, 1]))
        recx[r] = xEy[reci[r]]
        recz[r] = zEy[recj[r]]

    m = 4
    Kxmax = 5
    Kzmax = 5
    sigxmax = (m + 1) / (150 * np.pi * np.sqrt(ep / ep0) * dx)
    sigzmax = (m + 1) / (150 * np.pi * np.sqrt(ep / ep0) * dz)
    alpha = 0

    kpmlLout = 0
    kpmlLin = 2 * npml + 1
    kpmlRin = len(xprop) - (2 * npml + 2)
    kpmlRout = len(xprop) -1
    lpmlTout = 0
    lpmlTin = 2 * npml + 1
    lpmlBin = len(zprop) - (2 * npml + 2)
    lpmlBout = len(zprop) -1


    xdel = np.zeros((len(xprop),len(zprop)))
    k = np.arange(kpmlLout, kpmlLin)

    xdel[k, :] = matlib.repmat((kpmlLin - k) / (2 * npml),len(zprop),1).transpose()
    k = np.arange(kpmlRin, kpmlRout + 1)
    xdel[k, :] = matlib.repmat((k - kpmlRin) / (2 * npml),len(zprop),1).transpose()
    zdel = np.zeros((len(xprop),len(zprop)))
    l = np.arange(lpmlTout, lpmlTin + 1)
    zdel[:,l] = matlib.repmat((lpmlTin - l) / (2 * npml),len(xprop),1)
    l = np.arange(lpmlBin, lpmlBout + 1)
    zdel[:,l] = matlib.repmat((l - lpmlBin) / (2 * npml),len(xprop),1)


    sigx = sigxmax * np.power(xdel, m)
    sigz = sigzmax * np.power(zdel, m)
    Kx = 1 + (Kxmax - 1) * np.power(xdel, m)
    Kz = 1 + (Kzmax - 1) * np.power(zdel, m)


    Ca = (1 - dt * sig / (2 * ep)) / (1 + dt * sig / (2 * ep))
    Cbx = (dt / ep) / ((1 + dt * sig / (2 * ep)) * 24 * dx * Kx)
    Cbz = (dt / ep) / ((1 + dt * sig / (2 * ep)) * 24 * dz * Kz)
    Cc = (dt / ep) / (1 + dt * sig / (2 * ep))
    Dbx = (dt / (mu * Kx * 24 * dx))
    Dbz = (dt / (mu * Kz * 24 * dz))
    Dc = dt / mu
    Bx = np.exp(-((sigx / Kx + alpha) * (dt / ep0)))
    Bz = np.exp(-((sigz / Kz + alpha) * (dt / ep0)))
    Ax = (sigx / (sigx * Kx + Kx**2 * alpha + 1e-20) * (Bx - 1)) / (24 * dx)
    Az = (sigz / (sigz * Kz + Kz**2 * alpha + 1e-20) * (Bz - 1)) / (24 * dz)

    for s in range(nsrc):
        iii = srci[s]
        jjj = srcj[s]
        #print(f'index: {(iii, jjj)}')
        print(f'source #: {s+1}')
        print(f'source position: x = {srcx[s]:.1f}, z = {srcz[s]:.1f}')
        # zero all field matrices
        Ey = np.zeros((nx - 1, nz - 1))  # Ey component of electric field
        Hx = np.zeros((nx - 1, nz))  # Hx component of magnetic field
        Hz = np.zeros((nx, nz - 1))  # Hz component of magnetic field
        Eydiffx = np.zeros((nx, nz - 1))  # difference for dEy/dx
        Eydiffz = np.zeros((nx - 1, nz))  # difference for dEy/dz
        Hxdiffz = np.zeros((nx - 1, nz - 1))  # difference for dHx/dz
        Hzdiffx = np.zeros((nx - 1, nz - 1))  # difference for dHz/dx
        PEyx = np.zeros((nx - 1, nz - 1))  # psi_Eyx (for PML)
        PEyz = np.zeros((nx - 1, nz - 1))  # psi_Eyz (for PML)
        PHx = np.zeros((nx - 1, nz))  # psi_Hx (for PML)
        PHz = np.zeros((nx, nz - 1))  # psi_Hz (for PML)

        # time stepping loop
        out_counter = 0
        pbar = tqdm(range(numit), position=0, leave=True, disable=quiet)
        for it in pbar:
            ### update Hx component...
            # determine indices for entire, PML, and interior regions in Hx and property grids

            i = np.arange(1, nx - 2)  # indices for all components in Hx matrix to update
            j = np.arange(2, nz - 2)

            k = 2 * i + 1  # corresponding indices in property grids
            l = 2 * j
            kp = k[(k <= kpmlLin) | (k >= kpmlRin)]  # corresponding property indices in PML region
            lp = l[(l <= lpmlTin) | (l >= lpmlBin)]
            ki = k[(k > kpmlLin) & (k < kpmlRin)]  # corresponding property indices in interior (non-PML) region
            li = l[(l > lpmlTin) & (l < lpmlBin)]
            ip = (kp - 1) // 2
            jp = lp // 2  # Hx indices in PML region
            ii = (ki - 1) // 2
            ji = li // 2  # Hx indices in interior (non-PML) region

            # update to be applied to the whole Hx grid
            
            Eydiffz[ix_(i,j)] = -Ey[ix_(i,j+1)] + 27 * Ey[ix_(i,j)] - 27 * Ey[ix_(i,j - 1)] + Ey[ix_(i,j - 2)]
            Hx[ix_(i,j)] = Hx[ix_(i,j)] - Dbz[ix_(k,l)] * Eydiffz[ix_(i,j)]

            # update to be applied only to the PML region
            PHx[ix_(ip,j)] = Bz[ix_(kp, l)] * PHx[ix_(ip, j)] + Az[ix_(kp, l)] * Eydiffz[ix_(ip, j)]
            PHx[ix_(ii,jp)] = Bz[ix_(ki, lp)] * PHx[ix_(ii, jp)] + Az[ix_(ki, lp)] * Eydiffz[ix_(ii, jp)]
            Hx[ix_(ip,j)] = Hx[ix_(ip,j)] - Dc[ix_(kp, l)] * PHx[ix_(ip, j)]
            Hx[ix_(ii,jp)] =  Hx[ix_(ii,jp)] - Dc[ix_(ki, lp)] * PHx[ix_(ii, jp)]

            ### update Hz component...
            # determine indices for entire, PML, and interior regions in Hz and property grids
            i = np.arange(2, nx - 2)  # indices for all components in Hz matrix to update
            j = np.arange(1, nz - 2)
            k = 2 * i   # corresponding indices in property grids
            l = 2 * j + 1
            kp = k[(k <= kpmlLin) | (k >= kpmlRin)]  # corresponding property indices in PML region
            lp = l[(l <= lpmlTin) | (l >= lpmlBin)]
            ki = k[(k > kpmlLin) & (k < kpmlRin)]  # corresponding property indices in interior (non-PML) region
            li = l[(l > lpmlTin) & (l < lpmlBin)]
            ip = (kp ) // 2
            jp = (lp -1 )// 2  # Hz indices in PML region
            ii = (ki ) // 2
            ji = (li -1) // 2  # Hz indices in interior (non-PML) region

            # update to be applied to the whole Hz grid
            Eydiffx[ix_(i, j)] = -Ey[ix_(i+1, j)] + 27 * Ey[ix_(i, j)] - 27 * Ey[ix_(i - 1, j)] + Ey[ix_(i - 2, j)]
            Hz[ix_(i, j)] = Hz[ix_(i, j)] + Dbx[ix_(k, l)] * Eydiffx[ix_(i, j)]

            # update to be applied only to the PML region
            PHz[ix_(ip, j)] = Bx[ix_(kp, l)] * PHz[ix_(ip, j)] + Ax[ix_(kp, l)] * Eydiffx[ix_(ip, j)]
            PHz[ix_(ii, jp)] = Bx[ix_(ki, lp)] * PHz[ix_(ii, jp)] + Ax[ix_(ki, lp)] * Eydiffx[ix_(ii, jp)]
            Hz[ix_(ip, j)] = Hz[ix_(ip, j)] + Dc[kp][:, l] * PHz[ix_(ip, j)]
            Hz[ix_(ii, jp)] = Hz[ix_(ii, jp)] + Dc[ix_(ki, lp)] * PHz[ix_(ii, jp)]

            ### update Ey component...
            # determine indices for entire, PML, and interior regions in Ey and property grids
            i = np.arange(1, nx - 2)  # indices for all components in Ey matrix to update
            j = np.arange(1, nz - 2)
            k = 2 * i + 1 # corresponding indices in property grids
            l = 2 * j + 1
            kp = k[(k <= kpmlLin) | (k >= kpmlRin)]  # corresponding property indices in PML region
            lp = l[(l <= lpmlTin) | (l >= lpmlBin)]
            ki = k[(k > kpmlLin) & (k < kpmlRin)]  # corresponding property indices in interior (non-PML) region
            li = l[(l > lpmlTin) & (l < lpmlBin)]
            ip = kp // 2
            jp = lp // 2  # Ey indices in PML region
            ii = ki // 2
            ji = li // 2  # Ey indices in interior (non-PML) region

            # update to be applied to the whole Ey grid
            Hxdiffz[ix_(i, j)] = -Hx[ix_(i, j + 2)] + 27 * Hx[ix_(i, j + 1)] - 27 * Hx[ix_(i, j)] + Hx[ix_(i, j - 1)]
            Hzdiffx[ix_(i, j)] = -Hz[ix_(i+ 2, j)] + 27 * Hz[ix_(i + 1, j)] - 27 * Hz[ix_(i, j)] + Hz[ix_(i - 1, j)]
            Ey[ix_(i, j)] = Ca[ix_(k, l)] * Ey[ix_(i, j)] + Cbx[ix_(k, l)] * Hzdiffx[ix_(i, j)] - Cbz[ix_(k, l)] * Hxdiffz[ix_(i, j)]


            # update to be applied only to the PML region
            PEyx[ix_(ip, j)] = Bx[ix_(kp, l)] * PEyx[ix_(ip, j)] + Ax[ix_(kp, l)] * Hzdiffx[ix_(ip, j)]
            PEyx[ix_(ii, jp)] = Bx[ix_(ki, lp)] * PEyx[ix_(ii, jp)] + Ax[ix_(ki, lp)] * Hzdiffx[ix_(ii, jp)]
            PEyz[ix_(ip, j)] = Bz[ix_(kp, l)] * PEyz[ix_(ip, j)] + Az[ix_(kp, l)] * Hxdiffz[ix_(ip, j)]
            PEyz[ix_(ii, jp)] = Bz[ix_(ki, lp)] * PEyz[ix_(ii, jp)] + Az[ix_(ki, lp)] * Hxdiffz[ix_(ii, jp)]
            Ey[ix_(ip,j)] = Ey[ix_(ip, j)]+ Cc[ix_(kp, l)] * (PEyx[ix_(ip, j)] - PEyz[ix_(ip, j)])
            Ey[ix_(ii, jp)] = Ey[ix_(ii, jp)]  + Cc[ix_(ki, lp)] * (PEyx[ix_(ii, jp)] - PEyz[ix_(ii, jp)])

            # add source pulse to Ey at source location
            # (emulates infinitesimal Ey directed line source with current = srcpulse)

            iii = srci[s]
            jjj = srcj[s]
            
            Ey[iii,jjj] = Ey[iii,jjj] + srcpulse[it]


            # plot the Ey wavefield if necessary
            if (sav_wav==True) & (plotopt[0] == 1):
                if (it - 1) % plotopt[1] == 0:
                    #print(f"Source {s+1}/{nsrc}, Iteration {it}/{numit}, t = {t[it-1]*1e9:.2f} ns")
                    wav_cache[s, out_counter, ...] = Ey.T
                    out_counter += 1


            # record the results in gather matrix if necessary

            if (it - 1) % outstep == 0:
                tout[(it - 1) // outstep] = t[it - 1]
                for r in range(nrec):
                    gather[(it - 1) // outstep, r, s - 1] = Ey[reci[r], recj[r]]

            pbar.set_description(f"Source {s+1}/{nsrc}, t = {t[it-1]*1e9:.2f} ns")
            
    return gather, tout, srcx, srcz, recx, recz, wav_cache

