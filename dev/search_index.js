var documenterSearchIndex = {"docs":
[{"location":"tutorial_catalog/#Tutorial:-Catalogue-and-Window","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"","category":"section"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"In this tutorial we assume that your basic starting point is that you have a catalogue of galaxy positions (rthetaphi) and some way to calculate the window function, for example each voxel being proportional to the number of galaxies in a random catalogue.","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"Further, we assume that SphericalFourierBesselDecompositions.jl is already installed, and that you have a basic understanding of what the package is supposed to do.","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"First, load the package and create a shortcut","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"using SphericalFourierBesselDecompositions\nSFB = SphericalFourierBesselDecompositions","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"We will always assume that this shortcut has been created, as the package SFB does not export any symbols itself.","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"To perform a SFB decomposition, we create a modes object amodes that contains the modes and basis functions, and for the pseudo-SFB power spectrum we create cmodes,","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"kmax = 0.05\nrmin = 500.0\nrmax = 1500.0\namodes = SFB.AnlmModes(kmax, rmin, rmax)\ncmodes = SFB.ClnnModes(amodes, Δnmax=0)","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"Here, kmax is the maximum k-mode to be calculated, rmin and rmax are the radial boundaries. Δnmax specifies how many off-diagonals k ≠ k' to include.","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"The objects amodes and cmodes are used to access elements in the arrays produced by the routines below. For example, performing the SFB transform on a catalog, we use anlm = SFB.cat2anlm(...), and the individual modes can be accessed via anlm[SFB.getidx(amodes, n, l, m)].","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"The window function is described by","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"nr = 50\nwmodes = SFB.ConfigurationSpaceModes(rmin, rmax, nr, amodes.nside)\nwin = SFB.SeparableArray(phi, mask, name1=:phi, name2=:mask)\nwin_rhat_ln = SFB.win_rhat_ln(win, wmodes, amodes)\nVeff = SFB.integrate_window(win, wmodes)","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"where nr is the number of radial bins, phi is an array of length nr, and mask is a HEALPix mask in ring order. In general, win can be a 2D-array, where the first dimension is radial, and the second dimension is the HEALPix mask at each radius. Using a SeparableArray uses Julia's dispatch mechanism to call more efficient specialized algorithms when the radial and angular window are separable. SFB.win_rhat_ln() performs the radial transform of the window, SFB.integrate_window() is a convenient way to calculate the effective volume Veff.","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"The SFB decomposition for a catalogue of galaxies is now performed with","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"anlm = SFB.cat2amln(rθϕ, amodes, nbar, win_rhat_ln)\nCNobs = SFB.amln2clnn(anlm, cmodes)","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"where rθϕ is a 3 × Ngalaxies array with the r, θ, and ϕ coordinates of each galaxy in the survey, and nbar = Ngalaxies / Veff is the average number density. The second line calculates the pseudo-SFB power spectrum.","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"Shot noise and pixel window are removed with","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"Nobs_th = SFB.win_lnn(win, wmodes, cmodes) ./ nbar\npixwin = SFB.pixwin(cmodes)\nCobs = @. (CNobs - Nobs_th) / pixwin ^ 2","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"Window deconvolution is performed with bandpower binning:","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"w̃mat, vmat = SFB.bandpower_binning_weights(cmodes; Δℓ=Δℓ, Δn=Δn)\nbcmodes = SFB.ClnnBinnedModes(w̃mat, vmat, cmodes)\nbcmix = SFB.power_win_mix(win, w̃mat, vmat, wmodes, bcmodes)\nC = bcmix \\ (w̃mat * Cobs)","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"The first line calculates binning matrices w̃ and v for bin sizes Δℓ ~ 1/fsky and Δn = 1, the second line describes modes similar to cmodes but for bandpower binned modes. The coupling matrix is calculated in the third line, and the last line does the binning and deconvolves the window function.","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"To compare with a theoretical prediction, we calculate the deconvolved binning matrix wmat,","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"using LinearAlgebra\nwmat = bcmix * SFB.power_win_mix(win, w̃mat, I, wmodes, bcmodes)","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"The modes of the pseudo-SFB power spectrum are given by","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"lkk = SFB.getlkk(bcmodes)","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"where for a given i the element lkk[1,i] is the ℓ-mode, lkk[2,i] is the n-mode, lkk[3,i] is the n'-mode of the pseudo-SFB power spectrum element C[i].","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"An unoptimized way to calculate the covariance matrix is","category":"page"},{"location":"tutorial_catalog/","page":"Tutorial: Catalogue and Window","title":"Tutorial: Catalogue and Window","text":"VW = SFB.calc_covariance_exact_chain(C_th, nbar, win, wmodes, cmodes)\nV = inv(bcmix) * w̃mat * VW * w̃mat' * inv(bcmix)'","category":"page"},{"location":"myindex/#Index","page":"Index","title":"Index","text":"","category":"section"},{"location":"myindex/#SFB","page":"Index","title":"SFB","text":"","category":"section"},{"location":"myindex/","page":"Index","title":"Index","text":"Modules = [SphericalFourierBesselDecompositions]","category":"page"},{"location":"myindex/#SFB.GNL","page":"Index","title":"SFB.GNL","text":"","category":"section"},{"location":"myindex/","page":"Index","title":"Index","text":"Modules = [SphericalFourierBesselDecompositions.GNL]","category":"page"},{"location":"myindex/#SFB.Modes","page":"Index","title":"SFB.Modes","text":"","category":"section"},{"location":"myindex/","page":"Index","title":"Index","text":"Modules = [SphericalFourierBesselDecompositions.Modes]","category":"page"},{"location":"myindex/#SFB.SeparableArrays","page":"Index","title":"SFB.SeparableArrays","text":"","category":"section"},{"location":"myindex/","page":"Index","title":"Index","text":"Modules = [SphericalFourierBesselDecompositions.SeparableArrays]","category":"page"},{"location":"myindex/#SFB.Cat2Anlm","page":"Index","title":"SFB.Cat2Anlm","text":"","category":"section"},{"location":"myindex/","page":"Index","title":"Index","text":"Modules = [SphericalFourierBesselDecompositions.Cat2Anlm]","category":"page"},{"location":"myindex/#SFB.WindowChains","page":"Index","title":"SFB.WindowChains","text":"","category":"section"},{"location":"myindex/","page":"Index","title":"Index","text":"Modules = [SphericalFourierBesselDecompositions.WindowChains]","category":"page"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"The functions in here are exported into the main module. We will often assume that the user has defined the shortcut to the module:","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"SFB = SphericalFourierBesselDecompositions","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Then, all functions can be called via SFB.funcname(). For example, the SFB.GNL.SphericalBesselGnl() constructor in the SFB.GNL module is called via SFB.SphericalBesselGnl().","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/#SFB","page":"Reference","title":"SFB","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [SphericalFourierBesselDecompositions]","category":"page"},{"location":"reference/#SphericalFourierBesselDecompositions.calc_fsky-Tuple{Any, Any}","page":"Reference","title":"SphericalFourierBesselDecompositions.calc_fsky","text":"calc_fsky(win, wmodes)\n\nThis functions returns a measure of the sky fraction covered by the survey with window win. The exact implementation is considered an implementation detail and can change in the future.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SphericalFourierBesselDecompositions.pixwin-Tuple{Any}","page":"Reference","title":"SphericalFourierBesselDecompositions.pixwin","text":"pixwin(cmodes)\n\nReturn the angular pixel window for application to a Clnn object.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SFB.GNL","page":"Reference","title":"SFB.GNL","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [SphericalFourierBesselDecompositions.GNL]","category":"page"},{"location":"reference/#SphericalFourierBesselDecompositions.GNL.SphericalBesselGnl-NTuple{4, Any}","page":"Reference","title":"SphericalFourierBesselDecompositions.GNL.SphericalBesselGnl","text":"SphericalBesselGnl(nmax, lmax, rmin, rmax)\nSphericalBesselGnl(kmax, rmin, rmax)\n\nGenerate gnl(n,l,r). Returns a struct that can be called for calculating gnl. Note that the last argument is r, not kr.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SphericalFourierBesselDecompositions.GNL.calc_first_n_zeros-Tuple{Any, Any}","page":"Reference","title":"SphericalFourierBesselDecompositions.GNL.calc_first_n_zeros","text":"calc_first_n_zeros(func, nmax; δ=π/20, xmin=0.0)\n\nCalculate the first nmax zeros of the function func. Assumes that zeros are spaced more than δ apart. First zero could be xmin.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SphericalFourierBesselDecompositions.GNL.calc_knl_potential-NTuple{4, Any}","page":"Reference","title":"SphericalFourierBesselDecompositions.GNL.calc_knl_potential","text":"calc_knl_potential(nmax, lmax, rmin, rmax)\ncalc_knl_potential(kmax, rmin, rmax)\n\nCalculate the knl for potential boundary conditions.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SphericalFourierBesselDecompositions.GNL.calc_zeros-Tuple{Any, Any, Any}","page":"Reference","title":"SphericalFourierBesselDecompositions.GNL.calc_zeros","text":"calc_zeros(func, xmin, xmax; δ=π/20)\n\nCalculate all zeros of the function func between xmin and xmax. Assumes that zeros are spaced more than δ apart.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SFB.Modes","page":"Reference","title":"SFB.Modes","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [SphericalFourierBesselDecompositions.Modes]","category":"page"},{"location":"reference/#SphericalFourierBesselDecompositions.Modes.AnlmModes","page":"Reference","title":"SphericalFourierBesselDecompositions.Modes.AnlmModes","text":"AnlmModes\n\nThis is where we define which modes are included. As our criterium, we set a maximum qmax = kmax * rmax, and we include all modes below that.\n\nThe modes are arranged in the following order. The fastest loop is through 'm', then 'l', finally 'n', from small number to larger number. We restrict 'm' to m > 0, and we assume a real field.\n\nExample:\n\njulia> kmax = 0.2\njulia> rmax = 1000.0\njulia> modes = AnlmModes(kmax * rmax)\n\n\n\n\n\n","category":"type"},{"location":"reference/#SphericalFourierBesselDecompositions.Modes.ClnnModes","page":"Reference","title":"SphericalFourierBesselDecompositions.Modes.ClnnModes","text":"ClnnModes(::AnlmModes)\n\nThis is where we define which modes are included in the power spectrum, given a AnlmModes struct.\n\nThe modes are arranged in the following order. The fastest loop is through 'n̄', then 'Δn', finally 'ℓ', from small number to larger number.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SphericalFourierBesselDecompositions.Modes.getlkk-Tuple{SphericalFourierBesselDecompositions.Modes.ClnnModes, Any}","page":"Reference","title":"SphericalFourierBesselDecompositions.Modes.getlkk","text":"getlkk(::ClnnModes, [i])\ngetlkk(::ClnnBinnedModes, [i])\n\nGet the physical modes ℓ, k, and k' corresponding to the index i. If i is left out, an array lkk of all modes is returned so that lkk[1,:] are all the ℓ-values, lkk[2,:] all the k-values, and lkk[3,:] are all the k'-values.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SFB.SeparableArrays","page":"Reference","title":"SFB.SeparableArrays","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [SphericalFourierBesselDecompositions.SeparableArrays]","category":"page"},{"location":"reference/#SphericalFourierBesselDecompositions.SeparableArrays.exponentiate-Tuple{SphericalFourierBesselDecompositions.SeparableArrays.SeparableArray, Number}","page":"Reference","title":"SphericalFourierBesselDecompositions.SeparableArrays.exponentiate","text":"exponentiate(s::SeparableArray, exp::Number)\n\nThis function is used for elementwise exponentiation of the array 's'. It could be made more elegant by extending the broadcast syntax. PRs welcome.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SFB.Cat2Anlm","page":"Reference","title":"SFB.Cat2Anlm","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [SphericalFourierBesselDecompositions.Cat2Anlm]","category":"page"},{"location":"reference/#SphericalFourierBesselDecompositions.Cat2Anlm.cat2amln-NTuple{4, Any}","page":"Reference","title":"SphericalFourierBesselDecompositions.Cat2Anlm.cat2amln","text":"cat2amln(rθϕ, amodes, nbar, win_rhat_ln)\n\nComputes the spherical Fourier-Bessel decomposition coefficients from a catalogue of sources. The number density is measured from the survey as bar n = N_mathrmgals  V_mathrmeff.\n\nExample\n\njulia> using SphericalFourierBesselDecompositions\njulia> cat2amln(rθϕ, ...)\n\n\n\n\n\n","category":"method"},{"location":"reference/#SFB.WindowChains","page":"Reference","title":"SFB.WindowChains","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [SphericalFourierBesselDecompositions.WindowChains]","category":"page"},{"location":"reference/#SphericalFourierBesselDecompositions.WindowChains","page":"Reference","title":"SphericalFourierBesselDecompositions.WindowChains","text":"WindowChains\n\nThis module exposes several ways to calculate a chain of window functions. All require a cache to be created that speeds up subsequent calls. The type of cache determines which method is used.\n\nTo create the cache, simply call one of\n\njulia> cache = SFB.WindowChains.WindowChainsCacheWignerChain(win, wmodes, amodes)\njulia> cache = SFB.WindowChains.WindowChainsCacheFullWmix(win, wmodes, amodes)\njulia> cache = SFB.WindowChains.WindowChainsCacheFullWmixOntheflyWmix(win, wmodes, amodes)\njulia> cache = SFB.WindowChains.WindowChainsCacheSeparableWmix(win, wmodes, amodes)\njulia> cache = SFB.WindowChains.WindowChainsCacheSeparableWmixOntheflyWlmlm(win, wmodes, amodes)\njulia> cache = SFB.WindowChainsCache(win, wmodes, amodes)\n\nThe last one will automatically select the typically fastest algorithm.\n\nThen to calculate an element of a chain of windows, use window_chain(ell, n1, n2, cache). For example, for a chain of four windows,\n\njulia> ell = [0, 1, 2, 3]\njulia> n1 = [1, 2, 3, 4]\njulia> n2 = [5, 6, 7, 8]\njulia> Wk = SFB.window_chain(ell, n1, n2, cache)\n\n\n\n\n\n","category":"module"},{"location":"reference/#SphericalFourierBesselDecompositions.WindowChains.window_chain-NTuple{5, Any}","page":"Reference","title":"SphericalFourierBesselDecompositions.WindowChains.window_chain","text":"window_chain(ell, n1, n2, cache, symmetries)\n\nThis version adds up several window chains taking into account the symmetries. symmetries is an array of pairs of numbers specifying the symmetries to consider. Each pair specifies the ℓnn′ index and the type of symmetry. For example, when k≥3, then symmetries = [1=>0, 2=>1, 3=>2] would specify that no symmetries are taken into account for the first lnn′ combination, the second symmetry will flip n and n′ and add the result only when n ≠ n′, and the third will add the result regardless whether the n equal or not.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SphericalFourierBesselDecompositions","category":"page"},{"location":"#SuperFaB-Documentation","page":"Home","title":"SuperFaB Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The purpose of this Julia module is to provide an efficient implementation for spherical Fourier-Bessel decomposition of a scalar field. The details of the algorithm are presented in 2102.10079. SuperFaB is implemented in the package SphericalFourierBesselDecompositions.jl.","category":"page"},{"location":"#Contents","page":"Home","title":"Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install SuperFaB, start the Julia REPL and type","category":"page"},{"location":"","page":"Home","title":"Home","text":"]add SphericalFourierBesselDecompositions","category":"page"},{"location":"","page":"Home","title":"Home","text":"# If that doesn't work (there is a mandatory waiting period for the Julia\n# General registry), install it directly\n# ```julia\n# ]add https://github.com/hsgg/SphericalFourierBesselDecompositions.jl.git\n# ```","category":"page"},{"location":"","page":"Home","title":"Home","text":"The package makes use of some python packages (i.e. healpy) that are only supported on MacOSX and Linux. The above command should work if healpy is already installed, but if problems occur when first using the package, see PyCall. Specifically, if you do ENV[\"PYTHON\"]=\"\"; ]build PyCall, then Julia will download its own python and use Conda to download healpy and dependencies once you load SuperFaB.","category":"page"}]
}
