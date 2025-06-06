from scipy.optimize import minimize
import scipy.sparse as sp
import time
import numpy as np
import math
from tqdm import tqdm 

def locate_ancestors(samples, times, 
                     shared_times_chopped, shared_times_chopped_centered_inverted, locations, 
                     log_weights=[0], sigma=1, x0_final=None, BLUP=False, BLUP_var=False, quiet=False, sample_times=None):

    """
    Locate genetic ancestors given sample locations and shared times.
    """
   
    if not quiet: print('\n%%%%%%%%%%%% locating ancestors with spacetrees %%%%%%%%%%%%')

    M = len(shared_times_chopped)
    try: 
        n, d = locations.shape
    except:
        n = len(locations)
        d = 1
    if not quiet: print('number of trees per locus:',M,'\nnumber of samples:',n,'\nnumber of spatial dimensions:',d)
    if not quiet: print('samples:',samples,'\ntimes:',times)

    # preprocess locations 
    mean_location = np.mean(locations, axis=0) #mean location
    Tmat = np.identity(n) - [[1/n for _ in range(n)] for _ in range(n)]; Tmat = Tmat[:-1] #mean centering matrix
    locations_centered = np.matmul(Tmat, locations) #centered locations

    # preprocess shared times
    stmrs = []
    stms = []
    stcilcs = []
    for stsc, stci in zip(shared_times_chopped, shared_times_chopped_centered_inverted): #over trees
        stmr = np.mean(stsc, axis=1) #average times in each row
        stmrs.append(stmr)
        stm = np.mean(stmr) #average times in whole matrix
        stms.append(stm)
        stcilc = np.matmul(stci, locations_centered) #a product we will use
        stcilcs.append(stcilc)

    if sample_times is None:
        sample_times = np.zeros(n) #all contemporary unless otherwise stated

    ancestor_locations = []
    for sample in tqdm(samples):
        for time in times:

            if time < sample_times[sample]:
           
                print('trying to locate ancestor more recent than sample')
                mle = locations[sample]

            else:

                # calculate likelihoods or mles over loci
                fs = []
                mles = []
                bvars = []
                for stsc, stmr, stm, stcilc in zip(shared_times_chopped, stmrs, stms, stcilcs): #over trees
                
                    at = _anc_times(stsc, time, sample) #shared times between samples and ancestor of sample at time 
                    atc = np.matmul(Tmat, (at[:-1] - stmr)) #center this
                    taac = at[-1] - 2*np.mean(at[:-1]) + stm #center shared times of ancestor with itself
                    mle = mean_location + np.matmul(atc.transpose(), stcilc) #most likely location
               
                    # if getting best linear unbiased predictor we collect the mles at each tree (and optionally variance)   
                    if BLUP:
                        mles.append(mle)
                        if BLUP_var:
                            var = (taac - np.matmul(np.matmul(atc.transpose(), stci), atc)) #variance in loc (multiply by sigma later)
                            bvars.append(var)
                    # and otherwise we get the full likelihood at each tree
                    else:
                        var = (taac - np.matmul(np.matmul(atc.transpose(), stci), atc)) * sigma #variance in loc
                        fs.append(lambda x: _lognormpdf(x, mle, var)) #append likelihood
               
                # locate ancestor
                if BLUP:
                    blup = np.zeros(d) 
                    tot_weight = 0
                    # weighted average of mles
                    for mle, log_weight in zip(mles, log_weights):
                         blup += mle * np.exp(log_weight)
                         tot_weight += np.exp(log_weight)
                    mle = blup/tot_weight
                    # weighted average of variances
                    if BLUP_var:
                        blup_var = 0
                        for bvar, log_weight in zip(bvars, log_weights):
                             blup_var += bvar * np.exp(log_weight)
                        mle = [mle, blup_var/tot_weight]
                else:
                    # find min of negative of log of summed likelihoods (weighted by importance)
                    def g(x): 
                        return -_logsumexp([f(x) + log_weight for f,log_weight in zip(fs, log_weights)])
                    x0 = locations[sample] 
                    if x0_final is not None:
                        x0 = x0 + (x0_final - x0)*time/times[-1] #make a linear guess
                    mle = minimize(g, x0=x0).x
            
            ancestor_locations.append(np.append([sample,time],mle))
        
    return ancestor_locations

def estimate_dispersal(locations, shared_times_inverted, shared_times_logdet=None, 
                       sigma0=None, bnds=None, method='L-BFGS-B', callbackF=None,
                       important=True, branching_times=None, sample_times=None, phi0=None, scale_phi=None, logpcoals=None,
                       quiet=False, BLUP=False):

    """
    Numerically estimate maximum likelihood dispersal rate (and possibly branching rate) given sample locations and shared times.
    """

    if not quiet: print('\n%%%%%%%%%%%% inferring dispersal with spacetrees %%%%%%%%%%%%')

    L = len(shared_times_inverted)
    M = len(shared_times_inverted[0])
    try: 
        n, d = locations.shape
    except:
        n = len(locations)
        d = 1
    if not quiet: print('number of loci:',L,'\nnumber of trees per locus:',M,'\nnumber of samples:',n,'\nnumber of spatial dimensions:',d,'\n')

    # mean center locations
    Tmat = np.identity(n) - [[1/n for _ in range(n)] for _ in range(n)]; Tmat = Tmat[0:-1] #mean centering matrix 
    locations = np.matmul(Tmat, locations) #mean centered locations
    locations_vector = np.transpose(locations).flatten() #make a vector

    # find decent initial dispersal rate
    if sigma0 is None or BLUP:
        if not quiet: print('initializing dispersal rate...')
        guess = np.zeros((d,d))
        for stss in tqdm(shared_times_inverted): #loop over loci
            for sts in stss: #loop over trees
                guess += _mle_dispersal_tree(locations, sts) 
        guess = guess/(L*M) #avg mle over all trees and loci (note that we can avg over all trees and loci simultaneously because same number of trees at every locus)
        x0 = _sigma_to_params(guess) #convert initial dispersal rate to standard deviations and correlations, to feed into numerical search
        if BLUP:            
            return x0 #best linear unbiased predictor (returned as sds and corrs, like numerical search below)
        x0 = [i/2 for i in x0] #heuristic because the estimate seems to be a consistent overestimate
        if not quiet: print('initial dispersal rate:',x0)
    else:                   
        x0 = sigma0         
                            
    # initializing branching rate
    if important:
        if phi0 is None:
            phi0 = np.mean([np.log(n/(n-len(bts)+1))/bts[-1] for btss in branching_times for bts in btss]) #initial guess at branching rate, from n(t)=n(0)e^(phi*t) - assumes all samples contemporary but just a rough estimate anyhow
            if not quiet: print('initial branching rate:',phi0) 
        if scale_phi is None:
            scale_phi = x0[0]/phi0 #we will search for the value of phi*scale_phi that maximizes the likelihood (putting phi on same scale as dispersal accelarates search) 
        x0.append(phi0*scale_phi)
        
    # negative composite log likelihood ratio, as function of x
    f = _sum_mc(locations=locations_vector, shared_times_inverted=shared_times_inverted, shared_times_logdet=shared_times_logdet,
                important=important, branching_times=branching_times, sample_times=sample_times, scale_phi=scale_phi, logpcoals=logpcoals)

    # impose bounds on parameters
    if bnds is None:
        bnds = [(1e-6,None)] #sd_x
        if d==2:
            bnds.append((1e-6,None)) #sd_y
            bnds.append((-0.99,0.99)) #cor_xy
        if d==3:
            bnds.append((1e-6,None)) #sd_y
            bnds.append((1e-6,None)) #sd_z
            bnds.append((-0.99,0.99)) #cor_xy
            bnds.append((-0.99,0.99)) #cor_xz
            bnds.append((-0.99,0.99)) #cor_yz
        if important:
            bnds.append((1e-6,None)) #scaled phi

    # find mle
    if not quiet: print('\nsearching for maximum likelihood parameters...')
    if callbackF is not None: callbackF(x0)
    t0 = time.time()
    m = minimize(f, x0=x0, bounds=bnds, method=method, callback=callbackF) #find MLE
    if not quiet: print(m)
    if not quiet: print('finding the max took', time.time()-t0, 'seconds')

    mle = m.x
    if important:
        mle[-1] = mle[-1]/scale_phi #unscale phi
    if not quiet:
        if important:
            sigma = _params_to_sigma(mle[:-1]) #convert to covariance matrix
            print('\nmaximum likelihood branching rate:',mle[-1])
        else:
            sigma = _params_to_sigma(mle)
        print('\nmaximum likelihood dispersal rate:\n',sigma)

    return mle 

def _get_focal_index(focal_node, listoflists):

    """
    get the subtree and index within that subtree for focal_node (listoflists here is list of samples for each subtree)
    """

    for i,j in enumerate(listoflists):
        if focal_node in j:
            n = i
            for k,l in enumerate(j):
                if focal_node == l:
                    m = k
    return n,m

def _anc_times(shared_times, ancestor_time, sample):

    """
    get shared times with ancestor 
    """
    
    taa = np.max(shared_times) - ancestor_time #shared time of ancestor with itself 

    anc_times = [] 
    for t in shared_times[sample]:
        anc_times.append(min(t, taa)) # shared times between ancestor and each sample lineage

    anc_times.append(taa) #add shared time with itself
        
    return np.array(anc_times)

def _lognormpdf(x, mu, S):

    """
    Calculate log probability density of x, when x ~ N(mu,S)
    """

    norm_coeff = np.linalg.slogdet(S)[1] #just care about relative likelihood so drop the constant

    # term in exponential (times -2)
    err = x - mu #difference between mean and data
    if sp.issparse(S):
        numerator = spln.spsolve(S, err).T.dot(err) #use faster sparse methods if possible
    else:
        numerator = np.linalg.solve(S, err).T.dot(err) #just a fancy way of calculating err.T * S^-1  * err

    return -0.5 * (norm_coeff + numerator) #add the two terms together and multiply by -1/2

def _mle_dispersal_tree(locations, shared_times_inverted):

    """
    Maximum likelihood estimate of dispersal rate given locations and (inverted) shared times between lineages in a tree.
    """

    return np.matmul(np.matmul(np.transpose(locations), shared_times_inverted), locations) / len(locations)

def _sigma_to_params(sigma):

    """
    Extract list of parameters from covariance matrix
    """

    sdx = sigma[0,0]**0.5 
    d = len(sigma)
    if d==1:
      return [sdx]
    if d>1:
      sdy = sigma[1,1]**0.5 
      corxy = sigma[0,1]/(sdx*sdy)
    if d==3:
      sdz = sigma[2,2]**0.5
      corxz= sigma[0,2]/(sdx*sdz)
      coryz = sigma[1,2]/(sdy*sdz)
      return [sdx,sdy,sdz,corxy,corxz,coryz]
    if d==2:
      return [sdx,sdy,corxy]

def _params_to_sigma(x):

    """
    Convert list of parameters to covariance matrix
    """

    sdx = x[0]
    if len(x) == 1:
        return np.array([[sdx**2]])
    if len(x) == 3:
        sdy = x[1]
        rho = x[2]
        cov = sdx*sdy*rho
        return np.array([[sdx**2, cov], [cov, sdy**2]])
    if len(x) == 6:
        sdy = x[1]
        sdz = x[2]
        corxy = x[3]
        corxz = x[4]
        coryz = x[5]
        covxy = sdx*sdy*corxy
        covxz = sdx*sdz*corxz
        covyz = sdy*sdz*coryz
        return np.array([[sdx**2, covxy, covxz], [covxy, sdy**2, covyz], [covxz, covyz, sdz**2]])

def _sum_mc(locations, shared_times_inverted, shared_times_logdet,
            important=False, branching_times=None, sample_times=None, scale_phi=None, logpcoals=None):

    """
    Negative log composite likelihood of parameters x given the locations and shared times at all loci and subtrees, as function of x.
    """

    if not important:
        L = len(shared_times_logdet) #number of loci
        branching_times = [None for _ in range(L)]
        logpcoals = branching_times

    def sumf(x):

        # reformulate parameters
        if important:
            sigma = _params_to_sigma(x[:-1])
            phi = x[-1]/scale_phi
        else:
            sigma = _params_to_sigma(x)
            phi = None 
        log_det_sigma = np.linalg.slogdet(sigma)[1] #log of determinant
        sigma_inverted = np.linalg.inv(sigma) #inverse

        # calculate negative log composite likelihood ratio
        # by subtracting log likelihood ratio at each locus
        g = 0
        for sti, ldst, bts, lpcs in zip(shared_times_inverted, shared_times_logdet, branching_times, logpcoals): #loop over loci
            g -= _mc(locations=locations, shared_times_inverted=sti, shared_times_logdet=ldst,
                     sigma_inverted=sigma_inverted, log_det_sigma=log_det_sigma,
                     important=important, branching_times=bts, sample_times=sample_times, phi=phi, logpcoals=lpcs)
        return g
    
    return sumf

def _mc(locations, shared_times_inverted, shared_times_logdet, sigma_inverted, log_det_sigma,
        important=False, branching_times=None, sample_times=None, phi=None, logpcoals=None):

    """
    Monte Carlo estimate of log of likelihood ratio of the locations given parameters (sigma,phi) vs data given standard coalescent, for a given locus
    """

    LLRs = [] #log likelihood ratios at each tree

    # loop over trees at a locus
    if important:
        for sti, ldst, bts, lpc in zip(shared_times_inverted, shared_times_logdet, branching_times, logpcoals):
            LLRs.append(_log_likelihoodratio(locations=locations, shared_times_inverted=sti, shared_times_logdet=ldst,
                                             sigma_inverted=sigma_inverted, log_det_sigma=log_det_sigma, 
                                             important=important, branching_times=bts, phi=phi, logpcoals=lpc))

    else:
        for sti, ldst in zip(shared_times_inverted, shared_times_logdet):
            LLRs.append(_log_likelihoodratio(locations=locations, shared_times_inverted=sti, shared_times_logdet=ldst,
                                             sigma_inverted=sigma_inverted, log_det_sigma=log_det_sigma,
                                             important=important))
    
    return _logsumexp(np.array(LLRs)) #sum likelihood ratios over trees then take log

def _logsumexp(a):

    """
    Take the log of a sum of exponentials without losing information.
    """

    a_max = np.max(a) #max element in list a
    tmp = np.exp(a - a_max) #now subtract off the max from each a before taking exponential (ie divide sum of exponentials by exp(a_max))
    s = np.sum(tmp) #and sum those up
    out = np.log(s) #and take log
    out += a_max  #and then add max element back on (ie multiply sum by exp(a_max), ie add log(exp(a_max)) to logged sum)

    return out

def _log_likelihoodratio(locations, shared_times_inverted, shared_times_logdet, sigma_inverted, log_det_sigma,
                         important=False, branching_times=None, sample_times=None, phi=None, logpcoals=None):

    """ 
    Log of likelihood ratio of parameters under branching brownian motion vs standard coalescent.
    """
  
    # log likelihood of dispersal rate
    k = len(shared_times_inverted);
    LLR = _location_loglikelihood(locations, shared_times_inverted, shared_times_logdet, sigma_inverted)
    d,_ = sigma_inverted.shape
    LLR -= k/2 * (d*np.log(2*np.pi) + log_det_sigma)  #can factor this out over subtrees
    if sample_times is None:
        sample_times = np.zeros(k+1) #log_birth_density needs to know how many lineages to start with

    if important:
        # log probability of branching times given pure birth process with rate phi
        LLR += _log_birth_density(branching_times=branching_times, sample_times=sample_times, phi=phi) 
        # log probability of coalescence times given standard coalescent (precalculated as parameter-independent)
        LLR -= logpcoals
    
    return LLR

def _location_loglikelihood(locations, shared_times_inverted, shared_times_logdet, sigma_inverted):

    """
    Log probability density of locations when locations ~ MVN(0,sigma_inverted*shared_times_inverted).
    """
    
    # log of coefficient in front of exponential (times -2)
    d,_ = sigma_inverted.shape
    logcoeff = d*shared_times_logdet #just the part that depends on data

    # exponent (times -2)
    exponent = np.matmul(np.matmul(np.transpose(locations), np.kron(sigma_inverted, shared_times_inverted)), locations)   

    return -0.5 * (logcoeff + exponent) #add the two terms together and multiply by -1/2

def _log_birth_density(branching_times, sample_times, phi, condition_on_n=True):

    """
    Log probability of branching times given Yule process with branching rate phi.
    """

    T = branching_times[-1] #storing total time as last entry for convenience
    n = sum(sample_times<T) #number of samples before cutoff
    sample_times = sample_times[sample_times>0] #remove contemporary sample times
    sample_times = np.flip(T - sample_times) #forward in time perspective of sampling times
    sample_times = sample_times[sample_times>0] #ignore sampling older than cutoff
    n0 = n - (len(branching_times) - 1) #initial number of lineages (number of samples minus number of coalescence events)
    
    logp = 0 #initialize log probability
    prevt = 0 #initialize time
    k = n0 #initialize number of lineages
    i = 0 #index of next sampling time
    # probability of each branching time
    for t in branching_times[:-1]: #for each branching time t
        while i<len(sample_times) and sample_times[i] < t: #if next sampling happens before next branching
            logp += - k * phi * (sample_times[i] - prevt) #log prob of no branching until sampling
            prevt = sample_times[i] #update time
            k -= 1 #remove lineage
            i += 1 #next sample time
        logp += np.log(k * phi) - k * phi *  (t - prevt) #log probability of waiting time t-prevt with k lineages
        prevt = t #update time
        k += 1 #update number of lineages

    # deal with any remaining sampling times
    while i<len(sample_times): 
        logp += - k * phi * (sample_times[i] - prevt) #log prob of no branching until sampling
        prevt = sample_times[i] #update time
        k -= 1 #remove lineage
        i += 1 #next sample time

    # probability of no branching from most recent event to T
    logp += - k * phi * (T - prevt)

    # condition on having k samples from n0 in given time
    if condition_on_n:
        i = 0 #reset index of next sampling time
        prevt = 0
        while i<len(sample_times): 
            k = n0 + sum([1 for t in branching_times if t>prevt and t<sample_times[i]]) #number of lineages at next sampling time
            logp -= np.log(math.comb(k - 1, k - n0) * (1 - np.exp(-phi * (sample_times[i]-prevt)))**(k - n0)) - phi * n0 * (sample_times[i]-prevt) # see page 234 of https://www.pitt.edu/~super7/19011-20001/19531.pdf for two different expressions
            prevt = sample_times[i] #update time
            i += 1 #move to next sampling time
            n0 = k #update number of lineages

    return logp
