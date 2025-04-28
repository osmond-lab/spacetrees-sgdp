progdir = 'programs/'
relate = progdir + 'relate/'
datadir = "data/" # where to put data

TREESKIP = 1000 # change to 1000
Ms = [100]
CHRS = [18]
ms = ['1.25e-08']
#tCutoffs = [None, int(1e6), int(1e5), int(1e4)] #when to chop off the trees for dispersal rate estimates
tCutoffs = [None]
samples = range(556)

anc = "data/SGDP_v1_annot_ne_chr{CHR}.anc"
mut = "data/SGDP_v1_annot_ne_chr{CHR}.mut"
coal = "data/SGDP_v1_annot_ne_popsize.coal"
pair_coal = "data/SGDP_v1_annot_ne_popsize.pairwise.coal"
poplabels = "data/SGDP.poplabels"
metadata = "data/SGDP_metadata.279public.21signedLetter.samples.txt"
projected_locations = "data/SGDP_new_locations.npy"

# -------------------- get relate ------------------

relate = progdir + 'relate/'

rule get_relate:
  input:
  output:
   relate + 'bin/Relate'
  resources:
    time = 1
  threads: 1
  shell:
    '''
    git clone https://github.com/MyersGroup/relate.git
    cd relate/build
    module load cmake/3.21.4 gcc/8.3.0 gsl/2.5
    cmake ..
    make
    #cd -
    #mv relate/ {progdir} #for some reason this fails with 'file exists', even when it doesnt exist (or so it seems), so i did this last step manualy
    ''' 




# ------------ decide which trees to sample ----------------


bps = anc.replace('anc','bps')

checkpoint get_bp:
  input:
    anc,
    mut
  output:
    bps 
  run:
    print('getting tree indices')
    ixs_start=[]
    ixs_end=[]
    with open(input[0], "r") as f:
      for i, line in enumerate(f): #each line is a tree, see https://myersgroup.github.io/relate/getting_started.html#Output
        if i==1: 
          n = int(line.split()[1]) #number of trees on this chromosome
          trees = [i for i in range(0,n+1,TREESKIP)] #which trees to sample
        if i > 1 and i-2 in trees:
          ixs_start.append(int(line.split(':')[0])) #index of first snp in sampled tree
        if i > 2 and i-3 in trees: 
          ixs_end.append(int(line.split(':')[0])-1) #index of last snp in sampled tree
    print('choose',len(ixs_start),'trees')
    print('getting start and stop basepairs')
    bps_start = []
    bps_end = []
    with open(input[1],"r") as f:
      for i,line in enumerate(f):
        if i>0 and int(line.split(';')[0]) in ixs_start:
          bps_start.append(int(line.split(';')[1])) #position of first snp in sampled tree
        if i>0 and int(line.split(';')[0]) in ixs_end:
          bps_end.append(int(line.split(';')[1])) #position of last snp in sampled tree
    print('writing to file')
    with open(output[0], 'w') as out:
      for start,end in zip(bps_start,bps_end):
        out.write(str(start) + ' ' + str(end) + '\n')
        
rule get_bps:
  input:
    expand(bps, CHR=CHRS)
    
# ------------ sample branch lengths at a particular location -------------

tree = anc.replace('.anc','_{start}-{stop}bps_{M}M.newick')


def input_func(name,ends = []): # give values to 3 wildcards (start/stop/M)

  def input_files(wildcards):
    filenames = []
    for CHR in CHRS:
      infile = checkpoints.get_bp.get(CHR=CHR).output[0]
      with open(infile,'r') as f:
        for line in f:
          i,j = line.strip().split(' ')
          d = {'{CHR}': CHR, '{start}': i, '{stop}': j}
          string = name
          for i,j in d.items():
            string = string.replace(i,str(j))
          filenames.append(string)
    return expand(filenames, M=Ms, tCutoff=tCutoffs, end=ends, allow_missing=True) 

  return input_files


rule sample_trees:
  input:
    input_func(tree) 

ruleorder: sample_tree > get_bp

rule sample_tree:
  input:
    anc,
    mut,
    coal,
    relate
  output:
    tree 
  params:
    prefix_in = anc.replace('.anc',''),
    prefix_out = tree.replace('.newick','')
  threads: 1
  resources: 
    runtime=100
  group: "sample_trees"
  shell:
    '''
    module load gcc/13.2.0
    {relate}scripts/SampleBranchLengths/SampleBranchLengths.sh \
                 -i {params.prefix_in} \
                 --coal {input[2]} \
                 -o {params.prefix_out} \
                 -m 1.25e-08 \
                 --format n \
                 --num_samples {wildcards.M} \
                 --first_bp {wildcards.start} \
                 --last_bp {wildcards.stop} \
                 --seed 1 
    '''

# snakemake sample_trees --profile slurm --group-components sample_trees=80 --jobs 10

# ------------ shared time matrices and coalescence times -------------

shared_times = tree.replace('.newick','_sts.npy')
coal_times = tree.replace('.newick','_cts.npy')

rule times:
  input:
    input_func(shared_times),
    input_func(coal_times)

rule time:
  input:
    tree
  output:
    shared_times,
    coal_times
  run:
    from tsconvert import from_newick
    from utils import get_shared_times
    import numpy as np
    stss = []
    ctss = []
    with open(input[0], mode='r') as f:
      next(f) #skip header
      for line in f: #for each tree sampled at a locus

        # Import
        string = line.split()[4] #extract newick string only (Relate adds some info beforehand)
        ts = from_newick(string,min_edge_length=1e-6) #convert to tskit "tree sequence" (only one tree)
        tree = ts.first() #the only tree
        
        # Shared times
        samples = [int(ts.node(node).metadata['name']) for node in ts.samples()] #get index of each sample in list we gave to relate
        # Where do these indices come from??
        sample_order = np.argsort(samples) #get indices to put in ascending order
        ordered_samples = [ts.samples()[i] for i in sample_order] #order samples as in relate
        sts = get_shared_times(tree, ordered_samples) #get shared times between all pairs of samples, with rows and columns ordered as in relate
        stss.append(sts)
        
        #coalescence times
        cts = sorted([tree.time(i) for i in tree.nodes() if not tree.is_sample(i)]) #coalescence times, in ascending order
        ctss.append(cts)
        
    np.save(output[0], np.array(stss))
    np.save(output[1], np.array(ctss))
    

# ------------ get location file -------------

locations = poplabels.replace('.poplabels','_locations.npy')

rule locations:
  input:
    expand(locations)

rule location:
  input:
    poplabels,
    metadata
  output:
    locations
  run:
    import numpy as np
    
    # get metadata
    lonlats = []
    with open(input[1],"rb") as f:
      next(f)
      for line in f:
        l = line.strip().split(b'\t')
        if l != [b'']:
          lonlat = []
          lonlat.append(l[4])
          lonlat.append(l[11])
          lonlat.append(l[12])
        lonlats.append(lonlat)
      
    # get accession names in metadata
    accessions = [i[0] for i in lonlats]
    
    # get ids from poplabels
    IDs = []
    with open(input[0],"rb") as f:
      next(f)
      for line in f:
        IDs.append(line.strip().split(b' ')[0])
        
    # for each row of metadata, get index in order we gave to relate 
    order = []
    for ID in IDs:
      ix = accessions.index(ID)
      order.append(ix)
    locations = np.array([list(map(float,[i[2],i[1]])) for i in lonlats])[order]
    np.save(output[0], locations)
   

# ------------- get coalessence file -----------

coal = anc.replace('.anc','_popsize{end}')
ends = ['.coal','.pairwise.coal','.pairwise.bin']

rule get_coal:
  input:
    anc,
    mut,
    poplabels,
    relate
  output:
    expand(coal, end=ends, allow_missing=True)
  params:
    prefix_in = anc.replace('.anc',''),
    prefix_out = coal.replace('{end}','')
  threads: 1
  resources: 
    runtime=60*4
  shell:
    '''
    module load gcc/8.3.0
    module load r/4.2.2-batteries-included #R needed for plots
    {relate}scripts/EstimatePopulationSize/EstimatePopulationSize.sh \
                 -i {params.prefix_in} \
                 -m 1.25e-8 \
                 --poplabels {input[2]} \
                 --seed 1 \
                 -o {params.prefix_out} \
                 --noanc 1 \
                 --num_iter 1 
    '''

rule get_coals:
  input:
    expand(coal, CHR=CHRS, end=ends) 

# snakemake get_coals --profile slurm --jobs 10


# ------------- process shared times ----------

processed_shared_times = shared_times.replace('_sts.npy','_sts_{tCutoff}tCutoff_{end}.npy')
ends = ['mc-invs','logdets','samples']

rule process_shared_time:
  input:
    shared_times
  output:
    expand(processed_shared_times, end=ends, allow_missing=True)
  threads: 1
  run:
    # taming numpy
    import os
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["GOTO_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    
    # transfer triangular matrix back to full stss tree matrices
    import numpy as np
    stss = np.load(input[0])
    _,n,_ = stss.shape
    
    # process trees 
    from utils import chop_shared_times, center_shared_times
    stss_inv = []
    stss_logdet = []
    smplsss = []
    tCutoff = wildcards.tCutoff
    if tCutoff=='None': 
      tCutoff=None 
    else:
      tCutoff=float(tCutoff)
    for sts in stss:
      # chop
      sts_chopped, smpls = chop_shared_times(sts, tCutoff=tCutoff) #shared times and samples of each subtree
      sts_inv = []
      sts_logdet = []
      smplss = []
      # process subtrees
      for st,sm in zip(sts_chopped, smpls):
        stc = center_shared_times(st) #mean center
        stc_inv = np.linalg.pinv(stc) #invert
        stc_logdet = np.linalg.slogdet(st)[1] #log determinant
        sts_inv.append(stc_inv)
        sts_logdet.append(stc_logdet) 
        smplss.append(sm) #samples
      stss_inv.append(sts_inv)
      stss_logdet.append(sts_logdet)
      smplsss.append(smplss)
    # save
    np.save(output[0], np.array(stss_inv, dtype=object))
    np.save(output[1], np.array(stss_logdet, dtype=object))
    np.save(output[2], np.array(smplsss, dtype=object))

rule process_shared_times:
  input:
    input_func(processed_shared_times,ends = ends)


# ------------- process coalescence times ----------
coal = "data/SGDP_v1_annot_ne_popsize.coal"
processed_coal_times = coal_times.replace('_cts.npy','_cts_{tCutoff}tCutoff_{end}.npy')
ends = ['bts','lpcs']

rule process_coal_time:
  input:
    coal_times,
    coal
  output:
    expand(processed_coal_times, end=ends, allow_missing=True)
  threads: 1 
  run:
    import os
    # taming numpy
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["GOTO_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    import numpy as np
    from spacetrees import _log_coal_density
    # get variable Nee
    epochs = np.genfromtxt(input[1], skip_header=1,skip_footer = 1) #time at which each epoch starts (and the final one ends)
    Nes = 0.5/np.genfromtxt(input[1], skip_header=2)[2:] #effective population size during each epoch
    # process coal times
    ctss = np.load(input[0]) #coalescence times in ascending order, for each tree
    btss = []
    lpcs = []
    tCutoff = wildcards.tCutoff
    if tCutoff=='None': 
      tCutoff=None 
    else:
      tCutoff=float(tCutoff)
    for cts in ctss: 
      # get branching times in ascending order
      T = cts[-1] #TMRCA (to be replaced with tCutoff)
      if tCutoff is not None:
        if tCutoff < T:
          T = tCutoff
      bts = T - np.flip(cts) #branching times, in ascending order
      bts = bts[bts>0] #remove branching times at or before T
      bts = np.append(bts,T) #append total time as last item
      btss.append(bts)
      # get probability of coalescence times under panmictic coalescent with variable Ne
      lpc = _log_coal_density(times=cts, Nes=Nes, epochs=epochs, tCutoff=tCutoff)
      lpcs.append(lpc)
    # save
    np.save(output[0], np.array(btss, dtype=object))
    np.save(output[1], np.array(lpcs, dtype=object))

rule process_coal_times:
  input:
    input_func(processed_coal_times,ends=ends)
    
    
 # ------------- composite dispersal rates -------------

composite_dispersal_rate = processed_shared_times.replace('_chr{CHR}','').replace('_{start}-{stop}bps','').replace('_sts_{tCutoff}tCutoff_{end}.npy','_{tCutoff}tCutoff_mle-dispersal.npy')


rule composite_dispersal_rates:
  input:
   expand(composite_dispersal_rate, M=Ms, tCutoff=tCutoffs) 

#We need to load start, end, and CHR everytime we used the list of files, because the filename contains wildcards (see how they are named above, in anc and processed_coal_times... files)
def input_func_dispersal(name, ends):
  def input_files(wildcards):
    filenames = []
    for CHR in CHRS:
      bpfile = checkpoints.get_bp.get(CHR=CHR,**wildcards).output[0] #give the start & end numbers in the bps file, for later start & stop loading
      with open(bpfile,'r') as f:
        for line in f:
          start,stop = line.strip().split(' ')
          d = {'{CHR}': CHR, '{start}': start, '{stop}': stop} #load those 3 wildcards in the dictionary
          string = name
          for i,j in d.items():
            string = string.replace(i,str(j)) #replace {CHR} to the exact CHR number
          filenames.append(string)
    return expand(filenames, end=ends, allow_missing=True) 
  return input_files

rule composite_dispersal_rate:
  input:
    stss_mc_inv = input_func_dispersal(processed_shared_times, ['mc-invs']),
    stss_logdet = input_func_dispersal(processed_shared_times, ['logdets']),
    smplss = input_func_dispersal(processed_shared_times, ['samples']),
    btss = input_func_dispersal(processed_coal_times, ['bts']),
    lpcss = input_func_dispersal(processed_coal_times, ['lpcs']),
    locs = expand(locations, CHR=CHRS, allow_missing=True)[0]
  output:
    composite_dispersal_rate
  threads: 80
  resources:
    runtime = 60 * 24
  run:
    import os
    # taming numpy
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["GOTO_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    import numpy as np
    from spacetrees import mle_dispersal, _sds_rho_to_sigma
    from tqdm import tqdm
    # load locations
    locations = np.load(input.locs)
    locations = np.repeat(locations,2,axis =0) #need two locations for haploids
    # subsample for testing
    L = len(input.stss_mc_inv)
    M = int(wildcards.M)
    #L = 10 #number of loci
    #M = 10 #number of trees per locus
    #mean centered and inverted shared time matrices
    print('\nloading inverted shared times matrices')
    stss_mc_inv = []
    for f in tqdm(input.stss_mc_inv[:L]):
      sts_mc_inv = np.load(f, allow_pickle=True)[:M]
      stss_mc_inv.append(sts_mc_inv)
    #log determinants of mean centered shared time matrices    
    print('\nloading log determinants of shared times matrices')
    stss_logdet = []
    for f in tqdm(input.stss_logdet[:L]):
      sts_logdet = np.load(f, allow_pickle=True)[:M]
      stss_logdet.append(sts_logdet) 
    #subtree sampless    
    print('\nloading samples of shared times matrices')
    smplss = []
    for f in tqdm(input.smplss[:L]):
      smpls = np.load(f, allow_pickle=True)[:M]
      smplss.append(smpls) 
    #branching times
    print('\nloading branching times')
    btss = []
    for f in tqdm(input.btss[:L]):
      bts = np.load(f, allow_pickle=True)[:M] 
      btss.append(bts)
    #log probability of coalescent times    
    print('\nloading log probability of coalescence times')
    lpcss = []
    for f in tqdm(input.lpcss[:L]):
      lpcs = np.load(f, allow_pickle=True)[:M] 
      lpcss.append(lpcs)
    # function for updates
    def callbackF(x):
    #  print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}'.format(x[0], x[1], x[2])) #if important=False
      print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(x[0], x[1], x[2], x[3]))
    # find parameter estimates
    print('\nestimating dispersal rate')
    mle = mle_dispersal(locations=locations, shared_times_inverted=stss_mc_inv, log_det_shared_times=stss_logdet, samples=smplss, 
                        sigma0=None, phi0=None, #make educated guess based on first tree at each locus
                        #sigma0=_sds_rho_to_sigma(0.07, 0.06, 0.5), phi0=5e-5, #guess from tCutoff=1e6 mle
                        callbackF=callbackF, 
                        important=True, branching_times=btss, logpcoals=lpcss)
    print('\n',mle)
    np.save(output[0], mle)   
    
# ------------- ancestor locations using full trees -------------
ancestor_locations_full = processed_shared_times.replace('_sts','_10000+_generation').replace('{end}','anc-locs_full-trees_{sample}sample')

def input_func_locs(name):

  def input_files(wildcards):
    filenames = []
    for CHR in CHRS:
      infile = checkpoints.get_bp.get(CHR=CHR).output[0]
      with open(infile,'r') as f:
        for line in f:
          i,j = line.strip().split(' ')
          d = {'{CHR}': CHR, '{start}': i, '{stop}': j}
          string = name
          for i,j in d.items():
            string = string.replace(i,str(j))
          filenames.append(string)
    return expand(filenames, M=Ms, tCutoff=tCutoffs,sample=samples)

  return input_files

rule locate_ancestors_full:
  input:
    input_func_locs(ancestor_locations_full)

ruleorder: process_shared_time > locate_ancestor_full
ruleorder: process_coal_time > locate_ancestor_full

rule locate_ancestor_full:
  input:
    stss = shared_times,
    locs = locations,
    mle = composite_dispersal_rate,
    btss = processed_coal_times.replace('{end}','bts'),
    lpcs = processed_coal_times.replace('{end}','lpcs'),
  output:
    ancestor_locations_full 
  threads: 1 #get seg fault if >1
  resources:
    runtime = 15
  run:
    import os
    # taming numpy
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["GOTO_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    
    import numpy as np
    from spacetrees import _sds_rho_to_sigma, _log_birth_density, locate_ancestors
    from utils import chop_shared_times

    # sample locations
    locations = np.load(input.locs)
    
    locations = np.repeat(locations,2,axis =0) #need two locations for haploids
    n = len(locations)

    # who and when
    ancestor_samples = [int(wildcards.sample)] #find 0 sample
    #ancestor_samples = range(n) #which samples to find ancestors of
    tCutoff = wildcards.tCutoff #determines which dispersal rate we use and how far back we locate ancestors
 
    ancestor_times = np.logspace(np.log10(1e4),np.log10(1e5),5)[1:3] #times to find ancestors
    #ancestor_times = np.logspace(1, np.log10(1e4),10) #times to find ancestors
    #ancestor_times = np.linspace(1e3,1e4,10)

    # chop trees
    stss_chopped = []
    samples = []
    for sts in np.load(input.stss):
      sts_chopped, smpls = chop_shared_times(sts, tCutoff=None) #shared times and sample indices in each subtree (here just 1 subtree per tree)
      stss_chopped.append(sts_chopped)
      samples.append(smpls)
    
    #dispersal rate
    mle = np.load(input.mle) #mle dispersal rate and branching rate
    sigma = _sds_rho_to_sigma(mle[0:3]) #as covariance matrix

    # importance weights
    btss = np.load(input.btss, allow_pickle=True) #birth times
    phi = mle[-1] #mle branching rate
    lbds = np.array([_log_birth_density(bts, phi, n) for bts in btss]) #log probability densities of birth times
    lpcs = np.load(input.lpcs, allow_pickle=True) #log probability densities of coalescence times
    log_weights = lbds - lpcs #log importance weights
 
    # locate 
    ancestor_locations = locate_ancestors(ancestor_samples, ancestor_times, stss_chopped, samples, locations, log_weights, sigma) #switch parameters
    np.save(output[0], ancestor_locations)
# taking about 16m each with 1 thread
# snakemake locate_ancestors_full --profile slurm --groups locate_ancestor_full=locate --group-components locate=80 --jobs 100


# ------------------- New Location (projected) ---------------------

 # ------------- composite dispersal rates -------------

projected_composite_dispersal_rate = processed_shared_times.replace('_chr{CHR}','').replace('_{start}-{stop}bps','').replace('_sts_{tCutoff}tCutoff_{end}.npy','_{tCutoff}tCutoff_projected_mle-dispersal.npy')


rule projected_composite_dispersal_rates:
  input:
   expand(projected_composite_dispersal_rate, M=Ms, tCutoff=tCutoffs) 

#We need to load start, end, and CHR everytime we used the list of files, because the filename contains wildcards (see how they are named above, in anc and processed_coal_times... files)
def input_func_dispersal(name, ends):
  def input_files(wildcards):
    filenames = []
    for CHR in CHRS:
      bpfile = checkpoints.get_bp.get(CHR=CHR,**wildcards).output[0] #give the start & end numbers in the bps file, for later start & stop loading
      with open(bpfile,'r') as f:
        for line in f:
          start,stop = line.strip().split(' ')
          d = {'{CHR}': CHR, '{start}': start, '{stop}': stop} #load those 3 wildcards in the dictionary
          string = name
          for i,j in d.items():
            string = string.replace(i,str(j)) #replace {CHR} to the exact CHR number
          filenames.append(string)
    return expand(filenames, end=ends, allow_missing=True) 
  return input_files

rule projected_composite_dispersal_rate:
  input:
    stss_mc_inv = input_func_dispersal(processed_shared_times, ['mc-invs']),
    stss_logdet = input_func_dispersal(processed_shared_times, ['logdets']),
    smplss = input_func_dispersal(processed_shared_times, ['samples']),
    btss = input_func_dispersal(processed_coal_times, ['bts']),
    lpcss = input_func_dispersal(processed_coal_times, ['lpcs']),
    locs = expand(projected_locations, CHR=CHRS, allow_missing=True)[0]
  output:
    projected_composite_dispersal_rate
  threads: 80
  resources:
    runtime = 60 * 24
  run:
    import os
    # taming numpy
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["GOTO_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    import numpy as np
    from spacetrees import mle_dispersal, _sds_rho_to_sigma
    from tqdm import tqdm
    # load locations
    locations = np.load(input.locs)
    locations = np.repeat(locations,2,axis =0) #need two locations for haploids
    # subsample for testing
    L = len(input.stss_mc_inv)
    M = int(wildcards.M)
    #L = 10 #number of loci
    #M = 10 #number of trees per locus
    #mean centered and inverted shared time matrices
    print('\nloading inverted shared times matrices')
    stss_mc_inv = []
    for f in tqdm(input.stss_mc_inv[:L]):
      sts_mc_inv = np.load(f, allow_pickle=True)[:M]
      stss_mc_inv.append(sts_mc_inv)
    #log determinants of mean centered shared time matrices    
    print('\nloading log determinants of shared times matrices')
    stss_logdet = []
    for f in tqdm(input.stss_logdet[:L]):
      sts_logdet = np.load(f, allow_pickle=True)[:M]
      stss_logdet.append(sts_logdet) 
    #subtree sampless    
    print('\nloading samples of shared times matrices')
    smplss = []
    for f in tqdm(input.smplss[:L]):
      smpls = np.load(f, allow_pickle=True)[:M]
      smplss.append(smpls) 
    #branching times
    print('\nloading branching times')
    btss = []
    for f in tqdm(input.btss[:L]):
      bts = np.load(f, allow_pickle=True)[:M] 
      btss.append(bts)
    #log probability of coalescent times    
    print('\nloading log probability of coalescence times')
    lpcss = []
    for f in tqdm(input.lpcss[:L]):
      lpcs = np.load(f, allow_pickle=True)[:M] 
      lpcss.append(lpcs)
    # function for updates
    def callbackF(x):
    #  print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}'.format(x[0], x[1], x[2])) #if important=False
      print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(x[0], x[1], x[2], x[3]))
    # find parameter estimates
    print('\nestimating dispersal rate')
    mle = mle_dispersal(locations=locations, shared_times_inverted=stss_mc_inv, log_det_shared_times=stss_logdet, samples=smplss, 
                        sigma0=None, phi0=None, #make educated guess based on first tree at each locus
                        #sigma0=_sds_rho_to_sigma(0.07, 0.06, 0.5), phi0=5e-5, #guess from tCutoff=1e6 mle
                        callbackF=callbackF, 
                        important=True, branching_times=btss, logpcoals=lpcss)
    print('\n',mle)
    np.save(output[0], mle)   

# ------------- ancestor generation locations using full trees -------------
projected_ancestor_locations_full = processed_shared_times.replace('_sts','').replace('{end}','anc-locs_full-trees_{sample}sample_projected')

def input_func_locs(name):

  def input_files(wildcards):
    filenames = []
    for CHR in CHRS:
      infile = checkpoints.get_bp.get(CHR=CHR).output[0]
      with open(infile,'r') as f:
        for line in f:
          i,j = line.strip().split(' ')
          d = {'{CHR}': CHR, '{start}': i, '{stop}': j}
          string = name
          for i,j in d.items():
            string = string.replace(i,str(j))
          filenames.append(string)
    return expand(filenames, M=Ms, tCutoff=tCutoffs,sample=samples)

  return input_files

rule projected_locate_ancestors_full:
  input:
    input_func_locs(projected_ancestor_locations_full)

ruleorder: process_shared_time > projected_locate_ancestors_full
ruleorder: process_coal_time > projected_locate_ancestors_full

rule projected_locate_ancestor_full:
  input:
    stss = shared_times,
    locs = projected_locations,
    mle = projected_composite_dispersal_rate,
    btss = processed_coal_times.replace('{end}','bts'),
    lpcs = processed_coal_times.replace('{end}','lpcs'),
  output:
    projected_ancestor_locations_full 
  threads: 1 #get seg fault if >1
  resources:
    runtime = 60
  run:
    import os
    # taming numpy
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["GOTO_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    
    import numpy as np
    from spacetrees import _sds_rho_to_sigma, _log_birth_density, locate_ancestors
    from utils import chop_shared_times

    # sample locations
    locations = np.load(input.locs)
    n = len(locations)

    # who and when
    ancestor_samples = [int(wildcards.sample)] #find 0 sample
    #ancestor_samples = range(n) #which samples to find ancestors of
    tCutoff = wildcards.tCutoff #determines which dispersal rate we use and how far back we locate ancestors
 
    times1 = np.logspace(1, np.log10(1e4), 10)         # 10 points from 10 to 10000
    times2 = np.logspace(np.log10(1e4), np.log10(1e5), 5)[1:3]  # select 2 points between 1e4 and 1e5
    ancestor_times = np.concatenate([times1, times2])

    # chop trees
    stss_chopped = []
    samples = []
    for sts in np.load(input.stss):
      sts_chopped, smpls = chop_shared_times(sts, tCutoff=None) #shared times and sample indices in each subtree (here just 1 subtree per tree)
      stss_chopped.append(sts_chopped)
      samples.append(smpls)
    
    #dispersal rate
    mle = np.load(input.mle) #mle dispersal rate and branching rate
    sigma = _sds_rho_to_sigma(mle[0:3]) #as covariance matrix

    # importance weights
    btss = np.load(input.btss, allow_pickle=True) #birth times
    print(np.max(btss))
    phi = mle[-1] #mle branching rate
    lbds = np.array([_log_birth_density(bts, phi, n) for bts in btss]) #log probability densities of birth times
    lpcs = np.load(input.lpcs, allow_pickle=True) #log probability densities of coalescence times
    log_weights = lbds - lpcs #log importance weights
 
    # locate 
    ancestor_locations = locate_ancestors(ancestor_samples, ancestor_times, stss_chopped, samples, locations, log_weights, sigma) #switch parameters
    np.save(output[0], ancestor_locations)
# taking about 16m each with 1 thread
# snakemake projected_locate_ancestors_full --profile slurm --groups projected_locate_ancestor_full=locate --group-components locate=50 --jobs 100

