#Usage: python calc.py
#
#This script...
#
#Arguments:
# warnings-file[path]: file to output warnings to; default: standard error
#these are for debugging
#import cProfile
#import resource

#import urllib.request #used (below) only if you specify a URL as an input file
#import requests #used (below) only if you specify --lmm-auth-key

import optparse
import sys
import os
import copy
import scipy
import scipy.sparse as sparse
import scipy.stats
from scipy.stats import truncnorm, norm
from scipy.special import psi as digamma
from scipy.special import erfc
import numpy as np
from numpy.random import gamma
from numpy.random import normal
from numpy.random import exponential
import itertools
import gzip
import random

random.seed(0)

def bail(message):
    sys.stderr.write("%s\n" % (message))
    sys.exit(1)

usage = "usage: priors.py [beta_tildes|sigma|betas|priors|gibbs|factor] [options]"

def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

parser = optparse.OptionParser(usage)
#gene x gene_set matrix
#each specification of these files is a different batch
#can use "," to group multiple files or lists within each --X
#can combine into batches with "@{batch_id}" after the file/list
#by default, the same @{batch_id} is appended to a list, which meansit will be appended to all files in the list that do not already have a batch
#this can be overriden by specifying batches to files within the list
#these batches are used for parameter learning (see below)
parser.add_option("","--X-in",action="append",default=None)
parser.add_option("","--X-list",action="append",default=None)
parser.add_option("","--Xd-in",action="append",default=None)
parser.add_option("","--Xd-list",action="append",default=None)
parser.add_option("","--X-out",default=None)
parser.add_option("","--Xd-out",default=None)
parser.add_option("","--batch-separator",default="@") #separator for batches
parser.add_option("","--file-separator",default=None) #separator for multiple files

#model parameters
parser.add_option("","--p-noninf",type=float,default=0.001) #initial parameter for p
parser.add_option("","--top-gene-set-prior",type=float,default=None) #specify the top prior efect we are expecting any of the gene sets to have (after all of the calculations). Precedence 1
parser.add_option("","--num-gene-sets-for-prior",type=int,default=None) #specify the top prior efect we are expecting any of the gene sets to have (after all of the calculations) Precedence 1
parser.add_option("","--sigma2-ext",type=float,default=None) #specify sigma in external units. Precedence 2
parser.add_option("","--sigma2",type=float,default=None) #specify sigma in internal units (this is what the code outputs to --sigma-out). Precedence 3
parser.add_option("","--sigma2-cond",type=float,default=1e-5) #specify conditional sigma value (sigma/p). Precedence 4
parser.add_option("","--sigma-power",type='float',default=-4) #multiply sigma times np.power(scale_factors,sigma_power). 2=const_sigma, 0=default. Larger values weight larger gene sets more
parser.add_option("","--sigma-soft-threshold-95",type='float',default=None) #the gene set size at which threshold is 0.95
parser.add_option("","--sigma-soft-threshold-5",type='float',default=None) #the gene set size at which threshold is 0.05


parser.add_option("","--const-sigma",action='store_true') #assign constant variance across all gene sets independent of size (default is to scale inversely to size). Overrides sigma power and sets it to 2

parser.add_option("","--update-hyper",type='string',default="both",dest="update_hyper") #update either both,p,sigma,none
parser.add_option("","--sigma-num-devs-to-top",default=2.0,type=float) #update p by learning it for each batch
parser.add_option("","--p-noninf-inflate",default=1.0,type=float) #update p by multiplying it by this each time you learn it

parser.add_option("","--batch-all-for-hyper",action="store_true") #combine everything into one batch for learning hyper
parser.add_option("","--first-for-hyper",action="store_true") #use first batch / dataset (that is, the batch of the first --X; may include other files too) to learn parameters for unlabelled batches (label batches with "@{batch_id}" as above
parser.add_option("","--first-for-sigma-cond",action="store_true") #use first batch to fix sigma/p ratio and use that for all other batches. 

parser.add_option("","--background-prior",type=float,default=0.05) #specify background prior

#correlation matrix (otherwise will be calculated from X)
parser.add_option("","--V-in",default=None)
parser.add_option("","--V-out",default=None)
parser.add_option("","--shrink-mat-out",default=None)

#optional gene name map
parser.add_option("","--gene-map-in",default=None)
parser.add_option("","--gene-map-orig-gene-col",default=1) #1-based column for original gene
parser.add_option("","--gene-map-new-gene-col",default=2) #1-based column for original gene

#GWAS association statistics (for HuGECalc)
parser.add_option("","--gwas-in",default=None)
parser.add_option("","--gwas-locus-col",default=None)
parser.add_option("","--gwas-chrom-col",default=None)
parser.add_option("","--gwas-pos-col",default=None)
parser.add_option("","--gwas-p-col",default=None)
parser.add_option("","--gwas-beta-col",default=None)
parser.add_option("","--gwas-se-col",default=None)
parser.add_option("","--gwas-units",type=float,default=None)
parser.add_option("","--gwas-n-col",default=None)
parser.add_option("","--gwas-n",type='float',default=None)
parser.add_option("","--gwas-freq-col",default=None)
parser.add_option("","--gwas-ignore-p-threshold",type=float,default=None) #completely ignore anything with p above this threshold

#credible sets
parser.add_option("","--credible-sets-in",default=None) #pass in credible sets to use 
parser.add_option("","--credible-sets-id-col",default=None)
parser.add_option("","--credible-sets-chrom-col",default=None)
parser.add_option("","--credible-sets-pos-col",default=None)
parser.add_option("","--credible-sets-ppa-col",default=None)

#S2G values (for HuGeCalc)
parser.add_option("","--s2g-in",default=None)
parser.add_option("","--s2g-chrom-col",default=None)
parser.add_option("","--s2g-pos-col",default=None)
parser.add_option("","--s2g-gene-col",default=None)
parser.add_option("","--s2g-prob-col",default=None)

#Exomes association statistics (for HuGeCalc)
parser.add_option("","--exomes-in",default=None)
parser.add_option("","--exomes-gene-col",default=None)
parser.add_option("","--exomes-p-col",default=None)
parser.add_option("","--exomes-beta-col",default=None)
parser.add_option("","--exomes-se-col",default=None)
parser.add_option("","--exomes-units",type=float,default=None)
parser.add_option("","--exomes-n-col",default=None)
parser.add_option("","--exomes-n",type='float',default=None)

#Positive control genes
parser.add_option("","--positive-controls-in",default=None)
parser.add_option("","--positive-controls-id-col",default=None)
parser.add_option("","--positive-controls-prob-col",default=None)

#association statistics for gene bfs in each gene set (if precomputed)
parser.add_option("","--gene-set-stats-in",default=None)
parser.add_option("","--gene-set-stats-id-col",default="Gene_Set")
parser.add_option("","--gene-set-stats-exp-beta-tilde-col",default=None)
parser.add_option("","--gene-set-stats-beta-tilde-col",default=None)
parser.add_option("","--gene-set-stats-beta-col",default=None)
parser.add_option("","--gene-set-stats-beta-uncorrected-col",default=None)
parser.add_option("","--gene-set-stats-se-col",default=None)
parser.add_option("","--gene-set-stats-p-col",default=None)
parser.add_option("","--ignore-negative-exp-beta",action='store_true')

#if you have gene set betas
parser.add_option("","--gene-set-betas-in",default=None)

#gene BFs to use in calculating gene set statistics
parser.add_option("","--gene-bfs-in",default=None)
parser.add_option("","--gene-stats-in",dest="gene_bfs_in",default=None)
parser.add_option("","--gene-bfs-id-col",default=None)
parser.add_option("","--gene-bfs-log-bf-col",default=None)
parser.add_option("","--gene-bfs-combined-col",default=None)
parser.add_option("","--gene-bfs-prior-col",default=None)

#gene percentiles to use in calculating gene set statistics. Will be converted to BFs using inverse normal function
parser.add_option("","--gene-percentiles-in",default=None)
parser.add_option("","--gene-percentiles-id-col",default=None)
parser.add_option("","--gene-percentiles-value-col",default=None)
parser.add_option("","--gene-percentiles-higher-is-better",default=False,action='store_true')
parser.add_option("","--gene-zs-in",default=None)
parser.add_option("","--gene-zs-id-col",default=None)
parser.add_option("","--gene-zs-value-col",default=None)

#locations of genes
parser.add_option("","--gene-loc-file",default=None)
parser.add_option("","--gene-loc-file-huge",default=None)
parser.add_option("","--exons-loc-file-huge",default=None)
parser.add_option("","--gene-cor-file",default=None)
parser.add_option("","--gene-cor-file-gene-col",type=int,default=1)
parser.add_option("","--gene-cor-file-cor-start-col",type=int,default=10)
parser.add_option("","--ols",action='store_true')
parser.add_option("","--gls",action='store_true')

#output files for stats
parser.add_option("","--gene-set-stats-out",default=None)
parser.add_option("","--gene-set-stats-trace-out",default=None)
parser.add_option("","--betas-trace-out",default=None)
parser.add_option("","--gene-stats-out",default=None)
parser.add_option("","--gene-stats-trace-out",default=None)
parser.add_option("","--gene-gene-set-stats-out",default=None)
parser.add_option("","--gene-set-overlap-stats-out",default=None)
parser.add_option("","--gene-covs-out",default=None)
parser.add_option("","--gene-effectors-out",default=None)
parser.add_option("","--factors-out",default=None)
parser.add_option("","--marker-factors-out",default=None)
parser.add_option("","--gene-set-factors-out",default=None)
parser.add_option("","--gene-factors-out",default=None)
parser.add_option("","--gene-set-clusters-out",default=None)
parser.add_option("","--gene-clusters-out",default=None)

#output for parameters
parser.add_option("","--params-out",default=None)

#control output / logging
parser.add_option("","--log-file",default=None)
parser.add_option("","--warnings-file",default=None)
parser.add_option("","--debug-level",type='int',default=None)
parser.add_option("","--hide-progress",default=False,action='store_true')
parser.add_option("","--hide-opts",default=False,action='store_true')

#other control
parser.add_option("","--hold-out-chrom",type="string",default=None) #don't use this chromosome for input values (infer only priors, based on other chromosomes)

#parameters for controlling efficiency
#split genes into batches for calculating final statistics via cross-validation
parser.add_option("","--priors-num-gene-batches",type="int",default=20)
parser.add_option("","--gibbs-num-batches-parallel",type="int",default=10)
parser.add_option("","--gibbs-max-mb-X-h",type="int",default=100)
parser.add_option("","--batch-size",type=int,default=5000) #maximum number of dense X columns to hold in memory at once
parser.add_option("","--pre-filter-batch-size",type=int,default=None) #if more than this number of gene sets are about to go into non inf betas, do pre-filters on smaller batches. Assumes smaller batches will only have higher betas than full batches
parser.add_option("","--pre-filter-small-batch-size",type=int,default=500) #the limit to use for the smaller pre-filtering batches
parser.add_option("","--max-allowed-batch-correlation",type=float,default=0.5) #technically we need to update each gene set sequentially during sampling; for efficiency, group those for simultaneous updates that have max_allowed_batch_correlation below this threshold
parser.add_option("","--no-initial-linear-filter",default=True,action="store_false",dest="initial_linear_filter") #within gibbs sampling, first run a linear regression to remove non-associated gene sets (reducing number that require full logistic regression)

#parameters for filtering gene sets
parser.add_option("","--min-gene-set-size",type=int,default=10) #ignore genes with fewer genes than this (after removing for other reasons)
parser.add_option("","--filter-gene-set-p",type=float,default=0.005) #gene sets with p above this are never seen. If this is above --max-gene-set-p, then it will be lowered to match --max-gene-set-p
parser.add_option("","--increase-filter-gene-set-p",type=float,default=0.01) #require at least this fraction of gene sets to be kept from each file
parser.add_option("","--max-num-gene-sets",type=int,default=None) #ignore gene sets to reduce to this number. Begin with pruning, and then lower the --filter-gene-p-threshold
parser.add_option("","--min-num-gene-sets",type=int,default=100) #increase filter_gene_set_p as needed to achieve this number of gene sets
parser.add_option("","--filter-gene-set-metric-z",type=float,default=2.5) #gene sets with combined outlier metric z-score above this threshold are never seen (must have correct-huge turned on for this to work)
parser.add_option("","--max-gene-set-p",type=float,default=.05) #gene sets with p above this are excluded from the original beta analysis but included in gibbs
parser.add_option("","--min-gene-set-beta",type=float,default=1e-20) #gene sets with beta below this are excluded from reading in the gene stats file
parser.add_option("","--min-gene-set-beta-uncorrected",type=float,default=1e-20) #gene sets with beta below this are excluded from reading in the gene stats file
parser.add_option("","--x-sparsify",type="string",action="callback",callback=get_comma_separated_args,default=[50,100,1000]) #applies to continuous gene sets, which are converted to dichotomous gene sets internally. For each value N, generate a new dichotomous gene set with the most N extreme genes (see next three options)
parser.add_option("","--add-ext",default=False,action="store_true") #add the top and bottom extremes as a gene set
parser.add_option("","--no-add-top",default=True,action="store_false",dest="add_top") #add the top extremes as a gene set
parser.add_option("","--no-add-bottom",default=True,action="store_false",dest="add_bottom") #add the bottom extremes as a gene set
parser.add_option("","--no-filter-negative",default=True,action="store_false",dest="filter_negative") #after sparsifying, remove any gene sets with negative beta tilde (under assumption that we added the "wrong" extreme)

parser.add_option("","--threshold-weights",type='float',default=0.5) #weights below this fraction of top weight are set to 0
parser.add_option("","--max-gene-set-size",type=int,default=30000)
parser.add_option("","--prune-gene-sets",type=float,default=0.8) #gene sets with correlation above this threshold with any other gene set are removed (smallest gene set in correlation is retained)


#parameters for learning sigma
parser.add_option("","--chisq-dynamic",action="store_true") #dynamically determine the chisq threshold based on intercept and sigma
parser.add_option("","--desired-intercept-difference",type=float,default=1.3) #if dynamically determining chisq threshold, stop when intercept is less than this far away from 1
parser.add_option("","--chisq-threshold",type="float",default=5) #threshold outlier gene sets during sigma computation


#gene percentile parameters
parser.add_option("","--top-posterior",type=float,default=0.99) #specify maximum posterior, used in inverse normal conversion of percentile to posterior
parser.add_option("","--gws-threshold",type=float,default=2.5e-6) #specify significance threshold for genes to use for mapping to gws-posterior
parser.add_option("","--gws-prob-true",type=float,default=0.95) #specify probability genes at the significance threshold are true associations
#gene Z parameters
parser.add_option("","--max-mean-posterior",type=float,default=0.2) #specify significance threshold for genes to use for mapping to max-mean-posterior

#huge exomes parametersa
parser.add_option("","--exomes-high-p",type=float,default=5e-2) #specify the larger p-threshold for which we will constrain posterior
parser.add_option("","--exomes-high-p-posterior",type=float,default=0.1) #specify the posterior at the larger p-threshold
parser.add_option("","--exomes-low-p",type=float,default=2.5e-6) #specify the smaller p-threshold for which we will constrain posterior 
parser.add_option("","--exomes-low-p-posterior",type=float,default=0.95) #specify the posterior at the smaller p-threshold

#huge gwas parametersa
parser.add_option("","--gwas-high-p",type=float,default=1e-2) #specify the larger p-threshold for which we will constrain posterior
parser.add_option("","--gwas-high-p-posterior",type=float,default=0.01) #specify the posterior at the larger p-threshold
parser.add_option("","--gwas-low-p",type=float,default=5e-8) #specify the smaller p-threshold for which we will constrain posterior 
parser.add_option("","--gwas-low-p-posterior",type=float,default=0.75) #specify the posterior at the smaller p-threshold
parser.add_option("","--gwas-detect-low-power",type=int,default=None) #scale --gwas-low-p automatically to have at least this number signals reaching it; set to 0 to disable this
parser.add_option("","--gwas-detect-high-power",type=int,default=None) #scale --gwas-low-p automatically to have no more than this number of signals reaching it; set to a very high number to disable
parser.add_option("","--gwas-detect-adjust-huge",action="store_true") #by default, --gwas-detect-power will only affect prior calculations; enable this to detect for all
parser.add_option("","--learn-window",default=False,action='store_true') #learn the window function linking SNPs to genes based on empirical distances of SNPs to genes and the --closest-gene-prob
parser.add_option("","--min-var-posterior",type=float,default=0.01) #exclude all variants with posterior below this; this uses the default parameters before detect low power
parser.add_option("","--closest-gene-prob",type=float,default=0.7) #specify probability that closest gene is the causal gene
#these control how the probability of a SNP to gene link is scaled, independently of how many genes there are nearby
parser.add_option("","--no-scale-raw-closest-gene",default=True,action='store_false',dest="scale_raw_closest_gene") #scale_raw_closest_gene: set everything to have the closest gene as closest gene prob (shifting up or down as necessary) 
parser.add_option("","--cap-raw-closest-gene",default=False,action='store_true') #cap_raw_closest_gene: set everything to have probability no greater than closest gene prob (shifting down but not up)
parser.add_option("","--max-closest-gene-prob",type=float,default=0.9) #specify maximum probability that closest gene is the causal gene. This accounts for probability that gene might just lie very far from the window
parser.add_option("","--max-closest-gene-dist",type=float,default=2.5e5) #the maximum distance for which we will search for the closest gene
#these parameters control how all genes nearby a signal are scaled
parser.add_option("","--no-cap-region-posterior",default=True,action='store_false',dest="cap_region_posterior") #ensure that the sum of gene probabilities is no more than 1
parser.add_option("","--scale-region-posterior",default=False,action='store_true') #ensure that the sum of gene probabilities is always 1
parser.add_option("","--phantom-region-posterior",default=False,action='store_true') #if the sum of gene probabilities is less than 1, assign the rest to a "phantom" gene that always has prior=0.05. As priors change for the other genes, they will "eat up" some of the phantom gene's assigned probability
parser.add_option("","--allow-evidence-of-absence",default=False,action='store_true') #allow the posteriors of genes to decrease below the background if there is a lack of GWAS signals
parser.add_option("","--no-correct-huge",default=True,action='store_false',dest="correct_huge") #don't correct huge scores for confounding variables.
parser.add_option("","--no-correct-betas",default=True,action='store_false',dest="correct_betas") #don't correct gene set variables for confounding variables (which still may exist even if all genes are corrected)

parser.add_option("","--min-n-ratio",type=float,default=0.5) #ignore SNPs with sample size less than this ratio of the max
parser.add_option("","--max-clump-ld",type=float,default=0.5) #maximum ld threshold to use for clumping (when MAF is passed in)
parser.add_option("","--signal-window-size",type=float,default=250000) #window size to initially include variants in a signal
parser.add_option("","--signal-min-sep",type=float,default=100000) #extend the region until the distance to the last significant snp is greater than the signal_min_sep
parser.add_option("","--signal-max-logp-ratio",type=float,default=None) #ignore all variants that are this ratio below max in signal
parser.add_option("","--credible-set-span",type=float,default=25000) #if user specified credible sets, ignore all variants within this var of a variant in the credible set

#sampling parameters
parser.add_option("","--max-num-burn-in",type=int,default=None) #maximum number of burn initerations to run

#sparsity parameters
parser.add_option("","--no-sparse-solution",default=True,action="store_false",dest="sparse_solution") #zero out betas with small p_bar
parser.add_option("","--sparse-frac-gibbs",default=0.001,type=float) #zero out betas with with values below this fraction of the top; within the gibbs loop
parser.add_option("","--sparse-frac-betas",default=0.001,type=float) #zero out betas with with values below this fraction of the top

#gibbs parameters
parser.add_option("","--no-update-huge-scores",default=True,action='store_false',dest="update_huge_scores") #do not use priors to update huge scores (by default, priors affect "competition" for signal by nearby genes")
parser.add_option("","--top-gene-prior",type=float,default=None) #specify the top prior we are expecting any of the genes to have (after all of the calculations)
parser.add_option("","--increase-hyper-if-betas-below",type=float,default=None) #increase p if gene sets aren't significant enough

#factor parameters
parser.add_option("","--lmm-auth-key",default=None,type=str) #pass authorization key to enable LLM cluster labelling
parser.add_option("","--max-num-factors",default=15,type=int) #maximum k for factorization
parser.add_option("","--alpha0",default=10,type=float) #alpha prior on lambda k for factorization (larger makes more sparse)
parser.add_option("","--beta0",default=1,type=float) #beta prior on lambda k for factorization
parser.add_option("","--gene-set-filter-type",default="beta_uncorrected") #choose type of filter on gene sets; beta|beta_uncorrected|p|none
parser.add_option("","--gene-set-filter-value",type=float,default=0.01) #choose value of filter for gene sets
parser.add_option("","--gene-filter-type",default="combined")  #choose type of filter on gene sets; prior|combined|log_bf|none
parser.add_option("","--gene-filter-value",type=float,default=0) #choose value of filter for genes
parser.add_option("","--gene-set-multiply-type",default=None) #choose gene set value to multiply matrix by; beta|beta_uncorrected|none
parser.add_option("","--gene-multiply-type",default=None) #choose gene set value to multiply matrix by; prior|combined|log_bf|none
parser.add_option("","--no-transpose",action='store_true') #factor original X rather than tranpose
parser.add_option("","--run-tan",action='store_true') #run Tan et al method

#gibbs sampling parameters
parser.add_option("","--num-mad",type=int,default=10) #number of median absolute devs above which to treat chains as outliers
parser.add_option("","--min-num-iter",type=int,default=10) #minimum number of iterations to run for gibbs loop
parser.add_option("","--max-num-iter",type=int,default=500) #maximum number of iterations to run for gibbs loop
parser.add_option("","--num-chains",type=int,default=10) #number of chains for gibbs sampling
parser.add_option("","--r-threshold-burn-in",type=float,default=1.01) #maximum number of iterations to run for gibbs
parser.add_option("","--gauss-seidel",action="store_true") #run gauss seidel for gibbs sampling
parser.add_option("","--max-frac-sem",type=float,default=0.01) #the minimum z score (mean/sem) to allow after stopping sampling; continue sampling if this is too large
parser.add_option("","--use-sampled-betas-in-gibbs",action="store_true") #use a sample of the betas returned from the inner beta sampling within the gibbs samples; by default uses mean value which is smoother (more stable but more prone to not exploring full space)
parser.add_option("","--use-max-r-for-convergence",action="store_true") #use only the maximum R to evaluate convergence (most conservative). By default uses mean R

#beta sampling parameters
parser.add_option("","--min-num-iter-betas",type=int,default=10) #minimum number of iterations to run for beta sampling
parser.add_option("","--max-num-iter-betas",type=int,default=1100) #maximum number of iterations to run for beta sampling
parser.add_option("","--num-chains-betas",type=int,default=5) #number of chaings for beta sampling
parser.add_option("","--r-threshold-burn-in-betas",type=float,default=1.01) #threshold for R to consider a gene set as converged (that is, stop burn in and start sampling)
parser.add_option("","--gauss-seidel-betas",action="store_true") #run gauss seidel
parser.add_option("","--max-frac-sem-betas",type=float,default=0.01) #the minimum z score (mean/sem) to allow after stopping sampling; continue sampling if this is too large
parser.add_option("","--use-max-r-for-convergence-betas",action="store_true") #use only the maximum R across gene sets to evaluate convergence (most conservative). By default uses mean R


#TEMP DEBUGGING FLAGS
parser.add_option("","--debug-zero-sparse",action="store_true") #
parser.add_option("","--debug-just-check-header",action="store_true") #

(options, args) = parser.parse_args()


try:
    options.x_sparsify = [int(x) for x in options.x_sparsify]
except ValueError:
    bail("option --x-sparsify: invalid integer list %s" % options.x_sparsify)

if len(args) < 1:
    bail(usage)

mode = args[0]

run_huge = False
run_beta_tilde = False
run_sigma = False
run_beta = False
run_priors = False
run_gibbs = False
run_factor = False

if mode == "huge" or mode == "huge_calc":
    run_huge = True
elif mode == "beta_tildes" or mode == "beta_tilde":
    run_beta_tilde = True
elif mode == "sigma":
    run_sigma = True
elif mode == "betas" or mode == "beta":
    run_beta = True
elif mode == "priors" or mode == "prior":
    run_priors = True
elif mode == "gibbs" or mode == "em":
    run_gibbs = True
elif mode == "factor":
    run_factor = True
else:
    bail("Unrecognized mode %s" % mode)

log_fh = None
if options.log_file is not None:
    log_fh = open(options.log_file, 'w')
else:
    log_fh = sys.stderr

NONE=0
INFO=1
DEBUG=2
TRACE=3
debug_level = options.debug_level
if debug_level is None:
    debug_level = INFO
def log(message, level=INFO, end_char='\n'):
    if level <= debug_level:
        log_fh.write("%s%s" % (message, end_char))
        log_fh.flush()

#set up warnings
warnings_fh = None
if options.warnings_file is not None:
    warnings_fh = open(options.warnings_file, 'w')
else:
    warnings_fh = sys.stderr

def warn(message):
    if warnings_fh is not None:
        warnings_fh.write("Warning: %s\n" % message)
        warnings_fh.flush()
    log(message, level=INFO)

def is_gz_file(filepath, is_remote, flag=None):

    if len(filepath) >= 3 and (filepath[-3:] == ".gz" or filepath[-4:] == ".bgz") and (flag is None or 'w' not in flag):
        try:
            with gzip.open(filepath) as test_fh:
                try:
                    test_fh.readline()
                    return True
                except gzip.BadGzipFile:
                    return False
        except FileNotFoundError:
            return True

    else:
        flag = 'rb'
        if is_remote:
            import urllib.request
            test_f = urllib.request.urlopen(filepath)
        else:
            test_f = open(filepath, 'rb')

        is_gz = test_f.read(2) == b'\x1f\x8b'
        test_f.close()
        return is_gz

def open_gz(file, flag=None):
    is_remote = False
    remote_prefixes = ["http:", "https:", "ftp:"]
    for remote_prefix in remote_prefixes:
        if len(file) >= len(remote_prefix) and file[:len(remote_prefix)] == remote_prefix:
            is_remote = True
       
    if is_gz_file(file, is_remote):
        open_fun = gzip.open
        if flag is not None and len(flag) > 0 and not flag[-1] == 't':
            flag = "%st" % flag
        elif flag is None:
            flag = "rt"
    else:
        open_fun = open

    if is_remote:
        import urllib.request
        import io
        if flag is not None:
            if open_fun is open:
                fh = io.TextIOWrapper(urllib.request.urlopen(file, flag))
            else:
                fh = open_fun(urllib.request.urlopen(file), flag)
        else:
            if open_fun is open:
                fh = io.TextIOWrapper(urllib.request.urlopen(file))
            else:
                fh = open_fun(urllib.request.urlopen(file))
    else:
        if flag is not None:
            try:
                fh = open_fun(file, flag, encoding="utf-8")
            except LookupError:
                fh = open_fun(file, flag)
        else:
            try:
                fh = open_fun(file, encoding="utf-8")
            except LookupError:
                fh = open_fun(file)

    return fh

class GeneSetData(object):
    '''
    Stores gene and gene set annotations and derived matrices
    It allows reading X or V files and using these to determine the allowed gene sets and genes
    '''
    def __init__(self, background_prior=0.05, batch_size=4500):

        #empirical mean scale factor from mice
        self.MEAN_MOUSE_SCALE = 0.0448373

        if background_prior <= 0 or background_prior >= 1:
            bail("--background-prior must be in (0,1)")
        self.background_prior = background_prior
        self.background_log_bf = np.log(self.background_prior / (1 - self.background_prior))
        self.background_bf = np.exp(self.background_log_bf)


        #genes x gene set indicator matrix (sparse)
        #this is always the original matrix -- it is never rescaled or shifted
        #but, calculations of beta_tildes etc. are done relative to what would be obtained if it were scaled
        #similarly, when X/y is whitened, the "internal state" of the code is that both X and y are whitened (and scale factors reflect this)
        #but, for efficiency, X is maintained as a sparse matrix
        #so, ideally this should never be accessed directly; instead get_X_orig returns the original (sparse) matrix and makes the intent explicity to avoid any scaling/whitening
        #get_X_blocks returns the (unscaled) but whitened X
        self.X_orig = None
        #these are genes that we want to calculate priors for but which don't have gene-level statistics
        self.X_orig_missing_genes = None
        self.X_orig_missing_genes_missing_gene_sets = None
        self.X_orig_missing_gene_sets = None
        #internal cache
        self.last_X_block = None

        #genes x gene set normalized matrix
        #REMOVING THIS for memory savings
        #self.X = None

        #this is the number of gene sets to put into a batch when fetching blocks of X
        self.batch_size = batch_size

        #flag to indicate whether these scale factors correspond to X_orig or the (implicit) whitened version
        #if True, they can be used directly with _get_X_blocks
        #if False but y_corr_cholesky is True, then they need to be recomputed
        self.scale_is_for_whitened = False
        self.scale_factors = None
        self.mean_shifts = None

        self.scale_factors_missing = None
        self.mean_shifts_missing = None

        self.scale_factors_ignored = None
        self.mean_shifts_ignored = None

        #whether this was originally a dense or sparse gene set
        self.is_dense_gene_set = None
        self.is_dense_gene_set_missing = None

        self.gene_set_batches = None
        self.gene_set_batches_missing = None

        self.gene_set_labels = None
        self.gene_set_labels_missing = None
        self.gene_set_labels_ignored = None

        #ordered list of genes
        self.genes = None
        self.genes_missing = None
        self.gene_to_ind = None
        self.gene_missing_to_ind = None

        self.gene_chrom_name_pos = None
        self.gene_to_chrom = None
        self.gene_to_pos = None
        self.gene_to_gwas_huge_score = None
        self.gene_to_gwas_huge_score_uncorrected = None
        self.gene_to_exomes_huge_score = None
        self.gene_to_huge_score = None

        self.gene_to_positive_controls = None

        self.gene_label_map = None

        #ordered list of gene sets
        self.gene_sets = None
        self.gene_sets_missing = None
        self.gene_sets_ignored = None
        self.gene_set_to_ind = None

        #gene set association statistics
        #self.max_gene_set_p = None

        self.is_logistic = None

        self.beta_tildes = None
        self.p_values = None
        self.ses = None
        self.z_scores = None

        self.beta_tildes_orig = None
        self.p_values_orig = None
        self.ses_orig = None
        self.z_scores_orig = None

        #these store the inflation of SE relative to OLS (if ols_corrected is run)
        self.se_inflation_factors = None

        #these are gene sets we filtered out but need to persist for OSC
        self.beta_tildes_missing = None
        self.p_values_missing = None
        self.ses_missing = None
        self.z_scores_missing = None
        self.se_inflation_factors_missing = None

        #these are gene sets we ignored at the start
        self.col_sums_ignored = None

        self.beta_tildes_ignored = None
        self.p_values_ignored = None
        self.ses_ignored = None
        self.z_scores_ignored = None
        self.se_inflation_factors_ignored = None

        self.beta_tildes_missing_orig = None
        self.p_values_missing_orig = None
        self.ses_missing_orig = None
        self.z_scores_missing_orig = None

        #DO WE NEED THIS???
        #self.y_mean = None
        self.Y = None
        self.Y_exomes = None
        self.Y_positive_controls = None

        #this is to store altered variables if we detect power
        self.Y_for_regression = None

        self.y_var = 1 #total variance of the Y
        self.Y_orig = None
        self.Y_for_regression_orig = None
        self.Y_w_orig = None
        self.Y_fw_orig = None


        self.gene_locations = None #this stores sort orders for genes, which is populated when fitting correlation matrix from gene loc file
        
        self.huge_signal_bfs = None
        #covariates for genes
        self.huge_gene_covariates = None
        self.huge_gene_covariates_mask = None
        self.huge_gene_covariate_names = None
        self.huge_gene_covariate_intercept_index = None
        self.huge_gene_covariate_betas = None
        self.huge_gene_covariates_mat_inv = None
        self.huge_gene_covariate_zs = None

        self.huge_signals = None
        self.huge_signal_posteriors = None
        self.huge_signal_posteriors_for_regression = None
        self.huge_signal_sum_gene_cond_probabilities = None
        self.huge_signal_mean_gene_pos = None
        self.huge_signal_max_closest_gene_prob = None

        self.huge_cap_region_posterior = True
        self.huge_scale_region_posterior = False
        self.huge_phantom_region_posterior = False
        self.huge_allow_evidence_of_absence = False
        self.huge_correct_huge = True

        self.y_corr = None #this stores the (banded) correlation matrix for the Y values
        #In addition to storing banded correlation matrix, this signals that we are in partial GLS mode (OLS with inflated SEs)
        self.y_corr_sparse = None #another representation of the banded correlation matrix
        #In addition to storing cholesky decomp, this being set to not None triggers everything to operate in full GLS mode
        self.y_corr_cholesky = None #this stores the cholesky decomposition of the (banded) correlation matrix for the Y values
        #these are the "whitened" ys that are multiplied by sigma^{-1/2}
        self.Y_w = None
        self.y_w_var = 1 #total variance of the whitened Y
        self.y_w_mean = 0 #total mean of the whitened Y
        #these are the "full whitened" ys that are multiplied by sigma^{-1}
        self.Y_fw = None
        self.y_fw_var = 1 #total variance of the whitened Y
        self.y_fw_mean = 0 #total mean of the whitened Y

        #statistics for sigma regression
        self.osc = None
        self.X_osc = None
        self.osc_weights = None

        self.osc_missing = None
        self.X_osc_missing = None
        self.osc_weights_missing = None

        #statistics for gene set qc
        self.total_qc_metrics = None
        self.mean_qc_metrics = None

        self.total_qc_metrics_missing = None
        self.mean_qc_metrics_missing = None

        self.total_qc_metrics_ignored = None
        self.mean_qc_metrics_ignored = None

        self.p = None
        self.ps = None #this allows gene sets to have different ps
        self.ps_missing = None #this allows gene sets to have different ps
        self.sigma2 = None #sigma2 * np.power(scale_factor, sigma_power) is the prior used for the internal beta
        self.sigma2s = None #this allows gene sets to have different sigma2s
        self.sigma2s_missing = None #this allows gene sets to have different sigma2s

        self.sigma2_osc = None
        self.sigma2_se = None
        self.intercept = None
        self.sigma2_p = None
        self.sigma2_total_var = None
        self.sigma2_total_var_lower = None
        self.sigma2_total_var_upper = None

        #statistics for gene set betas
        self.betas = None
        self.betas_uncorrected = None
        self.inf_betas = None
        self.non_inf_avg_cond_betas = None
        self.non_inf_avg_postps = None

        self.betas_missing = None
        self.betas_uncorrected_missing = None
        self.inf_betas_missing = None
        self.non_inf_avg_cond_betas_missing = None
        self.non_inf_avg_postps_missing = None

        self.betas_orig = None
        self.betas_uncorrected_orig = None
        self.inf_betas_orig = None
        self.non_inf_avg_cond_betas_orig = None
        self.non_inf_avg_postps_orig = None

        self.betas_missing_orig = None
        self.betas_uncorrected_missing_orig = None
        self.inf_betas_missing_orig = None
        self.non_inf_avg_cond_betas_missing_orig = None
        self.non_inf_avg_postps_missing_orig = None

        #statistics for genes
        self.priors = None
        self.priors_adj = None
        self.combined_prior_Ys = None
        self.combined_prior_Ys_for_regression = None

        self.combined_prior_Ys_adj = None
        self.combined_prior_Y_ses = None
        self.combined_Ds = None
        self.combined_Ds_for_regression = None
        self.combined_Ds_missing = None
        self.priors_missing = None
        self.priors_adj_missing = None

        self.gene_N = None
        self.gene_ignored_N = None #number of ignored gene sets gene is in

        self.gene_N_missing = None #gene_N for genes with missing values for Y
        self.gene_ignored_N_missing = None #gene_N_missing for genes with missing values for Y

        self.batches = None

        self.priors_orig = None
        self.priors_adj_orig = None
        self.priors_missing_orig = None
        self.priors_adj_missing_orig = None

        #model parameters
        self.sigma_power = None

        #soft thresholding of sigmas
        self.sigma_threshold_k = None
        self.sigma_threshold_xo = None
        
        #stores all parameters used
        self.params = {}
        self.param_keys = []

        #stores factored matrices
        self.exp_gene_factors = None
        self.gene_factor_gene_mask = None

        self.exp_gene_set_factors = None
        self.gene_set_factor_gene_set_mask = None

        self.exp_lambdak = None
        self.factor_gene_set_scores = None
        self.factor_gene_scores = None
        self.factor_top_gene_sets = None
        self.factor_top_genes = None
        self.factor_labels = None


    def init_gene_locs(self, gene_loc_file):
        log("Reading --gene-loc-file %s" % gene_loc_file)
        (self.gene_chrom_name_pos, self.gene_to_chrom, self.gene_to_pos) = self._read_loc_file(gene_loc_file)

    def read_gene_map(self, gene_map_in, gene_map_orig_gene_col=1, gene_map_new_gene_col=2):

        if self.gene_label_map is None:
            self.gene_label_map = {}

        gene_map_orig_gene_col -= 1
        if gene_map_orig_gene_col < 0:
            bail("--gene-map-orig-gene-col must be greater than 1")
        gene_map_new_gene_col -= 1
        if gene_map_new_gene_col < 0:
            bail("--gene-map-new-gene-col must be greater than 1")

        with open(gene_map_in) as map_fh:
            for line in map_fh:
                cols = line.strip().split()
                if len(cols) <= gene_map_orig_gene_col or len(cols) <= gene_map_new_gene_col:
                    bail("Not enough columns in --gene-map-in:\n\t%s" % line)

                orig_gene = cols[0]
                new_gene = cols[1]
                self.gene_label_map[orig_gene] = new_gene

    def read_Y(self, gwas_in=None, exomes_in=None, positive_controls_in=None, gene_bfs_in=None, gene_percentiles_in=None, gene_zs_in=None, gene_loc_file=None, hold_out_chrom=None, **kwargs):

        Y1_exomes = np.array([])
        extra_genes_exomes = []
        extra_Y_exomes = []

        def __hold_out_chrom(Y, extra_genes, extra_Y):
            if hold_out_chrom is None:
                return (Y, extra_genes, extra_Y)

            if self.gene_to_chrom is None:
                (self.gene_chrom_name_pos, self.gene_to_chrom, self.gene_to_pos) = self._read_loc_file(gene_loc_file)

            extra_Y_mask = np.full(len(extra_Y), True)
            for i in range(len(extra_genes)):
                if extra_genes[i] in self.gene_to_chrom and self.gene_to_chrom[extra_genes[i]] == hold_out_chrom:
                    extra_Y_mask[i] = False
            if np.sum(~extra_Y_mask) > 0:
                extra_genes = [extra_genes[i] for i in range(len(extra_genes)) if extra_Y_mask[i]]
                extra_Y = extra_Y[extra_Y_mask]

            if self.genes is not None:
                Y_nan_mask = np.full(len(Y), False)
                for i in range(len(self.genes)):
                    if self.genes[i] in self.gene_to_chrom and self.gene_to_chrom[self.genes[i]] == hold_out_chrom:
                        Y_nan_mask[i] = True
                if np.sum(Y_nan_mask) > 0:
                    Y[Y_nan_mask] = np.nan

            return (Y, extra_genes, extra_Y)
        
        if exomes_in is not None:
            (Y1_exomes,extra_genes_exomes,extra_Y_exomes) = self.calculate_huge_scores_exomes(exomes_in, hold_out_chrom=hold_out_chrom, gene_loc_file=gene_loc_file, **kwargs)
            if self.genes is None:
                self._set_X(self.X_orig, extra_genes_exomes, self.gene_sets, skip_N=True, skip_V=True)
                #set this temporarily for use in huge
                self.Y_exomes = extra_Y_exomes
                Y1_exomes = extra_Y_exomes
                extra_genes_exomes = []
                extra_Y_exomes = np.array([])
            

        missing_value_exomes = 0
        missing_value_positive_controls = 0

        Y1_positive_controls = np.array([])
        extra_genes_exomes_positive_controls = extra_genes_exomes
        extra_Y_positive_controls = []

        if positive_controls_in is not None:
            (Y1_positive_controls,extra_genes_positive_controls,extra_Y_positive_controls) = self.read_positive_controls(positive_controls_in, hold_out_chrom=hold_out_chrom, gene_loc_file=gene_loc_file, **kwargs)
            if self.genes is None:
                assert(len(Y1_exomes) == 0)
                self._set_X(self.X_orig, extra_genes_positive_controls, self.gene_sets, skip_N=True, skip_V=True)
                #set this temporarily for use in huge
                self.Y_positive_controls = extra_Y_positive_controls
                Y1_positive_controls = extra_Y_positive_controls
                extra_genes_positive_controls = []
                extra_Y_positive_controls = np.array([])
                Y1_exomes = np.zeros(len(Y1_positive_controls))
            else:
                #exomes is already aligned to self.genes: Y1_exomes matches self.genes
                #extra_genes_exomes / extra_Y_exomes has anything not in it
                #we need to:
                #1. make sure Y1_exomes and Y1_positive_controls are the same length (already done)
                #2. combine extra_genes for the two
                #3. make sure that the 


                #align these so that genes includes the union of exomes and positive controls
                #and the extras moved in are no longer in extra
                #only need to remove the extras from exomes, since it was loaded into self.genes already
                extra_gene_to_ind = self._construct_map_to_ind(extra_genes_positive_controls)
                extra_Y_positive_controls = list(extra_Y_positive_controls)
                new_extra_Y_exomes = list(np.full(len(extra_Y_positive_controls), missing_value_exomes))
                num_add = 0
                extra_genes_exomes_positive_controls = extra_genes_positive_controls
                for i in range(len(extra_genes_exomes)):
                    if extra_genes_exomes[i] in extra_gene_to_ind:
                        new_extra_Y_exomes[extra_gene_to_ind[extra_genes_exomes[i]]] = extra_Y_exomes[i]
                    else:
                        num_add += 1
                        extra_genes_exomes_positive_controls.append(extra_genes_exomes[i])
                        extra_Y_positive_controls.append(missing_value_positive_controls)
                        new_extra_Y_exomes.append(extra_Y_exomes[i])

                extra_Y_exomes = np.array(new_extra_Y_exomes)
                extra_Y_positive_controls = np.array(extra_Y_positive_controls)
        else:
            Y1_positive_controls = np.zeros(len(Y1_exomes))
            extra_Y_positive_controls = np.zeros(len(extra_Y_positive_controls))
            extra_genes_exomes_positive_controls = extra_genes_exomes

        assert(len(extra_Y_exomes) == len(extra_genes_exomes_positive_controls))
        assert(len(extra_Y_exomes) == len(extra_Y_positive_controls))
        assert(len(Y1_exomes) == len(Y1_positive_controls))

        missing_value = None
        gene_combined_map = None
        gene_prior_map = None

        if gwas_in is not None:
            (Y1,extra_genes,extra_Y,Y1_for_regression,extra_Y_for_regression) = self.calculate_huge_scores_gwas(gwas_in, gene_loc_file=gene_loc_file, hold_out_chrom=hold_out_chrom, **kwargs)
            missing_value = 0

        else:
            self.huge_signal_bfs = None
            self.huge_gene_covariates = None
            self.huge_gene_covariates_mask = None
            self.huge_gene_covariate_names = None
            self.huge_gene_covariate_intercept_index = None            
            self.huge_gene_covariate_betas = None
            self.huge_gene_covariates_mat_inv = None
            self.huge_gene_covariate_zs = None
            if gene_bfs_in is not None:
                (Y1,extra_genes,extra_Y, gene_combined_map, gene_prior_map)  = self._read_gene_bfs(gene_bfs_in, **kwargs)
            elif gene_percentiles_in is not None:
                (Y1,extra_genes,extra_Y) = self._read_gene_percentiles(gene_percentiles_in, **kwargs)
            elif gene_zs_in is not None:
                (Y1,extra_genes,extra_Y) = self._read_gene_zs(gene_zs_in, **kwargs)
            elif exomes_in is not None:
                (Y1,extra_genes,extra_Y) = (np.zeros(Y1_exomes.shape), [], [])
            else:
                bail("Need to specify either gene_bfs_in or gene_percentiles_in or gene_zs_in or exomes_in")

            (Y1,extra_genes,extra_Y) = __hold_out_chrom(Y1,extra_genes,extra_Y)
            Y1_for_regression = copy.copy(Y1)
            extra_Y_for_regression = copy.copy(extra_Y)


        #we now need to construct several arrays
        #1. self.genes (if it hasn't been constructed already)
        #2. Y: the total combined gwas + exome values for all genes in self.genes
        #3. Y_exomes: the exome values for all genes in self.genes
        #4. extra_genes: genes not in self.genes but for which we have exome or gwas values
        #5. extra_Y: the total combined gwas + exome values for all genes in either gwas or exomes but not in self.genes
        #6. extra_Y_exomes: the total exome values for all genes in either gwas or exomes but not in self.genes

        if missing_value is None:
            if len(Y1) > 0:
                missing_value = np.nanmean(Y1)
            else:
                missing_value = 0

        if self.genes is None:
            assert(len(Y1) == 0)
            assert(len(Y1_exomes) == 0)
            assert(len(Y1_positive_controls) == 0)

            #combine everything
            genes_set = set(extra_genes).union(extra_genes_exomes_positive_controls)

            #really calling this just to set the genes
            self._set_X(self.X_orig, list(genes_set), self.gene_sets, skip_N=False)

            #now need to reorder
            Y = np.full(len(self.genes), missing_value, dtype=float)
            Y_for_regression = np.full(len(self.genes), missing_value, dtype=float)
            Y_exomes = np.full(len(self.genes), missing_value_exomes, dtype=float)
            Y_positive_controls = np.full(len(self.genes), missing_value_positive_controls, dtype=float)

            for i in range(len(extra_genes)):
                Y[self.gene_to_ind[extra_genes[i]]] = extra_Y[i]
                Y_for_regression[self.gene_to_ind[extra_genes[i]]] = extra_Y_for_regression[i]

            for i in range(len(extra_genes_exomes_positive_controls)):
                Y_exomes[self.gene_to_ind[extra_genes_exomes_positive_controls[i]]] = extra_Y_exomes[i]

            Y += Y_exomes
            Y += Y_positive_controls

            Y_for_regression += Y_exomes
            Y_for_regression += Y_positive_controls

            if self.huge_signal_bfs is not None or self.huge_gene_covariates is not None or self.huge_gene_covariates_mask is not None:

                #we need to reorder
                #the order is the same as extra_genes
                if self.huge_signal_bfs is not None:
                    index_map = {i: self.gene_to_ind[extra_genes[i]] for i in range(len(extra_genes))}
                    self.huge_signal_bfs = sparse.csc_matrix((self.huge_signal_bfs.data, [index_map[x] for x in self.huge_signal_bfs.indices], self.huge_signal_bfs.indptr), shape=self.huge_signal_bfs.shape)

                if self.huge_gene_covariates is not None or self.huge_gene_covariates_mask:
                    index_map_rev = {self.gene_to_ind[extra_genes[i]]: i for i in range(len(extra_genes))}
                    if self.huge_gene_covariates is not None:
                        self.huge_gene_covariates = self.huge_gene_covariates[[index_map_rev[x] for x in range(self.huge_gene_covariates.shape[0])],:]
                    if self.huge_gene_covariate_zs is not None:
                        self.huge_gene_covariate_zs = self.huge_gene_covariate_zs[[index_map_rev[x] for x in range(self.huge_gene_covariate_zs.shape[0])],:]

                    if self.huge_gene_covariates_mask is not None:
                        self.huge_gene_covariates_mask = self.huge_gene_covariates_mask[[index_map_rev[x] for x in range(self.huge_gene_covariates.shape[0])]]

            extra_genes = []
            extra_Y = np.array([])
            extra_Y_for_regression = np.array([])
            extra_Y_exomes = np.array([])
            extra_Y_positive_controls = np.array([])

        else:
            #sum the existing genes together
            Y = Y1 + Y1_exomes + Y1_positive_controls
            Y[np.isnan(Y1)] = Y1_exomes[np.isnan(Y1)] + Y1_positive_controls[np.isnan(Y1)] + missing_value
            Y[np.isnan(Y1_exomes)] = Y1[np.isnan(Y1_exomes)] + Y1_positive_controls[np.isnan(Y1_exomes)] + missing_value_exomes
            Y[np.isnan(Y1_positive_controls)] = Y1[np.isnan(Y1_positive_controls)] + Y1_exomes[np.isnan(Y1_positive_controls)] + missing_value_positive_controls

            Y_for_regression = Y1_for_regression + Y1_exomes + Y1_positive_controls
            Y_for_regression[np.isnan(Y1_for_regression)] = Y1_exomes[np.isnan(Y1_for_regression)] + Y1_positive_controls[np.isnan(Y1_for_regression)] + missing_value
            Y_for_regression[np.isnan(Y1_exomes)] = Y1_for_regression[np.isnan(Y1_exomes)] + Y1_positive_controls[np.isnan(Y1_exomes)] + missing_value_exomes
            Y_for_regression[np.isnan(Y1_positive_controls)] = Y1_for_regression[np.isnan(Y1_positive_controls)] + Y1_exomes[np.isnan(Y1_positive_controls)] + missing_value_positive_controls

            Y_exomes = Y1_exomes
            Y_exomes[np.isnan(Y1_exomes)] = missing_value_exomes

            Y_positive_controls = Y1_positive_controls
            Y_positive_controls[np.isnan(Y1_positive_controls)] = missing_value_positive_controls

            extra_gene_to_ind = self._construct_map_to_ind(extra_genes)
            extra_Y = list(extra_Y)
            extra_Y_for_regression = list(extra_Y_for_regression)
            new_extra_Y_exomes = list(np.full(len(extra_Y), missing_value_exomes))
            new_extra_Y_positive_controls = list(np.full(len(extra_Y), missing_value_positive_controls))

            num_add = 0
            for i in range(len(extra_genes_exomes_positive_controls)):
                if extra_genes_exomes_positive_controls[i] in extra_gene_to_ind:
                    extra_Y[extra_gene_to_ind[extra_genes_exomes_positive_controls[i]]] += (extra_Y_exomes[i] + extra_Y_positive_controls[i])
                    extra_Y_for_regression[extra_gene_to_ind[extra_genes_exomes_positive_controls[i]]] += (extra_Y_exomes[i] + extra_Y_positive_controls[i])
                    new_extra_Y_exomes[extra_gene_to_ind[extra_genes_exomes_positive_controls[i]]] = extra_Y_exomes[i]
                    new_extra_Y_positive_controls[extra_gene_to_ind[extra_genes_exomes_positive_controls[i]]] = extra_Y_positive_controls[i]
                else:
                    num_add += 1
                    extra_genes.append(extra_genes_exomes_positive_controls[i])
                    extra_Y.append(extra_Y_exomes[i] + extra_Y_positive_controls[i])
                    extra_Y_for_regression.append(extra_Y_exomes[i] + extra_Y_positive_controls[i])
                    new_extra_Y_exomes.append(extra_Y_exomes[i])
                    new_extra_Y_positive_controls.append(extra_Y_positive_controls[i])

            extra_Y = np.array(extra_Y)
            extra_Y_for_regression = np.array(extra_Y_for_regression)
            extra_Y_exomes = np.array(new_extra_Y_exomes)
            extra_Y_positive_controls = np.array(new_extra_Y_positive_controls)

            if self.huge_signal_bfs is not None:
                #have to add space for the exomes results that were added at the end
                self.huge_signal_bfs = sparse.csc_matrix((self.huge_signal_bfs.data, self.huge_signal_bfs.indices, self.huge_signal_bfs.indptr), shape=(self.huge_signal_bfs.shape[0] + num_add, self.huge_signal_bfs.shape[1]))

            if self.huge_gene_covariates is not None:
                add_gene_covariates = np.tile(np.mean(self.huge_gene_covariates, axis=0), num_add).reshape((num_add, self.huge_gene_covariates.shape[1]))
                
                self.huge_gene_covariates = np.vstack((self.huge_gene_covariates, add_gene_covariates))

            if self.huge_gene_covariate_zs is not None:
                add_gene_covariate_zs = np.tile(np.mean(self.huge_gene_covariate_zs, axis=0), num_add).reshape((num_add, self.huge_gene_covariate_zs.shape[1]))
                self.huge_gene_covariate_zs = np.vstack((self.huge_gene_covariate_zs, add_gene_covariate_zs))

            if self.huge_gene_covariates_mask is not None:
                self.huge_gene_covariates_mask = np.append(self.huge_gene_covariates_mask, np.full(num_add, False))

        #Y contains all of the genes in self.genes that have gene statistics
        #extra_Y contains additional genes not in self.genes that have gene statistics.
        #Since these will be used in the regression, they must be accounted for in the normalization of X and V

        if len(extra_Y) > 0:
            Y = np.concatenate((Y, extra_Y))
            Y_for_regression = np.concatenate((Y_for_regression, extra_Y_for_regression))
            Y_exomes = np.concatenate((Y_exomes, extra_Y_exomes))
            Y_positive_controls = np.concatenate((Y_positive_controls, extra_Y_positive_controls))

        if self.X_orig is not None:
            #Use original X because no whitening has taken place yet
            log("Expanding matrix", TRACE)
            self._set_X(sparse.csc_matrix((self.X_orig.data, self.X_orig.indices, self.X_orig.indptr), shape=(self.X_orig.shape[0] + len(extra_Y), self.X_orig.shape[1])), self.genes, self.gene_sets, skip_V=True, skip_scale_factors=True, skip_N=False)

        if self.genes is not None:
            self._set_X(self.X_orig, self.genes + extra_genes, self.gene_sets, skip_N=False)

               
        self._set_Y(Y, Y_for_regression, Y_exomes, Y_positive_controls, skip_V=True, skip_scale_factors=True)

        #if we read in combined or priors
        if gene_combined_map is not None:
            self.combined_prior_Ys = copy.copy(self.Y)
            for i in range(len(self.genes)):
                if self.genes[i] in gene_combined_map:
                    self.combined_prior_Ys[i] = gene_combined_map[self.genes[i]]
        if gene_prior_map is not None:
            self.priors = np.zeros(len(self.Y))
            for i in range(len(self.genes)):
                if self.genes[i] in gene_prior_map:
                    self.priors[i] = gene_prior_map[self.genes[i]]

    #Initialize the matrices, genes, and gene sets
    #This can be called multiple times; it will subset the current matrices down to the new set of gene sets
    #any information regarding *genes* though is overwritten -- there is no way to subset the old genes down to a new set of genes
    #(although reading multiple files hasn't been tested thoroughly)
    def read_X(self, X_in, Xd_in=None, X_list=None, Xd_list=None, V_in=None, skip_V=True, force_reread=False, min_gene_set_size=1, max_gene_set_size=30000, only_ids=None, prune_gene_sets=0.8, x_sparsify=[50,100,200,500,1000], add_ext=False, add_top=True, add_bottom=True, filter_negative=True, threshold_weights=0.5, max_gene_set_p=None, filter_gene_set_p=1, increase_filter_gene_set_p=0.01, max_num_gene_sets=None, filter_gene_set_metric_z=2.5, initial_p=0.01, initial_sigma2=1e-3, initial_sigma2_cond=None, sigma_power=0, sigma_soft_threshold_95=None, sigma_soft_threshold_5=None, run_logistic=True, run_gls=False, run_corrected_ols=False, gene_loc_file=None, gene_cor_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, update_hyper_p=False, update_hyper_sigma=False, batch_all_for_hyper=False, first_for_hyper=False, first_for_sigma_cond=False, sigma_num_devs_to_top=2.0, p_noninf_inflate=1, batch_separator="@", file_separator=None, max_num_burn_in=None, max_num_iter_betas=1100, min_num_iter_betas=10, num_chains_betas=10, r_threshold_burn_in_betas=1.01, use_max_r_for_convergence_betas=True, max_frac_sem_betas=0.01, max_allowed_batch_correlation=None, sparse_solution=False, sparse_frac_betas=None, betas_trace_out=None, show_progress=True):
        X_format = "<gene_set_id> <gene 1> <gene 2> ... <gene n>"
        V_format = "<gene_set1> <gene_set_2> ...<gene_set_n>\n<V11> <V12> ... <V1n>\n<V21> <V22> ... <V2n>"

        if not force_reread and self.X_orig is not None:
            return

        self._set_X(None, self.genes, None, skip_N=True)

        self._record_params({"filter_gene_set_p": filter_gene_set_p, "filter_negative": filter_negative, "threshold_weights": threshold_weights, "max_num_gene_sets": max_num_gene_sets, "filter_gene_set_metric_z": filter_gene_set_metric_z, "num_chains_betas": num_chains_betas, "sigma_num_devs_to_top": sigma_num_devs_to_top, "p_noninf_inflate": p_noninf_inflate})

        def expand_Xs(Xs, orig_files):
            new_Xs = []
            batches = []
            labels = []
            new_orig_files = []
            for i in range(len(Xs)):
                X = Xs[i]
                orig_file = orig_files[i]
                batch = None
                label = os.path.basename(orig_file)
                if "." in label:
                    label = label.split(".")[:-1]
                if batch_separator in X:
                    batch = X.split(batch_separator)[-1]
                    label = batch
                    X = batch_separator.join(X.split(batch_separator)[:-1])
                if file_separator is not None:
                    x_to_add = X.split(file_separator)
                else:
                    x_to_add = [X]
                new_Xs += x_to_add
                batches += [batch] * len(x_to_add)
                labels += [label] * len(x_to_add)
                new_orig_files += [orig_file] * len(x_to_add)
            return (new_Xs, batches, labels, new_orig_files)

        #list of the X files specified on the command line
        X_ins = []
        orig_files = []
        if X_in is not None:
            if type(X_in) == str:
                X_ins = [X_in]
                orig_files = [X_in]
            elif type(X_in) == list:
                X_ins = X_in
                orig_files = copy.copy(X_in)

        is_dense = []

        if X_list is not None:
            X_lists = []
            if type(X_list) == str:
                X_lists = [X_list]
            elif type(X_list) == list:
                X_lists = X_list

            for X_list in X_lists:
                batch = None
                if batch_separator in X_list:
                    batch = X_list.split(batch_separator)[-1]
                    X_list = batch_separator.join(X_list.split(batch_separator)[:-1])

                with open(X_list) as X_list_fh:
                    for line in X_list_fh:
                        line = line.strip()
                        if batch is not None and batch_separator not in line:
                            line = "%s%s%s" % (line, batch_separator, batch)
                        X_ins.append(line)
                        orig_files.append(X_list)

        X_ins, batches, labels, orig_files = expand_Xs(X_ins, orig_files)

        #TODO: read in labels here, batches2, and then when append

        is_dense = [False for x in X_ins]

        Xd_ins = []
        orig_dfiles = []
        if Xd_in is not None:
            if type(Xd_in) == str:
                Xd_ins = [Xd_in]
                orig_dfiles = [Xd_in]
            elif type(Xd_in) == list:
                Xd_ins = Xd_in
                orig_dfiles = Xd_in

        if Xd_list is not None:
            if type(Xd_list) == str:
                Xd_lists = [Xd_list]
            elif type(Xd_list) == list:
                Xd_lists = Xd_list

            for Xd_list in Xd_lists:
                batch = None
                if batch_separator in Xd_list:
                    batch = Xd_list.split(batch_separator)[-1]
                    Xd_list = batch_separator.join(Xd_list.split(batch_separator)[:-1])

                with open(Xd_list) as Xd_list_fh:
                    for line in Xd_list_fh:
                        line = line.strip()
                        if batch is not None and batch_separator not in line:
                            line = "%s%s%s" % (line, batch_separator, batch)
                        Xd_ins.append(line)
                        orig_dfiles.append(Xd_list)

        Xd_ins, batches2, labels2, orig_dfiles = expand_Xs(Xd_ins, orig_dfiles)

        X_ins += Xd_ins
        batches += batches2
        labels += labels2
        orig_files += orig_dfiles
        is_dense += [True for x in Xd_ins]

        #first reorder the files so that those with batches are at the front
        
        #X_ins = [X_ins[i] for i in range(len(batches)) if batches[i] is not None] + [X_ins[i] for i in range(len(batches)) if batches[i] is None]
        #is_dense = [is_dense[i] for i in range(len(batches)) if batches[i] is not None] + [is_dense[i] for i in range(len(batches)) if batches[i] is None]
        #orig_files = [orig_files[i] for i in range(len(batches)) if batches[i] is not None] + [orig_files[i] for i in range(len(batches)) if batches[i] is None]

        #batches = [batches[i] for i in range(len(batches)) if batches[i] is not None] + [batches[i] for i in range(len(batches)) if batches[i] is None]

        #README: batching / hyper semantics 
        #Use of @{batch} after file is the way to label a file
        #1. First we take each input file and assign it a batch
        #   If the file is labelled, that is the batch
        #   If the first file is not labelled, it is assigned a batch
        #   If the remaining files are not labelled, AND first-for-hyper is NOT set, they are assigned a batch.
        #   If first-for-hyper is set, batches with no labels bave None for batch
        #2. We then learn p (if update_hyper_p is specified) and sigma (if update_hyper_sigma is specified) separately for each batch
        #   All files with the same batch are pooled for learning the p and sigma
        #   If first_for_sigma_cond is specified, then the sigma to p ratio learned by the first batch is fixed throughout
        #   This means that if only one of sigma or p is learned, the other is adjusted to keep the sigma/p ratio the same

        #now handle the None batches
        #semantics are that things with a batch have value learned from all files with that batch,
        #things with None have it learned from first batch that appears in arg list

        used_batches = set([str(b) for b in batches if b is not None])
        next_batch_num = 1
        def __generate_new_batch(new_batch_num):
            new_batch = "BATCH%d" % new_batch_num
            while new_batch in used_batches:
                new_batch_num += 1
                new_batch = "BATCH%d" % new_batch_num
            used_batches.add(new_batch)
            return new_batch, new_batch_num

        for i in range(len(batches)):
            if batches[i] is None:
                batches[i], next_batch_num = __generate_new_batch(next_batch_num)

                if batch_all_for_hyper:
                    for j in range(i+1,len(batches)):
                        batches[j] = batches[i]
                    break
                else:
                    #now find all other none batches with the same file and update them too
                    for j in range(i+1,len(batches)):
                        if batches[j] is None and orig_files[i] == orig_files[j]:
                            batches[j] = batches[i]

            if first_for_hyper:
                #make sure though that at least one batch is not None (this is what we will use to learn everything)
                #but then break; keep None batches to learn from the first batch
                #also set all other batches to None (we won't be learning for those)
                for j in range(i+1, len(batches)):
                    if batches[j] != batches[i]:
                        batches[j] = None
                break


        self._record_params({"num_X_batches": len(batches)})

        log("Will learn parameters for %d files as %d batches and fill in %d additional files from the first" % (len([x for x in batches if x is not None]), len(set([x for x in batches if x is not None])), len([x for x in batches if x is None])))
        if first_for_sigma_cond:
            log("Will fix conditional sigma from the first batch")
        #this will store the number of ignored gene sets per file
        num_ignored_gene_sets = np.zeros((len(batches)))

        #expands the file batches to have one per gene set
        self.gene_set_batches = np.array([])
        self.gene_set_labels = np.array([])

        if self.genes is None:
            self.genes = []

        self.gene_sets = []
        self.is_dense_gene_set = np.array([], dtype=bool)

        if (filter_gene_set_p < 1 or filter_gene_set_metric_z) and self.Y is not None:
            self.gene_sets_ignored = []
            if self.gene_set_labels is not None:
                self.gene_set_labels_ignored = np.array([])

            self.col_sums_ignored = np.array([])
            self.scale_factors_ignored = np.array([])
            self.mean_shifts_ignored = np.array([])
            self.beta_tildes_ignored = np.array([])
            self.p_values_ignored = np.array([])
            self.ses_ignored = np.array([])
            self.z_scores_ignored = np.array([])
            self.se_inflation_factors_ignored = np.array([])


            self.beta_tildes = np.array([])
            self.p_values = np.array([])
            self.ses = np.array([])
            self.z_scores = np.array([])

            self.se_inflation_factors = None

            self.total_qc_metrics = None
            self.mean_qc_metrics = None

            self.total_qc_metrics_missing = None
            self.mean_qc_metrics_missing = None

            self.total_qc_metrics_ignored = None
            self.mean_qc_metrics_ignored = None

            self.ps = None
            self.ps_missing = None
            self.sigma2s = None
            self.sigma2s_missing = None


            if (run_gls or run_corrected_ols) and self.y_corr is None:
                correlation_m = self._read_correlations(gene_cor_file, gene_loc_file, gene_cor_file_gene_col=gene_cor_file_gene_col, gene_cor_file_cor_start_col=gene_cor_file_cor_start_col)

                #convert X and Y to their new values
                min_correlation = 0.05
                self._set_Y(self.Y, self.Y_for_regression, self.Y_exomes, self.Y_positive_controls, Y_corr_m=correlation_m, store_cholesky=run_gls, store_corr_sparse=run_corrected_ols, skip_V=True, skip_scale_factors=True, min_correlation=min_correlation)

        #returns num added, num ignored
        def __add_to_X(mat_info, genes, gene_sets, tag=None, skip_scale_factors=False, fname=None):

            #if self.genes_missing is not None:
            #    gene_to_ind = self._construct_map_to_ind(genes)
            #    #we are going to construct the full matrices including all of the missing genes
            #    #and then subset the matrix down
            #    genes += [x for x in self.genes_missing if x not in gene_to_ind]

            if tag is not None:
                gene_sets = ["%s_%s" % (tag, x) for x in gene_sets]

            is_dense = False
            if type(mat_info) is tuple:
                (data, row, col) = mat_info
                cur_X = sparse.csc_matrix((data, (row, col)), shape=(len(genes), len(gene_sets)))
                is_dense = False
                if cur_X.shape[1] == 0:
                    return (0, 0)
            else:

                #is_dense = True
                #disabling this setting
                is_dense = False

                if self.gene_label_map is not None:
                    genes = list(map(lambda x: self.gene_label_map[x] if x in self.gene_label_map else x, genes))

                #make sure no repeated genes
                if len(set(genes)) != len(genes):
                    #make the mask
                    seen_genes = set()
                    unique_mask = np.full(len(genes), True)

                    for i in range(len(genes)):
                        if genes[i] in seen_genes:
                            unique_mask[i] = False
                        else:
                            seen_genes.add(genes[i])
                    #now subset both down
                    mat_info = mat_info[unique_mask,:]
                    genes = [genes[i] for i in range(len(genes)) if unique_mask[i]] 


                #check if actually sparse
                if len(x_sparsify) > 0:
                    sparsity_threshold = 1 - np.max(x_sparsify).astype(float) / mat_info.shape[0]
                else:
                    sparsity_threshold = 0.95

                orig_dense_gene_sets = gene_sets

                cur_X = None
                #convert to sparse if (a) many zeros
                convert_to_sparse = np.sum(mat_info == 0, axis=0) / mat_info.shape[0] > sparsity_threshold

                # or (b) if all non-zero are same value
                abs_mat_info = np.abs(mat_info)
                max_weights = abs_mat_info.max(axis=0)
                all_non_zero_same = np.sum(abs_mat_info * (abs_mat_info != max_weights), axis=0) == 0

                convert_to_sparse = np.logical_or(convert_to_sparse, all_non_zero_same)
                if np.any(convert_to_sparse):
                    log("Detected sparse matrix for %d of %d columns" % (np.sum(convert_to_sparse), len(convert_to_sparse)), DEBUG)
                    cur_X = sparse.csc_matrix(mat_info[:,convert_to_sparse])
                    #update the gene sets, as well as the dense ones we will expand later
                    gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if convert_to_sparse[i]]
                    orig_dense_gene_sets = [orig_dense_gene_sets[i] for i in range(len(orig_dense_gene_sets)) if not convert_to_sparse[i]]

                    mat_info = mat_info[:,~convert_to_sparse]
                    #respect min gene size
                    enough_genes = self.get_col_sums(cur_X, num_nonzero=True) >= min_gene_set_size
                    if np.any(~enough_genes):
                        log("Excluded %d gene sets due to too small size" % np.sum(~enough_genes), DEBUG)
                        cur_X = cur_X[:,enough_genes]
                        gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if enough_genes[i]]

                if mat_info.shape[1] > 0:

                    mat_sd = np.std(mat_info, axis=0)
                    if np.any(mat_sd == 0):
                        mat_info = mat_info[:,mat_sd != 0]

                    mat_info = (mat_info - np.mean(mat_info, axis=0)) / np.std(mat_info, axis=0)

                    subset_mask = np.full(len(genes), True)
                    x_for_stats = mat_info
                    if self.Y is not None and self.genes is not None:
                        #make a mask to subset it down for the purposes of quantiles
                        subset_mask[[i for i in range(len(genes)) if genes[i] not in self.gene_to_ind]] = False
                        x_for_stats = mat_info[subset_mask,:]

                    if x_for_stats.shape[0] == 0:
                        warn("No genes in --Xd-in %swere seen before so skipping; example genes: %s" % ("%s " % fname if fname is not None else "", ",".join(genes[:4])))
                        return (0, 0)

                    top_numbers = list(reversed(sorted(x_sparsify)))

                    top_fractions = np.array(top_numbers, dtype=float) / x_for_stats.shape[0]

                    top_fractions[top_fractions > 1] = 1
                    top_fractions[top_fractions < 0] = 0

                    if len(top_fractions) == 0:
                        bail("No --X-sparsify set so doing nothing")
                        return (0, 0)

                    upper_quantiles = np.quantile(x_for_stats, 1 - top_fractions, axis=0)
                    lower_quantiles = np.quantile(x_for_stats, top_fractions, axis=0)

                    upper = copy.copy(mat_info)
                    lower = copy.copy(mat_info)

                    assert(np.all(upper_quantiles[0,:] == np.min(upper_quantiles, axis=0)))
                    assert(np.all(lower_quantiles[0,:] == np.max(lower_quantiles, axis=0)))

                    for i in range(len(top_numbers)):
                        #since we are sorted in descending order, can throw away everything below current threshold
                        upper_threshold_mask = upper < upper_quantiles[i,:]
                        if np.sum(upper_threshold_mask) == 0:
                            upper_threshold_mask = upper <= upper_quantiles[i,:]

                        lower_threshold_mask = lower > lower_quantiles[i,:]
                        if np.sum(lower_threshold_mask) == 0:
                            lower_threshold_mask = lower >= lower_quantiles[i,:]

                        mat_info[np.logical_and(upper_threshold_mask, lower_threshold_mask)] = 0
                        upper[upper_threshold_mask] = 0
                        lower[lower_threshold_mask] = 0

                        if add_ext:
                            temp_X = sparse.csc_matrix(mat_info)
                            top_gene_sets = ["%s_ext%d" % (x, top_numbers[i]) for x in orig_dense_gene_sets]
                            if cur_X is None:
                                cur_X = temp_X
                                gene_sets = top_gene_sets
                            else:
                                cur_X = sparse.hstack((cur_X, temp_X))
                                gene_sets = gene_sets + top_gene_sets

                        if add_bottom:
                            temp_X = sparse.csc_matrix(lower)
                            top_gene_sets = ["%s_bot%d" % (x, top_numbers[i]) for x in orig_dense_gene_sets]
                            if cur_X is None:
                                cur_X = temp_X
                                gene_sets = top_gene_sets
                            else:
                                cur_X = sparse.hstack((cur_X, temp_X))
                                gene_sets = gene_sets + top_gene_sets

                        if add_top or (not add_ext and not add_bottom):
                            temp_X = sparse.csc_matrix(upper)
                            top_gene_sets = ["%s_top%d" % (x, top_numbers[i]) for x in orig_dense_gene_sets]
                            if cur_X is None:
                                cur_X = temp_X
                                gene_sets = top_gene_sets
                            else:
                                gene_sets = gene_sets + top_gene_sets
                                cur_X = sparse.hstack((cur_X, temp_X))

                        if cur_X is None:
                            return (0, 0)

                        #if all of the values for a row are negative, flip the sign to make it positive
                        all_negative_mask = ((cur_X < 0).sum(axis=0) == cur_X.astype(bool).sum(axis=0)).A1
                        cur_X[:,all_negative_mask] = -cur_X[:,all_negative_mask]

                        cur_X.eliminate_zeros()

                    if cur_X is None or cur_X.shape[1] == 0:
                        return (0, 0)

                if self.genes is not None:
                    #need to reorder the genes to match the old and add the new ones
                    old_genes = genes
                    genes = self.genes
                    if self.genes_missing is not None:
                        genes += self.genes_missing
                    genes += [x for x in old_genes if (self.gene_to_ind is None or x not in self.gene_to_ind) and (self.gene_missing_to_ind is None or x not in self.gene_missing_to_ind)]
                    gene_to_ind = self._construct_map_to_ind(genes)
                    index_map = {i: gene_to_ind[old_genes[i]] for i in range(len(old_genes))}
                    cur_X = sparse.csc_matrix((cur_X.data, [index_map[x] for x in cur_X.indices], cur_X.indptr), shape=(len(genes), cur_X.shape[1]))

            denom = self.get_col_sums(cur_X, num_nonzero=True)
            denom[denom == 0] = 1
            avg_weights = np.abs(cur_X).sum(axis=0) / denom
            if np.sum(avg_weights != 1) > 0:
                #(mean_shifts_raw, scale_factors_raw) = self._calc_X_shift_scale(cur_X)
                #(mean_shifts_bool, scale_factors_bool) = self._calc_X_shift_scale(cur_X.astype(bool))
                #avg_weights = scale_factors_raw / scale_factors_bool

                #this is an option to use the max weight after throwing out outliers as the norm
                #it doesn't look to work as well as avg_weights
                max_weight_devs = None
                if max_weight_devs is not None:
                    dev_weights = np.sqrt(np.abs(cur_X).power(2).sum(axis=0) / denom - np.power(avg_weights, 2))
                    temp_X = copy.copy(np.abs(cur_X))
                    temp_X[temp_X > avg_weights + max_weight_devs * dev_weights] = 0

                    #I don't think we need to set really low ones to zero, since temp_X is only positive
                    #temp_X[temp_X < avg_weights - max_weight_devs * dev_weights] = 0

                    weight_norm = temp_X.max(axis=0).todense().A1
                else:
                    weight_norm = avg_weights.A1

                weight_norm = np.round(weight_norm, 10)
                weight_norm[weight_norm == 0] = 1

                #assume rows are already normalized if (a) all are below 1 and (b) threshold is None or all are above threshold 
                #so, normalize if (a) any is above 1 or (b) threshold is not None and any are below threshold 
                normalize_mask = (np.abs(cur_X) > 1).sum(axis=0).A1 > 0
                if threshold_weights is not None:
                    #check for those that have different number above 0 and above threshold
                    normalize_mask = np.logical_or(normalize_mask, (np.abs(cur_X) >= threshold_weights).sum(axis=0).A1 != (np.abs(cur_X) > 0).sum(axis=0).A1)

                #this uses less memory
                weight_norm[~normalize_mask] = 1.0
                cur_X = sparse.csc_matrix(cur_X.multiply(1.0 / weight_norm))
                #old method that uses higher memory
                #cur_X[:,normalize_mask] = sparse.csc_matrix(cur_X[:,normalize_mask].multiply(1.0 / weight_norm[normalize_mask]))

                #don't do binary; use threshold instead
                #if make_binary_weights is not None:
                #    cur_X.data[np.abs(cur_X.data) < make_binary_weights] = 0
                #    cur_X.data[np.abs(cur_X.data) >= make_binary_weights] = 1

                if threshold_weights is not None:
                    cur_X.data[np.abs(cur_X.data) < threshold_weights] = 0
                    cur_X.data[cur_X.data > 1] = 1
                    cur_X.data[cur_X.data < -1] = -1
                cur_X.eliminate_zeros()
            #now need to find any new genes that will be added as missing later, as well as any missing genes that need to be updated

            gene_ignored_N = None

            #these are the new missing that are in the old missing
            #these are not necessarily in the self.X structures, since self.genes could be set before that
            genes_missing_int = []
            cur_X_missing_genes_int = None
            gene_ignored_N_missing_int = None

            #these are the new missing that are not in the old missing
            genes_missing_new = []
            cur_X_missing_genes_new = None
            gene_ignored_N_missing_new = None

            if self.Y is not None and len(genes) > len(self.Y):
                genes_missing_old = self.genes_missing if self.genes_missing is not None else []
                gene_missing_old_to_ind = self._construct_map_to_ind(genes_missing_old)
                gene_to_ind = self._construct_map_to_ind(genes)

                #these are the genes that are new this time around
                genes_missing_new = [x for x in genes if x not in self.gene_to_ind and x not in gene_missing_old_to_ind]
                genes_missing_new_set = set(genes_missing_new)

                #these are missing genes shared with before
                genes_missing_int = [x for x in genes if x in gene_missing_old_to_ind]
                genes_missing_int_set = set(genes_missing_int)

                #all genes missing
                #genes_missing = set(genes_missing_new + genes_missing_int + genes_missing_old)
                #gene_missing_to_ind = self._construct_map_to_ind(genes_missing)
                #assert(len(genes_missing) == len(set(genes_missing)))

                #subset down X to only non missing

                int_mask = np.full(len(genes), False)
                int_mask[[i for i in range(len(genes)) if genes[i] in genes_missing_int_set]] = True
                if np.sum(int_mask) > 0:
                    cur_X_missing_genes_int = cur_X[int_mask,:]

                new_mask = np.full(len(genes), False)
                new_mask[[i for i in range(len(genes)) if genes[i] in genes_missing_new_set]] = True
                if np.sum(new_mask) > 0:
                    cur_X_missing_genes_new = cur_X[new_mask,:]

                subset_mask = np.full(len(genes), True)
                subset_mask[[i for i in range(len(genes)) if genes[i] not in self.gene_to_ind]] = False

                cur_X = cur_X[subset_mask,:]

                genes = [x for x in genes if x in self.gene_to_ind]

                #remove empty gene sets
                gene_set_nonempty_mask = self.get_col_sums(cur_X) > 0

                if np.sum(~gene_set_nonempty_mask) > 0:
                    cur_X = cur_X[:,gene_set_nonempty_mask]

                    if cur_X_missing_genes_int is not None:
                        cur_X_missing_genes_int = cur_X_missing_genes_int[:,gene_set_nonempty_mask]
                    if cur_X_missing_genes_new is not None:
                        cur_X_missing_genes_new = cur_X_missing_genes_new[:,gene_set_nonempty_mask]

                    gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if gene_set_nonempty_mask[i]]

                #at this point, we have subset X down to only non missing genes before
                assert(len(genes) == len(self.Y))

                #our missing genes come from two sources: self.X_orig_missing_genes (those are the old ones) and cur_X_missing_gnenes (those are the new ones). genes_missing_to_add tells us which are new

                #we only added genes at the end
                #num_add = len(genes) - len(self.Y)

                #new_Y = np.append(self.Y, np.full(num_add, np.nanmean(self.Y)))
                #new_Y_exomes = self.Y_exomes
                #if self.Y_exomes is not None:
                #    new_Y_exomes = np.append(self.Y_exomes, np.full(num_add, np.nanmean(self.Y_exomes)))

                #if self.y_corr is not None:
                #    padding = np.zeros((self.y_corr.shape[0], num_add))
                #    padding[0,:] = 1
                #    self.y_corr = np.hstack((self.y_corr, padding))

                #self._set_Y(new_Y, new_Y_exomes, Y_corr_m=self.y_corr, store_cholesky=run_gls and num_add > 0, store_corr_sparse=run_corrected_ols and num_add > 0, skip_V=skip_V)

                #if self.huge_signal_bfs is not None:
                #    self.huge_signal_bfs = sparse.csc_matrix((self.huge_signal_bfs.data, self.huge_signal_bfs.indices, self.huge_signal_bfs.indptr), shape=(self.huge_signal_bfs.shape[0] + num_add, self.huge_signal_bfs.shape[1]))

            p_value_ignore = None
            if (filter_gene_set_p < 1 or filter_gene_set_metric_z is not None) and self.Y is not None:

                log("Analyzing gene sets to pre-filter")

                (mean_shifts, scale_factors) = self._calc_X_shift_scale(cur_X)
                
                total_qc_metrics = None
                mean_qc_metrics = None                
                if self.huge_gene_covariates is not None:
                    cur_X_size = np.abs(cur_X).sum(axis=0)
                    cur_X_size[cur_X_size == 0] = 1

                    total_qc_metrics = (np.array(cur_X.T.dot(self.huge_gene_covariate_zs).T / cur_X_size)).T
                    total_qc_metrics = np.hstack((total_qc_metrics[:,:self.huge_gene_covariate_intercept_index], total_qc_metrics[:,self.huge_gene_covariate_intercept_index+1:]))
                    mean_qc_metrics = np.mean(total_qc_metrics, axis=1)

                Y_to_use = self.Y_for_regression
                if run_logistic:
                    Y = np.exp(Y_to_use + self.background_log_bf) / (1 + np.exp(Y_to_use + self.background_log_bf))
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors, alpha_tildes, diverged) = self._compute_logistic_beta_tildes(cur_X, Y, scale_factors, mean_shifts, resid_correlation_matrix=self.y_corr_sparse)
                else:
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = self._compute_beta_tildes(cur_X, Y_to_use, np.var(Y_to_use), scale_factors, mean_shifts, resid_correlation_matrix=self.y_corr_sparse)


                #if we have negative weights, that means we don't know which side is actually "better" for the trait (the feature is continuous). So flip the sign if the beta is negative
                negative_weights_mask = (cur_X < 0).sum(axis=0).A1 > 0
                if np.sum(negative_weights_mask) > 0:
                    flip_mask = np.logical_and(beta_tildes < 0, negative_weights_mask)
                    if np.sum(flip_mask) > 0:
                        log("Flipped %d gene sets" % np.sum(flip_mask), DEBUG)
                        beta_tildes[flip_mask] = -beta_tildes[flip_mask]
                        z_scores[flip_mask] = -z_scores[flip_mask]
                        cur_X[:,flip_mask] = -cur_X[:,flip_mask]

                p_value_mask = p_values <= filter_gene_set_p

                if increase_filter_gene_set_p is not None and np.mean(p_value_mask) < increase_filter_gene_set_p:
                    #choose a new more lenient threshold
                    p_from_quantile = np.quantile(p_values, increase_filter_gene_set_p)
                    log("Choosing revised p threshold %.3g to ensure keeping %.3g fraction of gene sets" % (p_from_quantile, increase_filter_gene_set_p), DEBUG)
                    p_value_mask = p_values <= p_from_quantile

                    if np.sum(~p_value_mask) > 0:
                        log("Ignoring %d gene sets due to p-value filters" % (np.sum(~p_value_mask)))

                if filter_negative:
                    negative_beta_tildes_mask = beta_tildes < 0
                    p_value_mask = np.logical_and(p_value_mask, ~negative_beta_tildes_mask)
                    if np.sum(negative_beta_tildes_mask) > 0:
                        log("Ignoring %d gene sets due to negative beta filters" % (np.sum(negative_beta_tildes_mask)))

                p_value_ignore = np.full(len(p_value_mask), False)
                if filter_gene_set_p < 1 or filter_gene_set_metric_z is not None:

                    p_value_ignore = ~p_value_mask
                    if np.sum(p_value_ignore) > 0:
                        log("Kept %d gene sets after p-value and beta filters" % (np.sum(p_value_mask)))

                    self.gene_sets_ignored = self.gene_sets_ignored + [gene_sets[i] for i in range(len(gene_sets)) if p_value_ignore[i]]
                    gene_sets = [gene_sets[i] for i in range(len(gene_sets)) if p_value_mask[i]]

                    self.col_sums_ignored = np.append(self.col_sums_ignored, self.get_col_sums(cur_X[:,p_value_ignore]))
                    self.scale_factors_ignored = np.append(self.scale_factors_ignored, scale_factors[p_value_ignore])
                    self.mean_shifts_ignored = np.append(self.mean_shifts_ignored, mean_shifts[p_value_ignore])
                    self.beta_tildes_ignored = np.append(self.beta_tildes_ignored, beta_tildes[p_value_ignore])
                    self.p_values_ignored = np.append(self.p_values_ignored, p_values[p_value_ignore])
                    self.ses_ignored = np.append(self.ses_ignored, ses[p_value_ignore])
                    self.z_scores_ignored = np.append(self.z_scores_ignored, z_scores[p_value_ignore])

                    self.beta_tildes = np.append(self.beta_tildes, beta_tildes[p_value_mask])
                    self.p_values = np.append(self.p_values, p_values[p_value_mask])
                    self.ses = np.append(self.ses, ses[p_value_mask])
                    self.z_scores = np.append(self.z_scores, z_scores[p_value_mask])

                    if se_inflation_factors is not None:
                        self.se_inflation_factors_ignored = np.append(self.se_inflation_factors_ignored, se_inflation_factors[p_value_ignore])
                        if self.se_inflation_factors is None:
                            self.se_inflation_factors = np.array([])
                        self.se_inflation_factors = np.append(self.se_inflation_factors, se_inflation_factors[p_value_mask])

                    if self.huge_gene_covariates is not None:
                        if self.total_qc_metrics_ignored is None:
                            self.total_qc_metrics_ignored = total_qc_metrics[p_value_ignore,:]
                            self.mean_qc_metrics_ignored = mean_qc_metrics[p_value_ignore]
                        else:
                            self.total_qc_metrics_ignored = np.vstack((self.total_qc_metrics_ignored, total_qc_metrics[p_value_ignore,:]))
                            self.mean_qc_metrics_ignored = np.append(self.mean_qc_metrics_ignored, mean_qc_metrics[p_value_ignore])

                        total_qc_metrics = total_qc_metrics[p_value_mask]
                        mean_qc_metrics = mean_qc_metrics[p_value_mask]                        

                    #need to record how many ignored
                    gene_ignored_N = self.get_col_sums(cur_X[:,p_value_ignore], axis=1)

                    if cur_X_missing_genes_new is not None:
                        gene_ignored_N_missing_new = np.array(np.abs(cur_X_missing_genes_new[:,p_value_ignore]).sum(axis=1)).flatten()
                        cur_X_missing_genes_new = cur_X_missing_genes_new[:,p_value_mask]

                    if cur_X_missing_genes_int is not None:
                        gene_ignored_N_missing_int = np.array(np.abs(cur_X_missing_genes_int[:,p_value_ignore]).sum(axis=1)).flatten()
                        cur_X_missing_genes_int = cur_X_missing_genes_int[:,p_value_mask]

                    cur_X = cur_X[:,p_value_mask]

            #construct the mean shifts / etc needed for compute beta tildes
            #then call compute beta tildes
            #then call compute betas without V
            #then filter

            self.is_dense_gene_set = np.append(self.is_dense_gene_set, np.full(len(gene_sets), is_dense))


            num_new_gene_sets = len(gene_sets)
            num_old_gene_sets = len(self.gene_sets) if self.gene_sets is not None else 0
            if self.X_orig is not None:
                cur_X = sparse.hstack((self.X_orig, cur_X))
                gene_sets = self.gene_sets + gene_sets

            if self.genes_missing is not None:
                genes += self.genes_missing

                if self.X_orig_missing_genes is None:
                    X_orig_missing_genes = sparse.csc_matrix(([], ([], [])), shape=(len(self.genes_missing), num_old_gene_sets))
                else:
                    X_orig_missing_genes = copy.copy(self.X_orig_missing_genes)                    

                if cur_X_missing_genes_int is not None:
                    if self.gene_ignored_N_missing is not None:
                        if gene_ignored_N_missing_int is not None:
                            self.gene_ignored_N_missing += gene_ignored_N_missing_int
                    else:
                        self.gene_ignored_N_missing = gene_ignored_N_missing_int

                    cur_X = sparse.vstack((cur_X, sparse.hstack((X_orig_missing_genes, cur_X_missing_genes_int))))
                elif X_orig_missing_genes is not None:
                    X_orig_missing_genes.resize((X_orig_missing_genes.shape[0], X_orig_missing_genes.shape[1] + num_new_gene_sets))
                    cur_X = sparse.vstack((cur_X, X_orig_missing_genes))

            if cur_X_missing_genes_new is not None:
                cur_X = sparse.vstack((cur_X, sparse.hstack((sparse.csc_matrix(([], ([], [])), shape=(cur_X_missing_genes_new.shape[0], num_old_gene_sets)), cur_X_missing_genes_new))))
                if self.gene_ignored_N_missing is not None:
                    if gene_ignored_N_missing_new is not None:
                        self.gene_ignored_N_missing = np.append(self.gene_ignored_N_missing, gene_ignored_N_missing_new)
                else:
                    self.gene_ignored_N_missing = gene_ignored_N_missing_new

                genes += genes_missing_new

            #save subset mask for later
            subset_mask = np.full(len(genes), True)
            if self.gene_to_ind is not None:
                subset_mask[[i for i in range(len(genes)) if genes[i] not in self.gene_to_ind]] = False

            #set full X with including new and old missing genes
            
            num_added = cur_X.shape[1]
            if self.X_orig is not None:
                num_added -= self.X_orig.shape[1]
            num_ignored = np.sum(p_value_ignore) if p_value_ignore is not None else 0

            self._set_X(sparse.csc_matrix(cur_X, shape=cur_X.shape), genes, gene_sets, skip_scale_factors=skip_scale_factors, skip_V=True, skip_N=False)

            #have to add ignored_N since this is only place we have the information
            if self.gene_ignored_N is not None:
                if gene_ignored_N is not None:
                    self.gene_ignored_N += gene_ignored_N
            else:
                self.gene_ignored_N = gene_ignored_N

            if self.gene_ignored_N is not None and self.gene_ignored_N_missing is not None:
                self.gene_ignored_N = np.append(self.gene_ignored_N, self.gene_ignored_N_missing)

            #have to call this function to ensure every data structure gets subsetted
            #don't subset Y since we didn't expand these
                    
            self._subset_genes(subset_mask, skip_V=True, overwrite_missing=True, skip_scale_factors=False, skip_Y=True)
            
            if self.huge_gene_covariates is not None:
                if self.total_qc_metrics is None:
                    self.total_qc_metrics = total_qc_metrics
                    self.mean_qc_metrics = mean_qc_metrics
                else:
                    self.total_qc_metrics = np.vstack((self.total_qc_metrics, total_qc_metrics))
                    self.mean_qc_metrics = np.append(self.mean_qc_metrics, mean_qc_metrics)

            return (num_added, num_ignored)

        ignored_gs = 0

        for i in range(len(X_ins)):
            X_in = X_ins[i]

            tag = None
            if ":" in X_in:
                tag_index = X_in.index(":")
                tag = X_in[:tag_index]

                X_in = X_in[tag_index+1:]
                if len(tag) == 0:
                    tag = None

            log("Reading X %d of %d from --X-in file %s" % (i+1,len(X_ins),X_in), INFO)

            genes = []
            gene_sets = []
            cur_X = None

            if is_dense[i]:
                with open_gz(X_in) as gene_sets_fh:
                    header = gene_sets_fh.readline().strip()
                    header = header.lstrip("# \t")
                    gene_sets = header.split()
                    if len(gene_sets) < 2:
                        warn("First line of --Xd-in %s must contain gene column followed by list of gene sets; skipping file" % X_in)
                        continue
                    #if header[0] != "#":
                    #    warn("Assuming first line is header line despite lack of #; first characters are '%s...'" % header[:10])

                    #first column is genes so split
                    gene_sets = gene_sets[1:]

                    cur_X = np.loadtxt(X_in, skiprows=1, dtype=str)
               
                    if cur_X.shape[1] != len(gene_sets) + 1:
                        bail("Xd matrix %s dimensions %s do not match number of gene sets in header line (%s)" % (X_in, cur_X.shape, len(gene_sets)))

                    genes = cur_X[:,0]
                    if self.gene_label_map is not None:
                        genes = list(map(lambda x: self.gene_label_map[x] if x in self.gene_label_map else x, genes))

                    cur_X = cur_X[:,1:].astype(float)

                    if only_ids is not None:
                        gene_set_mask = np.full(len(gene_sets), False)
                        for i in range(len(gene_sets)):
                            if gene_sets[i] in only_ids:
                                gene_set_mask[i] = True
                        if np.any(gene_set_mask):
                            cur_X = cur_X[:,gene_set_mask]
                        else:
                            continue

                    mat_info = cur_X

            else:

                data = []
                row = []
                col = []
                num_read = 0

                new_gene_to_ind = {}
                gene_set_to_ind = {}
                gene_to_ind = None
                if self.genes is not None:
                    #ensure that the matrix always contains all of the current genes
                    #this simplifies code in add_to_X
                    genes = copy.copy(self.genes)
                    if self.genes_missing is not None:
                        genes += self.genes_missing
                    gene_to_ind = self._construct_map_to_ind(genes)

                with open_gz(X_in) as gene_sets_fh:

                    max_num_entries_at_once = 200 * 10000
                    cur_num_read = 0
                    for line in gene_sets_fh:
                        line = line.strip()
                        cols = line.split()

                        if len(cols) < 2:
                            warn("Line does not match format for --X-in: %s" % (line))
                            continue
                        gs = cols[0]

                        if only_ids is not None and gs not in only_ids:
                            continue

                        if gs in gene_set_to_ind or (self.gene_set_to_ind is not None and gs in self.gene_set_to_ind):
                            warn("Gene set %s already seen; skipping" % gs)
                            continue

                        cur_genes = set(cols[1:])
                        if self.gene_label_map is not None:
                            cur_genes = set(map(lambda x: self.gene_label_map[x] if x in self.gene_label_map else x, cur_genes))

                        if len(cur_genes) < min_gene_set_size:
                            #avoid too small gene sets
                            continue

                        #initialize a new location for the gene set
                        gene_set_ind = len(gene_sets)
                        gene_sets.append(gs)
                        #add this to track duplicates in input file
                        gene_set_to_ind[gs] = gene_set_ind

                        for gene in cur_genes:

                            gene_array = gene.split(":")
                            gene = gene_array[0]
                            if len(gene_array) == 2:
                                weight = float(gene_array[1])
                            else:
                                weight = 1.0

                            if gene_to_ind is not None and gene in gene_to_ind:
                                #keep this gene when we harmonize at the end
                                gene_ind = gene_to_ind[gene]
                            else:
                                if gene not in new_gene_to_ind:
                                    gene_ind = len(new_gene_to_ind)                                
                                    if gene_to_ind is not None:
                                        gene_ind += len(gene_to_ind)

                                    new_gene_to_ind[gene] = gene_ind
                                    genes.append(gene)
                                else:
                                    gene_ind = new_gene_to_ind[gene]

                            #store data for the later matrices
                            col.append(gene_set_ind)
                            row.append(gene_ind)
                            data.append(weight)
                        num_read += 1
                        cur_num_read += 1

                        #add at end or when have hit maximum
                        if len(data) >= max_num_entries_at_once:
                            log("Batching %d lines to save memory" % cur_num_read)
                            num_added, num_ignored = __add_to_X((data, row, col), genes, gene_sets, tag, skip_scale_factors=False)
                            if i == 0 and num_added + num_ignored == 0:
                                bail("--first-for-hyper was specified but first file had no gene sets")
                            #add gene set batches here
                            self.gene_set_batches = np.append(self.gene_set_batches, np.full(num_added, batches[i]))
                            self.gene_set_labels = np.append(self.gene_set_labels, np.full(num_added, labels[i]))
                            self.gene_set_labels_ignored = np.append(self.gene_set_labels_ignored, np.full(num_ignored, labels[i]))
                            num_ignored_gene_sets[i] += num_ignored

                            #re-initialize things
                            genes = copy.copy(self.genes)
                            if self.genes_missing is not None:
                                genes += self.genes_missing
                            gene_to_ind = self._construct_map_to_ind(genes)
                            new_gene_to_ind = {}
                            gene_sets = []
                            data = []
                            row = []
                            col = []
                            num_read = 0
                            cur_num_read = 0
                            log("Continuing reading...")
                    #get the end if there are any
                    if len(data) > 0:

                        mat_info = (data, row, col)
                    else:
                        mat_info = None

            if mat_info is not None:
                num_added, num_ignored = __add_to_X(mat_info, genes, gene_sets, tag, skip_scale_factors=False)
                if i == 0 and num_added + num_ignored == 0:
                    bail("--first-for-hyper was specified but first file had no gene sets")
                #add gene set batches here
                self.gene_set_batches = np.append(self.gene_set_batches, np.full(num_added, batches[i]))
                self.gene_set_labels = np.append(self.gene_set_labels, np.full(num_added, labels[i]))
                self.gene_set_labels_ignored = np.append(self.gene_set_labels_ignored, np.full(num_ignored, labels[i]))
                num_ignored_gene_sets[i] += num_ignored

        if self.X_orig is None or self.X_orig.shape[1] == 0:
            log("No gene sets to analyze; returning")
            return

        self._record_param("gene_set_prune_threshold", prune_gene_sets)
        self._prune_gene_sets(prune_gene_sets, keep_missing=False, ignore_missing=True, skip_V=True)

        #if these were not set previously, use the initial values
        if self.p is None:
            self.set_p(initial_p)
        if self.sigma_power is None:
            self.set_sigma(self.sigma2, sigma_power)
        fixed_sigma_cond = False
        if self.sigma2 is None:
            if initial_sigma2_cond is not None:
                #if they specify cond sigma, we set the actual sigma (cond * p) and adjust for scale factors
                if not update_hyper_sigma:
                    fixed_sigma_cond = True
                self.set_sigma(initial_sigma2_cond / np.power(self.MEAN_MOUSE_SCALE, self.sigma_power), self.sigma_power)
            else:
                self.set_sigma(initial_sigma2, self.sigma_power)

        if sigma_soft_threshold_95 is not None and sigma_soft_threshold_5 is not None:
            if sigma_soft_threshold_95 < 0 or sigma_soft_threshold_5 < 0:
                warn("Ignoring sigma soft thresholding since both are not positive")
            else:
                #this will map scale factor to 
                frac_95 = float(sigma_soft_threshold_95) / len(self.genes)
                x1 = np.sqrt(frac_95 * (1 - frac_95))
                y1 = 0.95

                frac_5 = float(sigma_soft_threshold_5) / len(self.genes)
                x2 = np.sqrt(frac_5 * (1 - frac_5))
                y2 = 0.05
                L = 1

                if x2 < x1:
                    warn("--sigma-threshold-5 (%.3g) is less than --sigma-threshold-95 (%.3g); this is the opposite of what you usually want as it will threshold smaller gene sets rather than larger ones")

                self.sigma_threshold_k = -(np.log(1/y2 - L) - np.log(1/y1 - 1))/(x2-x1)
                self.sigma_threshold_xo = (x1 * np.log(1/y2 - L) - x2 * np.log(1/y1 - L)) / (np.log(1/y2 - L) - np.log(1/y1 - L))

                #self.sigma_threshold_xo = (x1 * np.log(L / y2 - 1) - x2 * np.log(L / y1 - 1)) / (np.log(L / y2 - 1) - np.log(L / y1 - 1))
                #self.sigma_threshold_k = -np.log(L / y2 - 1)/ (x2 - self.sigma_threshold_xo)

                log("Thresholding sigma with k=%.3g, xo=%.3g" % (self.sigma_threshold_k, self.sigma_threshold_xo))

        if self.p_values is not None and (update_hyper_p or update_hyper_sigma) and len(self.gene_set_batches) > 0:

            #now learn the hyper values
            assert(self.gene_set_batches[0] is not None)
            #first order the unique batches; batches has one value per file but we need info one per unique batch
            ordered_batches = [self.gene_set_batches[0]] + list(set([x for x in self.gene_set_batches if not x == self.gene_set_batches[0]]))
            #get the total number of ignored genes per batch
            batches_num_ignored = {}
            for i in range(len(batches)):
                if batches[i] not in batches_num_ignored:
                    batches_num_ignored[batches[i]] = 0
                batches_num_ignored[batches[i]] += num_ignored_gene_sets[i]

            self.ps = np.full(len(self.gene_set_batches), np.nan)
            self.sigma2s = np.full(len(self.gene_set_batches), np.nan)

            #none learns from first; rest learn from within themselves
            for ordered_batch_ind in range(len(ordered_batches)):

                if ordered_batches[ordered_batch_ind] is None:
                    #we'll be drawing this from the first
                    assert(first_for_hyper)
                    continue

                gene_sets_in_batch_mask = (self.gene_set_batches == ordered_batches[ordered_batch_ind])

                if ordered_batch_ind > 0 and np.sum(gene_sets_in_batch_mask) + batches_num_ignored[ordered_batches[ordered_batch_ind]] < 100:
                    log("Skipping learning hyper for batch %s since not enough gene sets" % (ordered_batches[ordered_batch_ind]))
                    continue

                #right now the way to pass these is to set member variables, so we save current and set
                orig_ps = self.ps
                orig_sigma2s = self.sigma2s
                #there are always none for running betas here
                self.ps = None
                self.sigma2s = None


                #orig_p = self.p
                #orig_sigma2 = self.sigma2
                #orig_sigma_power = self.sigma_power


                if np.sum(gene_sets_in_batch_mask) > self.batch_size:
                    V = None
                else:
                    V = self._calculate_V_internal(self.X_orig[:,gene_sets_in_batch_mask], self.y_corr_cholesky, self.mean_shifts[gene_sets_in_batch_mask], self.scale_factors[gene_sets_in_batch_mask])

                #run non_inf_betas
                #only add psuedo counts for large values
                num_p_pseudo = min(1, np.sum(gene_sets_in_batch_mask) / 1000)

                #adjust sigma means keep sigma/p constant (thereby adjusting unconditional variance=sigma)
                #if it is the first batch and first_for_hyper, we do not want to adjust the sigma
                #similarly, if it is not first_for_hyper, we do not want to adjust the sigma
                #we will learn it (if requested), but if not requested we assume that the specified sigma is the correct *UNCONDITIONAL* variance
                #thus, we will learn p subject to this constraint on total variance
                #after the first batch, however, when doing first_for_hyper, we will adjust sigma to keep the sigma/p fixed
                cur_update_hyper_p = update_hyper_p
                cur_update_hyper_sigma = update_hyper_sigma
                adjust_hyper_sigma_p = False
                if (first_for_sigma_cond and ordered_batch_ind > 0) or fixed_sigma_cond:
                    adjust_hyper_sigma_p = True
                    if cur_update_hyper_p:
                        cur_update_hyper_sigma = False
                Y_to_use = self.Y_for_regression
                Y = np.exp(Y_to_use + self.background_log_bf) / (1 + np.exp(Y_to_use + self.background_log_bf))

                (betas, avg_postp) = self._calculate_non_inf_betas(initial_p=None, beta_tildes=self.beta_tildes[gene_sets_in_batch_mask], ses=self.ses[gene_sets_in_batch_mask], V=V, X_orig=self.X_orig[:,gene_sets_in_batch_mask], scale_factors=self.scale_factors[gene_sets_in_batch_mask], mean_shifts=self.mean_shifts[gene_sets_in_batch_mask], is_dense_gene_set=self.is_dense_gene_set[gene_sets_in_batch_mask], ps=None, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter_betas, min_num_iter=min_num_iter_betas, num_chains=num_chains_betas, r_threshold_burn_in=r_threshold_burn_in_betas, use_max_r_for_convergence=use_max_r_for_convergence_betas, max_frac_sem=max_frac_sem_betas, max_allowed_batch_correlation=max_allowed_batch_correlation, gauss_seidel=False, update_hyper_sigma=cur_update_hyper_sigma, update_hyper_p=cur_update_hyper_p, adjust_hyper_sigma_p=adjust_hyper_sigma_p, sigma_num_devs_to_top=sigma_num_devs_to_top, p_noninf_inflate=p_noninf_inflate, num_p_pseudo=num_p_pseudo, num_missing_gene_sets=batches_num_ignored[ordered_batches[ordered_batch_ind]], sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, betas_trace_out=betas_trace_out, betas_trace_gene_sets=[self.gene_sets[j] for j in range(len(self.gene_sets)) if gene_sets_in_batch_mask[j]])

                #now save and restore
                computed_p = self.p
                computed_sigma2 = self.sigma2
                computed_sigma_power = self.sigma_power

                #don't reset 
                #if not first_for_sigma_cond and ordered_batch_ind > 0:
                #    self.set_p(orig_p)
                #    self.set_sigma(orig_sigma2, orig_sigma_power)

                self.ps = orig_ps
                self.sigma2s = orig_sigma2s

                log("Learned p=%.4g, sigma2=%.4g (sigma2/p=%.4g)" % (computed_p, computed_sigma2, computed_sigma2/computed_p))
                self._record_params({"p": computed_p, "sigma2": computed_sigma2, "sigma2_cond": computed_sigma2/computed_p, "sigma_power": computed_sigma_power, "sigma_threshold_k": self.sigma_threshold_k, "sigma_threshold_xo": self.sigma_threshold_xo})

                self.ps[gene_sets_in_batch_mask] = computed_p
                self.sigma2s[gene_sets_in_batch_mask] = computed_sigma2

            #take care of the missing ps

            assert(len(self.ps) > 0 and not np.isnan(self.ps[0]))
            assert(len(self.sigma2s) > 0 and not np.isnan(self.sigma2s[0]))

            if first_for_hyper:
                self.ps[np.isnan(self.ps)] = self.ps[0]
                self.sigma2s[np.isnan(self.sigma2s)] = self.sigma2s[0]
            else:
                #this should only occur if the gene sets were too small
                self.ps[np.isnan(self.ps)] = np.mean(self.ps[~np.isnan(self.ps)])
                self.sigma2s[np.isnan(self.sigma2s)] = np.mean(self.sigma2s[~np.isnan(self.sigma2s)])

            self.set_p(np.mean(self.ps))
            self.set_sigma(np.mean(self.sigma2s), self.sigma_power)
            
            #if shared_sigma_cond:
            #    #we want sigma2/self.p2 to be constant
            #    self.sigma2s = self.sigma2 * self.ps / self.p

        if filter_gene_set_p is not None and increase_filter_gene_set_p is not None and self.p_values is not None and self.p_values_ignored is not None:
            #since we required each batch to have increase_filter_gene_set_p, maybe we need to reduce
            if float(len(self.p_values)) / (len(self.p_values) + len(self.p_values_ignored)) > increase_filter_gene_set_p:
                #choose a potentially more strict threshold
                #want keep_frac * len(self.p_values) / (len(self.p_values) + len(self.p_values_ignored)) = filter_gene_set_p
                keep_frac = increase_filter_gene_set_p * float(len(self.p_values) + len(self.p_values_ignored)) / len(self.p_values)
                p_from_quantile = np.quantile(self.p_values, keep_frac)
                if p_from_quantile > filter_gene_set_p:
                    overcorrect_ignore = self.p_values > p_from_quantile
                    if np.sum(overcorrect_ignore) > 0:
                        overcorrect_mask = ~overcorrect_ignore
                        self._record_param("adjusted_filter_gene_set_p", p_from_quantile)
                        log("Ignoring %d gene sets due to p > %.3g (overaggressive adjustment of p-value filters; kept %d)" % (np.sum(overcorrect_ignore), p_from_quantile, np.sum(overcorrect_mask)))
                        self._subset_gene_sets(overcorrect_mask, ignore_missing=True, keep_missing=False, skip_V=True)

        #do another check of min_gene_set_size in case we converted some gene sets with weights
        if self.X_orig is not None:
            col_sums = self.get_col_sums(self.X_orig, num_nonzero=True)
            size_ignore = col_sums < min_gene_set_size

            if np.sum(size_ignore) > 0:
                size_mask = ~size_ignore
                log("Ignoring %d gene sets due to too few genes (kept %d)" % (np.sum(size_ignore), np.sum(size_mask)))
                self._subset_gene_sets(size_mask, keep_missing=False, skip_V=True)

            col_sums = self.get_col_sums(self.X_orig, num_nonzero=True)
            size_ignore = col_sums > max_gene_set_size
            if np.sum(size_ignore) > 0:
                size_mask = ~size_ignore
                log("Ignoring %d gene sets due to too many genes (kept %d)" % (np.sum(size_ignore), np.sum(size_mask)))
                self._subset_gene_sets(size_mask, keep_missing=False, skip_V=True)


            if self.total_qc_metrics is not None:
                total_qc_metrics = self.total_qc_metrics
                if self.total_qc_metrics_ignored is not None:
                    total_qc_metrics = np.vstack((self.total_qc_metrics, self.total_qc_metrics_ignored))

                self.total_qc_metrics = (self.total_qc_metrics - np.mean(total_qc_metrics, axis=0)) / np.std(total_qc_metrics, axis=0)
                if self.total_qc_metrics_ignored is not None:
                    self.total_qc_metrics_ignored = (self.total_qc_metrics_ignored - np.mean(total_qc_metrics, axis=0)) / np.std(total_qc_metrics, axis=0)

            if self.mean_qc_metrics is not None:
                mean_qc_metrics = np.append(self.mean_qc_metrics, self.mean_qc_metrics_ignored if self.mean_qc_metrics_ignored is not None else [])
                self.mean_qc_metrics = (self.mean_qc_metrics - np.mean(mean_qc_metrics)) / np.std(mean_qc_metrics)
                if self.mean_qc_metrics_ignored is not None:
                    self.mean_qc_metrics_ignored = (self.mean_qc_metrics_ignored - np.mean(mean_qc_metrics)) / np.std(mean_qc_metrics)

                if filter_gene_set_metric_z:
                    filter_mask = np.abs(self.mean_qc_metrics) < filter_gene_set_metric_z
                    filter_ignore = ~filter_mask
                    log("Ignoring %d gene sets due to QC metric filters (kept %d)" % (np.sum(filter_ignore), np.sum(filter_mask)))
                    self._subset_gene_sets(filter_mask, keep_missing=False, ignore_missing=True, skip_V=True)

                #print("TEMP")
                #self.total_qc_metrics = np.vstack((self.mean_qc_metrics, np.ones(len(self.mean_qc_metrics)))).T
                #self.total_qc_metrics_ignored = np.vstack((self.mean_qc_metrics_ignored, np.ones(len(self.mean_qc_metrics_ignored)))).T


        sort_rank = self.p_values
        if self.p_values is not None and filter_gene_set_p < 1:
            #remove those that have uncorrected beta equal to zero
            (betas, avg_postp) = self._calculate_non_inf_betas(initial_p=None, assume_independent=True, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter_betas, min_num_iter=min_num_iter_betas, num_chains=num_chains_betas, r_threshold_burn_in=r_threshold_burn_in_betas, use_max_r_for_convergence=use_max_r_for_convergence_betas, max_frac_sem=max_frac_sem_betas, max_allowed_batch_correlation=max_allowed_batch_correlation, gauss_seidel=False, update_hyper_sigma=False, update_hyper_p=False, adjust_hyper_sigma_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas)
            #for i in range(len(self.gene_sets)):
            #    print("1\t%s\t%.3g\t%.3g" % (self.gene_sets[i], betas[i], avg_postp[i]))

            beta_ignore = betas == 0
            beta_mask = ~beta_ignore
            log("Ignoring %d gene sets due to zero uncorrected betas (kept %d)" % (np.sum(beta_ignore), np.sum(beta_mask)))
            self._subset_gene_sets(beta_mask, keep_missing=False, ignore_missing=True, skip_V=True)
            sort_rank = -np.abs(betas[beta_mask])

        if max_num_gene_sets is not None and len(self.gene_sets) > max_num_gene_sets and max_num_gene_sets > 0:
            log("Current %d gene sets is greater than maximum specified %d; reducing using pruning + small beta removal" % (len(self.gene_sets), max_num_gene_sets), DEBUG)
            gene_set_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_orig, mean_shifts=self.mean_shifts, scale_factors=self.scale_factors, sort_values=sort_rank, stop_at=max_num_gene_sets)
            keep_mask = np.full(len(self.gene_sets), False)
            for gene_set_mask in gene_set_masks:
                keep_mask[gene_set_mask] = True
                log("Adding %d relatively uncorrelatd gene sets (total now %d)" % (np.sum(gene_set_mask), np.sum(keep_mask)), TRACE)
                if np.sum(keep_mask) > max_num_gene_sets:
                    break
            if np.sum(keep_mask) > max_num_gene_sets:
                threshold_value = sorted(sort_rank[keep_mask])[max_num_gene_sets - 1]
                keep_mask[sort_rank > threshold_value] = False
            if np.sum(~keep_mask) > 0:
                self._subset_gene_sets(keep_mask, keep_missing=False, ignore_missing=True, skip_V=True)

        self._record_param("num_gene_sets_read", len(self.gene_sets))
        self._record_param("num_genes_read", len(self.genes))

        log("Read %d gene sets and %d genes" % (len(self.gene_sets), len(self.genes)))
           
    #this reads a V matrix directly from a file
    #it does not initialize an X matrix; if the X-matrix is needed, read_X should be used instead
    def read_V(self, V_in):

        log("Reading V from --V-in file %s" % V_in, INFO)
        with open(V_in) as V_fh:
            header = V_fh.readline().strip()
        if len(header) == 0 or header[0] != "#":
            bail("First line of --V-in must be proceeded by #")
        header = header.lstrip("# \t")
        gene_sets = header.split()
        if len(gene_sets) < 1:
            bail("First line of --X-in must contain list of gene sets")

        gene_set_to_ind = self._construct_map_to_ind(gene_sets)
        V = np.genfromtxt(V_in, skip_header=1)
        if V.shape[0] != V.shape[1] or V.shape[0] != len(gene_sets):
            bail("V matrix dimensions %s do not match number of gene sets in header line (%s)" % (V.shape, len(gene_sets)))

        if self.gene_sets is None:
            self.gene_sets = gene_sets
            self.gene_set_to_ind = gene_set_to_ind
        else:
            #first remove everything from V that is not in gene sets previously
            subset_mask = np.array([(x in self.gene_set_to_ind) for x in gene_sets])
            if sum(subset_mask) != len(subset_mask):
                warn("Excluding %s values from previously loaded files because absent from --V-in file" % (len(subset_mask) - sum(subset_mask)))
                V = V[subset_mask,:][:,subset_mask]
                self.gene_sets = list(itertools.compress(self.gene_sets, subset_mask))
                self.gene_set_to_ind = self._construct_map_to_ind(self.gene_sets)
            #now remove everything from the other files that are not in V
            old_subset_mask = np.array([(x in gene_set_to_ind) for x in self.gene_sets])
            if sum(old_subset_mask) != len(old_subset_mask):
                warn("Excluding %s values from --V-in file because absent from previously loaded files" % (len(old_subset_mask) - sum(old_subset_mask)))
                self._subset_gene_sets(old_subset_mask, keep_missing=False, skip_V=True)
        return V


    def write_V(self, V_out):
        if self.X_orig is not None:
            V = self._get_V()
            log("Writing V matrix to %s" % V_out, INFO)
            np.savetxt(V_out, V, delimiter='\t', fmt="%.2g", comments="#", header="%s" % ("\t".join(self.gene_sets)))
        else:
            warn("V has not been initialized; skipping writing")

    def write_Xd(self, X_out):
        if self.X_orig is not None:
            log("Writing X matrix to %s" % X_out, INFO)
            #FIXME: get_orig_X
            np.savetxt(X_out, self.X_orig.toarray(), delimiter='\t', fmt="%.3g", comments="#", header="%s" % ("%s\n#%s" % ("\t".join(self.gene_sets), "\t".join(self.genes))))
        else:
            warn("X has not been initialized; skipping writing")

    def write_X(self, X_out):
        if self.genes is None or self.X_orig is None or self.gene_sets is None:
            return
            warn("X has not been initialized; skipping writing")
            return

        log("Writing X sparse matrix to %s" % X_out, INFO)

        with open_gz(X_out, 'w') as output_fh:

            for j in range(len(self.gene_sets)):
                line = self.gene_sets[j]
                nonzero_inds = self.X_orig[:,j].nonzero()[0]
                non_unity = np.sum(self.X_orig[nonzero_inds,j] == 1) < len(nonzero_inds)
                for i in nonzero_inds:
                    if non_unity:
                        line = "%s\t%s:%.2g" % (line, self.genes[i], self.X_orig[i,j])
                    else:
                        line = "%s\t%s" % (line, self.genes[i])
                
                output_fh.write("%s\n" % line)

    def calculate_huge_scores_gwas(self, gwas_in, gwas_chrom_col=None, gwas_pos_col=None, gwas_p_col=None, gene_loc_file=None, hold_out_chrom=None, exons_loc_file=None, gwas_beta_col=None, gwas_se_col=None, gwas_n_col=None, gwas_n=None, gwas_freq_col=None, gwas_locus_col=None, gwas_ignore_p_threshold=None, gwas_units=None, gwas_low_p=5e-8, gwas_high_p=1e-2, gwas_low_p_posterior=0.98, gwas_high_p_posterior=0.001, detect_low_power=None, detect_high_power=None, detect_adjust_huge=False, learn_window=False, closest_gene_prob=0.7, max_closest_gene_prob=0.9, scale_raw_closest_gene=True, cap_raw_closest_gene=False, cap_region_posterior=True, scale_region_posterior=False, phantom_region_posterior=False, allow_evidence_of_absence=False, correct_huge=True, max_signal_p=1e-5, signal_window_size=250000, signal_min_sep=100000, signal_max_logp_ratio=None, credible_set_span=25000, max_closest_gene_dist=2.5e5, min_n_ratio=0.5, max_clump_ld=0.2, min_var_posterior=0.01, s2g_in=None, s2g_chrom_col=None, s2g_pos_col=None, s2g_gene_col=None, s2g_prob_col=None, credible_sets_in=None, credible_sets_id_col=None, credible_sets_chrom_col=None, credible_sets_pos_col=None, credible_sets_ppa_col=None, **kwargs):
        if gwas_in is None:
            bail("Require --gwas-in for this operation")
        if gene_loc_file is None:
            bail("Require --gene-loc-file for this operation")

        if credible_sets_in is not None:
            if credible_sets_id_col is None:
                bail("Need --credible-set-id-col")


        if signal_window_size < 2 * signal_min_sep:
            signal_window_size = 2 * signal_min_sep

        if signal_max_logp_ratio is not None:
            if signal_max_logp_ratio > 1:
                warn("Thresholding --signal-max-logp-ratio at 1")
                signal_max_logp_ratio = 1

        self._record_params({"gwas_low_p": gwas_low_p, "gwas_high_p": gwas_high_p, "gwas_low_p_posterior": gwas_low_p_posterior, "gwas_high_p_posterior": gwas_high_p_posterior, "detect_low_power": detect_low_power, "detect_high_power": detect_high_power, "detect_adjust_huge": detect_adjust_huge, "closest_gene_prob": closest_gene_prob, "max_closest_gene_prob": max_closest_gene_prob, "scale_raw_closest_gene": scale_raw_closest_gene, "cap_raw_closest_gene": cap_raw_closest_gene, "cap_region_posterior": cap_region_posterior, "scale_region_posterior": scale_region_posterior, "max_signal_p": max_signal_p, "signal_window_size": signal_window_size, "signal_min_sep": signal_min_sep, "max_closest_gene_dist": max_closest_gene_dist, "min_n_ratio": min_n_ratio})

        #see if need to determine
        if (gwas_pos_col is None or gwas_chrom_col is None) and gwas_locus_col is None:
            need_columns = True
        else:
            has_se = gwas_se_col is not None or gwas_n_col is not None or gwas_n is not None
            if (gwas_p_col is not None and gwas_beta_col is not None) or (gwas_p_col is not None and has_se) or (gwas_beta_col is not None and has_se):
                need_columns = False
            else:
                need_columns = True

        if need_columns:
            (possible_gene_id_cols, possible_var_id_cols, possible_chrom_cols, possible_pos_cols, possible_locus_cols, possible_p_cols, possible_beta_cols, possible_se_cols, possible_freq_cols, possible_n_cols, header) = self._determine_columns(gwas_in)

            #now recompute
            if gwas_pos_col is None:
                if len(possible_pos_cols) == 1:
                    gwas_pos_col = possible_pos_cols[0]
                    log("Using %s for position column; change with --gwas-pos-col if incorrect" % gwas_pos_col)
                else:
                    log("Could not determine position column from header %s; specify with --gwas-pos-col" % header)
            if gwas_chrom_col is None:
                if len(possible_chrom_cols) == 1:
                    gwas_chrom_col = possible_chrom_cols[0]
                    log("Using %s for chrom column; change with --gwas-chrom-col if incorrect" % gwas_chrom_col)
                else:
                    log("Could not determine chrom column from header %s; specify with --gwas-chrom-col" % header)
            if (gwas_pos_col is None or gwas_chrom_col is None) and gwas_locus_col is None:
                if len(possible_locus_cols) == 1:
                    gwas_locus_col = possible_locus_cols[0]
                    log("Using %s for locus column; change with --gwas-locus-col if incorrect" % gwas_locus_col)
                else:
                    bail("Could not determine chrom and pos columns from header %s; specify with --gwas-chrom-col and --gwas-pos-col or with --gwas-locus-col" % header)

            if gwas_p_col is None:
                if len(possible_p_cols) == 1:
                    gwas_p_col = possible_p_cols[0]
                    log("Using %s for p column; change with --gwas-p-col if incorrect" % gwas_p_col)
                else:
                    log("Could not determine p column from header %s; if desired specify with --gwas-p-col" % header)
            if gwas_se_col is None:
                if len(possible_se_cols) == 1:
                    gwas_se_col = possible_se_cols[0]
                    log("Using %s for se column; change with --gwas-se-col if incorrect" % gwas_se_col)
                else:
                    log("Could not determine se column from header %s; if desired specify with --gwas-se-col" % header)
            if gwas_beta_col is None:
                if len(possible_beta_cols) == 1:
                    gwas_beta_col = possible_beta_cols[0]
                    log("Using %s for beta column; change with --gwas-beta-col if incorrect" % gwas_beta_col)
                else:
                    log("Could not determine beta column from header %s; if desired specify with --gwas-beta-col" % header)

            if gwas_n_col is None:
                if len(possible_n_cols) == 1:
                    gwas_n_col = possible_n_cols[0]
                    log("Using %s for N column; change with --gwas-n-col if incorrect" % gwas_n_col)
                else:
                    log("Could not determine N column from header %s; if desired specify with --gwas-n-col" % header)

            if gwas_freq_col is None:
                if len(possible_freq_cols) == 1:
                    gwas_freq_col = possible_freq_cols[0]
                    log("Using %s for freq column; change with --gwas-freq-col if incorrect" % gwas_freq_col)

            has_se = gwas_se_col is not None
            has_n = gwas_n_col is not None or gwas_n is not None
            if (gwas_p_col is not None and gwas_beta_col is not None) or (gwas_p_col is not None and (has_se or has_n)) or (gwas_beta_col is not None and has_se):
                pass
            else:
                bail("Require information about p-value and se or N or beta, or beta and se; specify with --gwas-p-col, --gwas-beta-col, and --gwas-se-col")

            if options.debug_just_check_header:
                bail("Done checking headers")


        #use this to store the exons 
        class IntervalTree(object):
            __slots__ = ('interval_starts', 'interval_stops', 'left', 'right', 'center')
            def __init__(self, intervals, depth=16, minbucket=96, _extent=None, maxbucket=4096):
                depth -= 1
                if (depth == 0 or len(intervals) < minbucket) and len(intervals) > maxbucket:
                    self.interval_starts, self.interval_stops = zip(*intervals)
                    self.left = self.right = None
                    return

                left, right = _extent or (min(i[0] for i in intervals), max(i[1] for i in intervals))
                center = (left + right) / 2.0

                self.interval_starts = []
                self.interval_stops = []
                lefts, rights  = [], []

                for interval in intervals:
                    if interval[1] < center:
                        lefts.append(interval)
                    elif interval[0] > center:
                        rights.append(interval)
                    else: # overlapping.
                        self.interval_starts.append(interval[0])
                        self.interval_stops.append(interval[1])

                self.interval_starts = np.array(self.interval_starts)
                self.interval_stops = np.array(self.interval_stops)
                self.left   = lefts  and IntervalTree(lefts,  depth, minbucket, (left,  center)) or None
                self.right  = rights and IntervalTree(rights, depth, minbucket, (center, right)) or None
                self.center = center

            def find(self, start, stop, index_map=None):

                #find overlapping intervals for set of (start, stop) pairs
                #start is array of starting points
                #stop is array of corresponding stopping points
                #return list of two np arrays. First of passed in indices for which there is an overlap (indices can be repeated)
                #second a list of intervals that overlap
                """find all elements between (or overlapping) start and stop"""
                #array with rows equal to intervals length, columns equal to stop, true if intervals less than or equal to stop
                less_mask = np.less(self.interval_starts, stop[:,np.newaxis] + 1)
                #array with rows equal to intervals length, columns equal to stop, true if intervals greater than or equal to stop
                greater_mask = np.greater(self.interval_stops, start[:,np.newaxis] - 1)
                #interval x variant pos array with the intervals that overlap each variant pos
                overlapping_mask = np.logical_and(less_mask, greater_mask)

                #tuple of (overlapping interval indices, passed in start/stop index with the overlap)
                overlapping_where = np.where(overlapping_mask)

                overlapping_indices = (overlapping_where[0], self.interval_starts[overlapping_where[1]], self.interval_stops[overlapping_where[1]])
                #overlapping = [i for i in self.intervals if i[1] >= start and i[0] <= stop]

                start_less_mask = start <= self.center
                if self.left and np.any(start_less_mask):
                    left_overlapping_indices = self.left.find(start[start_less_mask], stop[start_less_mask], index_map=np.where(start_less_mask)[0])
                    overlapping_indices = (np.append(overlapping_indices[0], left_overlapping_indices[0]), np.append(overlapping_indices[1], left_overlapping_indices[1]), np.append(overlapping_indices[2], left_overlapping_indices[2]))

                stop_greater_mask = stop >= self.center
                if self.right and np.any(stop_greater_mask):
                    right_overlapping_indices = self.right.find(start[stop_greater_mask], stop[stop_greater_mask], index_map=np.where(stop_greater_mask)[0])
                    overlapping_indices = (np.append(overlapping_indices[0], right_overlapping_indices[0]), np.append(overlapping_indices[1], right_overlapping_indices[1]), np.append(overlapping_indices[2], right_overlapping_indices[2]))

                if index_map is not None and len(overlapping_indices[0]) > 0:
                    overlapping_indices = (index_map[overlapping_indices[0]], overlapping_indices[1], overlapping_indices[2])

                return overlapping_indices


        #store the gene locations
        log("Reading gene locations")
        (gene_chrom_name_pos, gene_to_chrom, gene_to_pos) = self._read_loc_file(gene_loc_file, hold_out_chrom=hold_out_chrom)

        for chrom in gene_chrom_name_pos:
            serialized_gene_info = []
            for gene in gene_chrom_name_pos[chrom]:
                for pos in gene_chrom_name_pos[chrom][gene]:
                    serialized_gene_info.append((gene,pos))
            gene_chrom_name_pos[chrom] = serialized_gene_info 

        chrom_to_interval_tree = None
        if exons_loc_file is not None:

            log("Reading exon locations")

            chrom_interval_to_gene = self._read_loc_file(exons_loc_file, return_intervals=True)
            chrom_to_interval_tree = {}
            for chrom in chrom_interval_to_gene:
                chrom_to_interval_tree[chrom] = IntervalTree(chrom_interval_to_gene[chrom].keys())

        (allelic_var_k, gwas_prior_odds) = self.compute_allelic_var_and_prior(gwas_high_p, gwas_high_p_posterior, gwas_low_p, gwas_low_p_posterior)
        #this stores the original values, in case we detect low or high power
        (allelic_var_k_detect, gwas_prior_odds_detect) = (allelic_var_k, gwas_prior_odds)

        var_z_threshold = None
        var_p_threshold = None
        if min_var_posterior is not None and min_var_posterior > 0:
            #var_log_bf + np.log(gwas_prior_odds) > np.log(min_var_posterior / (1 - min_var_posterior))
            #var_log_bf > np.log(min_var_posterior / (1 - min_var_posterior)) - np.log(gwas_prior_odds) = threshold
            #-np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(var_z) * allelic_var_k / (1 + allelic_var_k) > threshold
            #0.5 * np.square(var_z) * allelic_var_k / (1 + allelic_var_k) > threshold + np.log(np.sqrt(1 + allelic_var_k))
            #np.square(var_z) * allelic_var_k / (1 + allelic_var_k) > 2 * (threshold + np.log(np.sqrt(1 + allelic_var_k)))
            #np.square(var_z) * allelic_var_k > 2 * (1 + allelic_var_k) * (threshold + np.log(np.sqrt(1 + allelic_var_k)))
            log_bf_threshold = np.log(min_var_posterior / (1 - min_var_posterior)) - np.log(gwas_prior_odds) + np.log(np.sqrt(1 + allelic_var_k))
            if log_bf_threshold > 0:
                var_z_threshold = np.sqrt(2 * (1 + allelic_var_k) * (log_bf_threshold) / allelic_var_k)
                var_p_threshold = 2*scipy.stats.norm.cdf(-np.abs(var_z_threshold))
                log("Keeping only variants with p < %.4g" % var_p_threshold)

        log("Reading --gwas-in file %s" % gwas_in, INFO)

        with open_gz(gwas_in) as gwas_fh:
            header_cols = gwas_fh.readline().strip().split()

            chrom_col = None
            pos_col = None
            locus_col = None
            if gwas_chrom_col is not None and gwas_pos_col is not None:
                chrom_col = self._get_col(gwas_chrom_col, header_cols)
                pos_col = self._get_col(gwas_pos_col, header_cols)
            else:
                locus_col = self._get_col(gwas_locus_col, header_cols)

            p_col = None
            if gwas_p_col is not None:
                p_col = self._get_col(gwas_p_col, header_cols)

            beta_col = None
            if gwas_beta_col is not None:
                beta_col = self._get_col(gwas_beta_col, header_cols)

            n_col = None
            se_col = None
            if gwas_n_col is not None:
                n_col = self._get_col(gwas_n_col, header_cols)
            if gwas_se_col is not None:
                se_col = self._get_col(gwas_se_col, header_cols)

            freq_col = None
            if gwas_freq_col is not None:
                freq_col = self._get_col(gwas_freq_col, header_cols)

            chrom_pos_p_beta_se_freq = {}
            seen_chrom_pos = {}

            if (chrom_col is None or pos_col is None) and locus_col is None:
                bail("Operation requires --gwas-chrom-col and --gwas-pos-col or --gwas-locus-col")

            #read in the gwas associations
            total_num_vars = 0

            mean_n = 0

            warned_pos = False
            warned_stats = False
            
            for line in gwas_fh:

                #TODO: allow a separate snp-loc file to be used

                cols = line.strip().split()
                if (chrom_col is not None and chrom_col > len(cols)) or (pos_col is not None and pos_col > len(cols)) or (locus_col is not None and locus_col > len(cols)) or (p_col is not None and p_col > len(cols)) or (se_col is not None and se_col > len(cols)) or (n_col is not None and n_col > len(cols)) or (freq_col is not None and freq_col > len(cols)):
                    warn("Skipping line do to too few columns: %s" % line)
                    continue

                if chrom_col is not None and pos_col is not None:
                    chrom = cols[chrom_col]
                    pos = cols[pos_col]
                else:
                    locus = cols[locus_col]
                    locus_tokens = None
                    for locus_delim in [":", "_"]:
                        if locus_delim in locus:
                            locus_tokens = locus.split(locus_delim)
                            break
                    if locus_tokens is None or len(locus_tokens) <= 2:
                        bail("Could not split locus %s on either : or _" % locus)
                    chrom = locus_tokens[0]
                    pos = locus_tokens[1]
                
                if hold_out_chrom is not None and chrom == hold_out_chrom:
                    continue
                try:
                    pos = int(pos)
                except ValueError:
                    if not warned_pos:
                        warn("Skipping unconvertible pos value %s" % (cols[pos_col]))
                        warned_pos = True

                    continue

                p = None
                if p_col is not None:
                    try:
                        p = float(cols[p_col])
                    except ValueError:
                        if not cols[p_col] == "NA":
                            if not warned_stats:
                                warn("Skipping unconvertible p value %s" % (cols[p_col]))
                                warned_stats = True
                        continue

                    min_p = 1e-250
                    if p < min_p:
                        p = min_p

                    if p <= 0 or p > 1:
                        if not warned_stats:
                            warn("Skipping invalid p value %s" % (p))
                            warned_stats = True
                        continue

                    if gwas_ignore_p_threshold is not None and p > gwas_ignore_p_threshold:
                        continue

                beta = None
                if beta_col is not None:
                    try:
                        beta = float(cols[beta_col])
                    except ValueError:
                        if not cols[beta_col] == "NA":
                            if not warned_stats:
                                warn("Skipping unconvertible beta value %s" % (cols[beta_col]))
                                warned_stats = True
                        continue

                se = None
                if se_col is not None:
                    try:
                        se = float(cols[se_col])
                    except ValueError:
                        if not cols[se_col] == "NA":
                            if not warned_stats:
                                warn("Skipping unconvertible se value %s" % (cols[se_col]))
                                warned_stats = True
                        continue
                elif n_col is not None:
                    try:
                        n = float(cols[n_col])
                    except ValueError:
                        if not cols[n_col] == "NA":
                            if not warned_stats:
                                warn("Skipping unconvertible n value %s" % (cols[n_col]))
                                warned_stats = True
                        continue
                        
                    if n <= 0:
                        if not warned_stats:
                            warn("Skipping invalid N value %s" % (n))
                            warned_stats = True
                        continue
                    se = 1 / np.sqrt(n)
                elif gwas_n is not None:
                    if gwas_n <= 0:
                        bail("Invalid gwas-n value: %s" % (gwasa_n))
                        continue
                    n = gwas_n
                    se = 1 / np.sqrt(n)
                else:
                    assert(p is not None and beta is not None)

                if var_z_threshold is not None:
                    if p is not None:
                        if p > var_p_threshold:
                            continue
                    else:
                        z = np.abs(beta / se)
                        if z < var_z_threshold:
                            continue
                    
                freq = None
                if freq_col is not None:
                    try:
                        freq = float(cols[freq_col])
                    except ValueError:
                        if not cols[freq_col] == "NA":
                            warn("Skipping unconvertible n value %s" % (cols[freq_col]))
                        continue
                    if freq > 1 or freq < 0:
                        warn("Skipping invalid freq value %s" % freq)

                if chrom not in chrom_pos_p_beta_se_freq:
                    chrom_pos_p_beta_se_freq[chrom] = []

                chrom_pos_p_beta_se_freq[chrom].append((pos, p, beta, se, freq))
                if chrom not in seen_chrom_pos:
                    seen_chrom_pos[chrom] = set()
                seen_chrom_pos[chrom].add(pos)

                total_num_vars += 1

            log("Read in %d variants" % total_num_vars)
            chrom_pos_to_gene_prob = None
            if s2g_in is not None:
                chrom_pos_to_gene_prob = {}

                log("Reading --s2g-in file %s" % s2g_in, INFO)

                #see if need to determine
                if s2g_pos_col is None or s2g_chrom_col is None or s2g_gene_col is None:
                    (possible_s2g_gene_cols, possible_s2g_var_id_cols, possible_s2g_chrom_cols, possible_s2g_pos_cols, possible_s2g_locus_cols, possible_s2g_p_cols, possible_s2g_beta_cols, possible_s2g_se_cols, possible_s2g_freq_cols, possible_s2g_n_cols) = self._determine_columns(s2g_in)

                    if s2g_pos_col is None:
                        if len(possible_s2g_pos_cols) == 1:
                            s2g_pos_col = possible_s2g_pos_cols[0]
                            log("Using %s for position column; change with --s2g-pos-col if incorrect" % s2g_pos_col)
                        else:
                            bail("Could not determine position column; specify with --s2g-pos-col")
                    if s2g_chrom_col is None:
                        if len(possible_s2g_chrom_cols) == 1:
                            s2g_chrom_col = possible_s2g_chrom_cols[0]
                            log("Using %s for chromition column; change with --s2g-chrom-col if incorrect" % s2g_chrom_col)
                        else:
                            bail("Could not determine chrom column; specify with --s2g-chrom-col")
                    if s2g_gene_col is None:
                        if len(possible_s2g_gene_cols) == 1:
                            s2g_gene_col = possible_s2g_gene_cols[0]
                            log("Using %s for geneition column; change with --s2g-gene-col if incorrect" % s2g_gene_col)
                        else:
                            bail("Could not determine gene column; specify with --s2g-gene-col")

                with open_gz(s2g_in) as s2g_fh:
                    header_cols = s2g_fh.readline().strip().split()
                    chrom_col = self._get_col(s2g_chrom_col, header_cols)
                    pos_col = self._get_col(s2g_pos_col, header_cols)
                    gene_col = self._get_col(s2g_gene_col, header_cols)
                    prob_col = None
                    if s2g_prob_col is not None:
                        prob_col = self._get_col(s2g_prob_col, header_cols)

                    for line in s2g_fh:

                        cols = line.strip().split()
                        if chrom_col > len(cols) or pos_col > len(cols) or gene_col > len(cols) or (prob_col is not None and prob_col > len(cols)):
                            warn("Skipping due to too few columns in line: %s" % line)
                            continue

                        chrom = cols[chrom_col]
                        if hold_out_chrom is not None and chrom == hold_out_chrom:
                            continue

                        try:
                            pos = int(cols[pos_col])
                        except ValueError:
                            warn("Skipping unconvertible pos value %s" % (cols[pos_col]))
                            continue
                        gene = cols[gene_col]

                        if gene in self.gene_label_map:
                            gene = self.gene_label_map[gene]

                        max_s2g_prob=0.95
                        prob = max_s2g_prob
                        if prob_col is not None:
                            try:
                                prob = float(cols[prob_col])
                            except ValueError:
                                warn("Skipping unconvertible prob value %s" % (cols[prob_col]))
                                continue
                        if prob > max_s2g_prob:
                            prob = max_s2g_prob

                        if chrom in seen_chrom_pos and pos in seen_chrom_pos[chrom]:
                            if chrom not in chrom_pos_to_gene_prob:
                                chrom_pos_to_gene_prob[chrom] = {}
                            if pos not in chrom_pos_to_gene_prob[chrom]:
                                chrom_pos_to_gene_prob[chrom][pos] = []
                            chrom_pos_to_gene_prob[chrom][pos].append((gene, prob))


            added_chrom_pos = {}
            input_credible_set_info = {}
            if credible_sets_in is not None:

                log("Reading --credible-sets-in file %s" % credible_sets_in, INFO)

                #see if need to determine
                if credible_sets_pos_col is None or credible_sets_chrom_col is None:
                    (_, _, possible_credible_sets_chrom_cols, possible_credible_sets_pos_cols, _, _, _, _, _, _, header) = self._determine_columns(credible_sets_in)

                    if credible_sets_pos_col is None:
                        if len(possible_credible_sets_pos_cols) == 1:
                            credible_sets_pos_col = possible_credible_sets_pos_cols[0]
                            log("Using %s for position column; change with --credible-sets-pos-col if incorrect" % credible_sets_pos_col)
                        else:
                            bail("Could not determine position column; specify with --credible-sets-pos-col")
                    if credible_sets_chrom_col is None:
                        if len(possible_credible_sets_chrom_cols) == 1:
                            credible_sets_chrom_col = possible_credible_sets_chrom_cols[0]
                            log("Using %s for chromition column; change with --credible-sets-chrom-col if incorrect" % credible_sets_chrom_col)
                        else:
                            bail("Could not determine chrom column; specify with --credible-sets-chrom-col")

                with open_gz(credible_sets_in) as credible_sets_fh:
                    header_cols = credible_sets_fh.readline().strip().split()
                    chrom_col = self._get_col(credible_sets_chrom_col, header_cols)
                    pos_col = self._get_col(credible_sets_pos_col, header_cols)
                    id_col = self._get_col(credible_sets_id_col, header_cols)
                    ppa_col = None
                    if credible_sets_ppa_col is not None:
                        ppa_col = self._get_col(credible_sets_ppa_col, header_cols)

                    for line in credible_sets_fh:

                        cols = line.strip().split()
                        if id_col > len(cols) or chrom_col > len(cols) or pos_col > len(cols) or (ppa_col is not None and ppa_col > len(cols)):
                            warn("Skipping due to too few columns in line: %s" % line)
                            continue

                        cs_id = cols[id_col]

                        chrom = cols[chrom_col]

                        if hold_out_chrom is not None and chrom == hold_out_chrom:
                            continue

                        try:
                            pos = int(cols[pos_col])
                        except ValueError:
                            warn("Skipping unconvertible pos value %s" % (cols[pos_col]))
                            continue

                        ppa = 1
                        if ppa_col is not None:
                            try:
                                ppa = float(cols[ppa_col])
                            except ValueError:
                                warn("Skipping unconvertible ppa value %s" % (cols[ppa_col]))
                                continue

                        if chrom in seen_chrom_pos:
                            if pos not in seen_chrom_pos[chrom]:
                                #make up a beta
                                (p, beta, se, freq) = (var_p_threshold, 1, None, None)
                                chrom_pos_p_beta_se_freq[chrom].append((pos, p, beta, se, freq))
                                seen_chrom_pos[chrom].add(pos)
                                if chrom not in added_chrom_pos:
                                    added_chrom_pos[chrom] = set()
                                added_chrom_pos[chrom].add(pos)

                            if chrom not in input_credible_set_info:
                                input_credible_set_info[chrom] = {}
                            if cs_id not in input_credible_set_info[chrom]:
                                input_credible_set_info[chrom][cs_id] = []
                            input_credible_set_info[chrom][cs_id].append((pos, ppa))

            if total_num_vars == 0:
                bail("Didn't read in any variants!")

            gene_output_data = {}
            total_prob_causal = 0

            #run through twice
            #first, learn the window function
            closest_dist_Y = np.array([])
            closest_dist_X = np.array([])
            window_fun_intercept = None
            window_fun_slope = None

            var_all_p = np.array([])

            #store the gene probabilities for each signal
            gene_bf_data = []
            gene_prob_rows = []
            gene_prob_genes = []
            gene_prob_cols = []
            gene_prob_col_num = 0
            gene_covariate_genes = []
            self.huge_signals = []
            self.huge_signal_posteriors = []
            self.huge_signal_posteriors_for_regression = []
            self.huge_signal_sum_gene_cond_probabilities = []
            self.huge_signal_mean_gene_pos = []
            self.huge_gene_covariates = None
            self.huge_gene_covariates_mask = None
            self.huge_gene_covariate_names = None
            self.huge_gene_covariate_intercept_index = None

            #second, compute the huge scores
            for learn_params in [True, False]:
                index_var_chrom_pos_ps = {}
                if learn_params:
                    log("Learning window function and allelic var scale factor")
                else:
                    log("Calculating GWAS HuGE scores")

                for chrom in chrom_pos_p_beta_se_freq:

                    #convert all of these to np arrays sorted by chromosome
                    #sorted arrays of variant positions and p-values

                    chrom_pos_p_beta_se_freq[chrom].sort(key=lambda k: k[0])
                    vars_zipped = list(zip(*chrom_pos_p_beta_se_freq[chrom]))

                    if len(vars_zipped) == 0:
                        continue

                    var_pos = np.array(vars_zipped[0], dtype=float)
                    var_p = np.array(vars_zipped[1], dtype=float)
                    var_beta = np.array(vars_zipped[2], dtype=float)
                    var_se = np.array(vars_zipped[3], dtype=float)

                    (var_p, var_beta, var_se) = self._complete_p_beta_se(var_p, var_beta, var_se)


                    var_z = var_beta / var_se
                    var_se2 = np.square(var_se)

                    #this will vary slightly by chromosome but probably okay
                    mean_n = np.mean(1 / var_se2)

                    #sorted arrays of gene positions and p-values
                    if chrom not in gene_chrom_name_pos:
                        warn("Could not find chromosome %s in --gene-loc-file; skipping for now" % chrom)
                        continue

                    index_var_chrom_pos_ps[chrom] = []

                    gene_chrom_name_pos[chrom].sort(key=lambda k: k[1])
                    gene_zipped = list(zip(*gene_chrom_name_pos[chrom]))

                    #gene_names is array of the unique gene names
                    #gene_index_to_name_index is an array of the positions (each gene has multiple) and tells us which gene name corresponds to each position
                    gene_names_non_unique = np.array(gene_zipped[0])

                    gene_names, gene_index_to_name_index = np.unique(gene_names_non_unique, return_inverse=True)
                    gene_name_to_index = self._construct_map_to_ind(gene_names)
                    gene_pos = np.array(gene_zipped[1])

                    #get a map from position to gene
                    pos_to_gene_prob = None
                    if chrom_pos_to_gene_prob is not None and chrom in chrom_pos_to_gene_prob:
                        pos_to_gene_prob = chrom_pos_to_gene_prob[chrom]                        

                    #gene_prob_causal = np.full(len(gene_names), self.background_prior)

                    exon_interval_tree = None
                    interval_to_gene = None
                    if exons_loc_file is not None and chrom in chrom_to_interval_tree:
                        exon_interval_tree = chrom_to_interval_tree[chrom]
                        interval_to_gene = chrom_interval_to_gene[chrom]

                    def __get_closest_gene_indices(region_pos):
                        gene_indices = np.searchsorted(gene_pos, region_pos)
                        gene_indices[gene_indices == len(gene_pos)] -= 1

                        #look to the left and the right to see which gene closer
                        lower_mask = np.abs(region_pos - gene_pos[gene_indices - 1]) < np.abs(region_pos - gene_pos[gene_indices])
                        gene_indices[lower_mask] = gene_indices[lower_mask] - 1
                        return gene_indices

                    def __get_gene_posterior(region_pos, full_prob, window_fun_slope, window_fun_intercept, exon_interval_tree=None, interval_to_gene=None, pos_to_gene_prob=None, max_offset=20, cap=True, do_print=True):

                        #TODO: read in file of coding variants and set those to 95% for the closest gene, rather than using the gaussian below
                        closest_gene_indices = __get_closest_gene_indices(region_pos)

                        var_offset_prob = np.zeros((max_offset * 2 + 1, len(region_pos)))
                        var_gene_index = np.full((max_offset * 2 + 1, len(region_pos)), -1)

                        offsets = np.arange(-max_offset,max_offset+1)
                        var_offset_prob = np.zeros((len(offsets), len(region_pos)))
                        var_gene_index = np.full(var_offset_prob.shape, -1)
                        cur_gene_indices = np.add.outer(offsets, closest_gene_indices)
                        cur_gene_indices[cur_gene_indices >= len(gene_pos)] = len(gene_pos) - 1
                        cur_gene_indices[cur_gene_indices <= 0] = 0

                        prob_causal_odds = np.exp(window_fun_slope * np.abs(gene_pos[cur_gene_indices] - region_pos) + window_fun_intercept)
                        #get the 


                        cur_prob_causal = full_prob * (prob_causal_odds / (1 + prob_causal_odds))
                        cur_prob_causal[cur_prob_causal < 0] = 0

                        #take only the maximum value across all genes, since each gene can have multiple indices, 
                        #the following code generates a mask of all of the spots that contain the maximum value per group
                        #to do so though it has to sort the arrays
                        groups = gene_index_to_name_index[cur_gene_indices]
                        data = copy.copy(cur_prob_causal)
                        order = np.lexsort((data, groups), axis=0)

                        order2 = np.arange(groups.shape[1])
                        groups2 = groups[order, order2]
                        data2 = data[order, order2]
                        max_by_group_mask = np.empty(groups2.shape, 'bool')
                        max_by_group_mask[-1,:] = True
                        max_by_group_mask[:-1,:] = groups2[1:,:] != groups2[:-1,:]

                        #now "unsort" the mask
                        rev_order = np.empty_like(order)
                        rev_order[order, order2] = np.repeat(np.arange(order.shape[0]), order.shape[1]).reshape(order.shape[0], order.shape[1])
                        rev_max_by_group_mask = max_by_group_mask[rev_order, order2]

                        #need to keep only the maximum probability for each gene for each variant (in case some genes appear multiple times)
                        #zero out the values that are not max by group
                        cur_prob_causal[~rev_max_by_group_mask] = 0

                        var_offset_prob = cur_prob_causal
                        var_gene_index = gene_index_to_name_index[cur_gene_indices]


                        def __add_var_rows(_var_inds, _gene_prob_lists, _var_offset_prob, _var_gene_index):
                            #var_inds: indices into var_gene_index and var_offset_probs
                            #_gene_prob: list of list of (gene, prob) pairs; outer list same length as var_inds
                            var_to_seen_genes = {}
                            num_added = 0
                            for i in range(len(_var_inds)):
                                cur_var_index = _var_inds[i]
                                if cur_var_index not in var_to_seen_genes:
                                    var_to_seen_genes[cur_var_index] = set()
                                for cur_gene,cur_prob in _gene_prob_lists[i]:
                                    if cur_gene in gene_name_to_index:
                                        cur_gene_index = gene_name_to_index[cur_gene]
                                        if cur_gene_index not in var_to_seen_genes[cur_var_index]:
                                            var_to_seen_genes[cur_var_index].add(cur_gene_index)
                                            if num_added < len(var_to_seen_genes[cur_var_index]):
                                                _var_offset_prob = np.vstack((_var_offset_prob, np.zeros((1, _var_offset_prob.shape[1]))))
                                                _var_gene_index = np.vstack((_var_gene_index, np.zeros((1, _var_gene_index.shape[1]))))
                                                num_added += 1
                                            #first need to set anything else with this index to be 0
                                            _var_offset_prob[_var_gene_index[:,cur_var_index] == cur_gene_index, cur_var_index] = 0
                                            #then scale everything non-zero down to account for likelihood that the variant is actually coding
                                            _var_offset_prob[_var_gene_index[:,cur_var_index] == cur_gene_index, cur_var_index] *= (1 - cur_prob)
                                            #this is where to write exon probability
                                            row_index = _var_offset_prob.shape[0] - (num_added - len(var_to_seen_genes[cur_var_index])) - 1
                                            _var_offset_prob[row_index,cur_var_index] = full_prob[cur_var_index] * cur_prob
                                            _var_gene_index[row_index,cur_var_index] = cur_gene_index

                            return((_var_offset_prob, _var_gene_index))
                                            


                        if exon_interval_tree is not None and interval_to_gene is not None:
                            #now add in a row for the exons
                            #this is the list of region_pos that overlap an exon
                            (region_with_overlap_inds, overlapping_interval_starts, overlapping_interval_stops) = exon_interval_tree.find(region_pos, region_pos)
                            coding_var_linkage_prob = np.maximum(np.exp(window_fun_slope + window_fun_intercept)/(1+np.exp(window_fun_slope + window_fun_intercept)), 0.95)

                            if True:
                                #this needs to have the gene, prob corresponding to each position
                                gene_lists = [interval_to_gene[(overlapping_interval_starts[i], overlapping_interval_stops[i])] for i in range(len(region_with_overlap_inds))]
                                gene_prob_lists = []
                                for i in range(len(gene_lists)):
                                    gene_prob_lists.append(list(zip(gene_lists[i], [coding_var_linkage_prob for j in range(len(gene_lists[i]))])))

                                var_offset_prob, var_gene_index = __add_var_rows(region_with_overlap_inds, gene_prob_lists, var_offset_prob, var_gene_index)
                            else:
                                #TODO: DELETE THIS
                                #THIS IS OLD CODE IN CASE ABOVE DOESN'T WORK

                                #append a column to the var_offset_prob and var_gene_index corresponding to the exon
                                #may need to append more than one column if a variant is in exons of more than one gene
                                var_to_seen_genes = {}
                                num_added = 0
                                for i in range(len(region_with_overlap_inds)):
                                    cur_var_index = region_with_overlap_inds[i]
                                    if cur_var_index not in var_to_seen_genes:
                                        var_to_seen_genes[cur_var_index] = set()
                                    cur_genes = interval_to_gene[(overlapping_interval_starts[i], overlapping_interval_stops[i])]
                                    for cur_gene in cur_genes:
                                        if cur_gene in gene_name_to_index:

                                            cur_gene_index = gene_name_to_index[cur_gene]
                                            if cur_gene_index not in var_to_seen_genes[cur_var_index]:
                                                var_to_seen_genes[cur_var_index].add(cur_gene_index)
                                                if num_added < len(var_to_seen_genes[cur_var_index]):
                                                    var_offset_prob = np.vstack((var_offset_prob, np.zeros((1, var_offset_prob.shape[1]))))
                                                    var_gene_index = np.vstack((var_gene_index, np.zeros((1, var_gene_index.shape[1]))))
                                                    num_added += 1
                                                #first need to set anything else with this index to be 0
                                                var_offset_prob[var_gene_index[:,cur_var_index] == cur_gene_index, cur_var_index] = 0
                                                #then scale everything non-zero down to account for likelihood that the variant is actually coding
                                                var_offset_prob[var_gene_index[:,cur_var_index] == cur_gene_index, cur_var_index] *= (1 - coding_var_linkage_prob)
                                                #this is where to write exon probability
                                                row_index = var_offset_prob.shape[0] - (num_added - len(var_to_seen_genes[cur_var_index])) - 1
                                                var_offset_prob[row_index,cur_var_index] = full_prob[cur_var_index] * coding_var_linkage_prob
                                                var_gene_index[row_index,cur_var_index] = cur_gene_index


                        if pos_to_gene_prob is not None:
                            gene_prob_lists = []
                            for i in range(len(region_pos)):
                                probs = []
                                if region_pos[i] in pos_to_gene_prob:
                                    probs = pos_to_gene_prob[region_pos[i]]
                                gene_prob_lists.append(probs)
                            var_offset_prob, var_gene_index = __add_var_rows(range(len(region_pos)), gene_prob_lists, var_offset_prob, var_gene_index)

                        var_gene_index = var_gene_index.astype(int)

                        #first normalize not accounting for any dependencies across genes

                        #scale_raw_closest_gene: set everything to have the closest gene as closest gene prob
                        #cap_raw_closest_gene: set everything to have probability no greater than closest gene prob

                        if scale_raw_closest_gene or cap_raw_closest_gene:
                            var_offset_prob_max = var_offset_prob.max(axis=0)
                            var_offset_norm = np.ones(full_prob.shape)
                            var_offset_norm[var_offset_prob_max != 0] = full_prob[var_offset_prob_max != 0] * closest_gene_prob / var_offset_prob_max[var_offset_prob_max != 0]

                            if cap_raw_closest_gene:
                                cap_mask = var_offset_norm > 1
                                var_offset_norm[cap_mask] = 1
                        else:
                            var_offset_norm = 1

                        var_offset_prob *= var_offset_norm

                        def ___aggregate_var_gene_index(cur_var_offset_prob):
                            
                            cur_gene_indices, idx = np.unique(var_gene_index.ravel(), return_inverse=True)
                            cur_gene_prob_causal = np.bincount(idx, weights=cur_var_offset_prob.ravel())

                            #remove the very low ones
                            non_zero_mask = cur_gene_prob_causal > 0.001 * np.max(cur_gene_prob_causal)

                            cur_gene_prob_causal = cur_gene_prob_causal[non_zero_mask]
                            cur_gene_indices = cur_gene_indices[non_zero_mask]

                            #cap very high ones

                            cur_gene_po = None
                            if cap:
                                cur_gene_prob_causal[cur_gene_prob_causal > 0.999] = 0.999
                                cur_gene_po = cur_gene_prob_causal / (1 - cur_gene_prob_causal)

                            return (cur_gene_prob_causal, cur_gene_indices, cur_gene_po)

                        (cur_gene_prob_causal_no_norm, cur_gene_indices_no_norm, cur_gene_po_no_norm) = ___aggregate_var_gene_index(var_offset_prob)

                        #now do it normalized
                        var_offset_prob_sum = np.sum(var_offset_prob, axis=0)
                        var_offset_prob_sum[var_offset_prob_sum < 1] = 1
                        var_offset_prob_norm = var_offset_prob / var_offset_prob_sum
                        (cur_gene_prob_causal_norm, cur_gene_indices_norm, cur_gene_po_norm) = ___aggregate_var_gene_index(var_offset_prob_norm)



                        return (cur_gene_prob_causal_no_norm, cur_gene_indices_no_norm, cur_gene_po_no_norm, cur_gene_prob_causal_norm, cur_gene_indices_norm)

                    if learn_params:

                        #randomly sample 100K variants
                        region_vars = np.full(len(var_pos), False)
                        number_needed = 100000

                        region_vars[np.random.random(len(region_vars)) < (float(number_needed) / total_num_vars)] = True

                        closest_gene_indices = __get_closest_gene_indices(var_pos[region_vars])

                        closest_dists = np.abs(gene_pos[closest_gene_indices] - var_pos[region_vars])
                        closest_dists = closest_dists[closest_dists <= max_closest_gene_dist]

                        closest_dist_X = np.append(closest_dist_X, closest_dists)
                        closest_dist_Y = np.append(closest_dist_Y, np.full(len(closest_dists), closest_gene_prob))

                        var_all_p = np.append(var_all_p, var_p)

                        max_offset = 200

                        #new, vectorized
                        offsets = np.arange(-max_offset,max_offset+1)
                        cur_gene_indices = np.add.outer(offsets, closest_gene_indices)
                        cur_gene_indices[cur_gene_indices >= len(gene_pos)] = len(gene_pos) - 1
                        cur_gene_indices[cur_gene_indices <= 0] = 0

                        #ignore any that are actually the closest gene
                        cur_mask = (gene_names[gene_index_to_name_index[cur_gene_indices]] != gene_names[gene_index_to_name_index[closest_gene_indices]])
                        #remove everything above the maximum
                        non_closest_dists = np.abs(gene_pos[cur_gene_indices] - var_pos[region_vars])

                        cur_mask = np.logical_and(cur_mask, non_closest_dists <= max_closest_gene_dist)

                        #maximum is used here to avoid divide by 0
                        #any zeros will get subsetted out by cur_mask anyway
                        non_closest_probs = (np.full(non_closest_dists.shape, (1 - closest_gene_prob) / np.maximum(np.sum(cur_mask, axis=0), 1)))[cur_mask]
                        non_closest_dists = non_closest_dists[cur_mask]
                            
                        closest_dist_X = np.append(closest_dist_X, non_closest_dists)

                        closest_dist_Y = np.append(closest_dist_Y, np.full(len(non_closest_dists), non_closest_probs))

                    else:

                        #first 
                        max_gene_offset = 500

                        gene_offsets = np.arange(max_gene_offset+1)

                        gene_start_indices = np.zeros(len(gene_names), dtype=int)
                        gene_end_indices = np.zeros(len(gene_names), dtype=int)
                        gene_num_indices = np.zeros(len(gene_names), dtype=int)

                        gene_name_to_ind = self._construct_map_to_ind(gene_names)
                        for i in range(len(gene_names_non_unique)):
                            gene_name_ind = gene_name_to_ind[gene_names_non_unique[i]]
                            if gene_start_indices[gene_name_ind] == 0:
                                gene_start_indices[gene_name_ind] = i
                            gene_end_indices[gene_name_ind] = i
                            gene_num_indices[gene_name_ind] += 1

                        #these store the indices of genes to the left and right
                        genes_higher_indices = np.add.outer(gene_offsets, gene_end_indices).astype(int)
                        genes_ignore_indices = np.full(genes_higher_indices.shape, False)
                        genes_ignore_indices[genes_higher_indices >= len(gene_pos)] = True
                        genes_higher_indices[genes_higher_indices >= len(gene_pos)] = len(gene_pos) - 1
                        genes_lower_indices = np.add.outer(-gene_offsets, gene_start_indices).astype(int)
                        genes_ignore_indices[genes_lower_indices <= 0] = True
                        genes_lower_indices[genes_lower_indices <= 0] = 0

                        #ignore any that are actually the gene itself

                        higher_ignore_mask = np.logical_or(genes_ignore_indices, (gene_names[gene_index_to_name_index[genes_higher_indices]] == gene_names[gene_index_to_name_index[gene_end_indices]]))
                        lower_ignore_mask = np.logical_or(genes_ignore_indices, (gene_names[gene_index_to_name_index[genes_lower_indices]] == gene_names[gene_index_to_name_index[gene_start_indices]]))

                        right_dists = (gene_pos[genes_higher_indices] - gene_pos[gene_end_indices]).astype(float)

                        right_dists[higher_ignore_mask] = np.inf
                        right_dists[right_dists == 0] = 1

                        left_dists = (gene_pos[gene_start_indices] - gene_pos[genes_lower_indices]).astype(float)
                        left_dists[lower_ignore_mask] = np.inf
                        left_dists[left_dists == 0] = 1

                        # distance to next closest gene (left and right)

                        right_dist = np.min(right_dists, axis=0)
                        left_dist = np.min(left_dists, axis=0)

                        # sum of 1/distance (or logit distance) to 5 or 10 nearest genes (left and right)

                        right_sum = np.sum(1.0 / right_dists, axis=0)
                        left_sum = np.sum(1.0 / left_dists, axis=0)
                        right_left_sum = right_sum + left_sum

                        # number of genes within 1 Mb or 10 Mb (left and right)
                        large_dist = 250000
                        small_dist = 50000

                        num_right_small = np.sum(right_dists < small_dist, axis=0)
                        num_left_small = np.sum(left_dists < small_dist, axis=0)

                        num_right_large = np.sum(right_dists < large_dist, axis=0)
                        num_left_large = np.sum(left_dists < large_dist, axis=0)

                        num_small = num_right_small + num_left_small
                        num_large = num_right_large + num_left_large

                        # expanse of the gene
                        gene_size = gene_pos[gene_end_indices] - gene_pos[gene_start_indices]

                        # number of locations
                        #gene_num_indices

                        #sum of linkqge probabilities
                        chrom_start = np.max((np.min(gene_pos) - 1e6, 0))
                        chrom_end = np.max(gene_pos) + 1e6
                        #space them evenly, with spacing equal to average distance between SNPs in a 10e6 SNP GWAS
                        sim_variant_positions = np.linspace(chrom_start, chrom_end, int((chrom_end - chrom_start) / (3e9/10e5)), dtype=int)

                        (sim_gene_prob_causal_orig, sim_gene_indices, sim_gene_po, sim_gene_prob_causal_norm_orig, sim_gene_indices_norm) = __get_gene_posterior(sim_variant_positions, np.ones(len(sim_variant_positions)), window_fun_slope, window_fun_intercept, max_offset=20, cap=False, do_print=False)

                        #have to map these over to the original indices in case the sim_gene_prob_causal_orig was missing some genes
                        sim_gene_prob_causal = np.zeros(len(gene_names))
                        for i in range(len(sim_gene_indices)):
                            sim_gene_prob_causal[sim_gene_indices[i]] = sim_gene_prob_causal_orig[i]
                        sim_gene_prob_causal_norm = np.zeros(len(gene_names))
                        for i in range(len(sim_gene_indices_norm)):
                            sim_gene_prob_causal_norm[sim_gene_indices_norm[i]] = sim_gene_prob_causal_norm_orig[i]

                        cur_huge_gene_covariates = np.vstack((right_left_sum, num_right_large, num_left_large, gene_num_indices, sim_gene_prob_causal, np.ones(len(gene_names)))).T

                        #OLD ONES
                        #cur_huge_gene_covariates = np.vstack((right_dist, left_dist, right_left_sum, num_right_large, num_left_large, gene_num_indices, sim_gene_prob_causal, sim_gene_prob_causal_norm, np.ones(len(gene_names)))).T
                        #cur_huge_gene_covariates = np.vstack((right_dist, left_dist, right_left_sum, num_right_large, num_left_large, sim_gene_prob_causal, sim_gene_prob_causal_norm, np.ones(len(gene_names)))).T
                        #cur_huge_gene_covariates = np.vstack((right_dist, left_dist, right_left_sum, num_right_large, num_left_large, gene_size, gene_num_indices, sim_gene_prob_causal, sim_gene_prob_causal_norm, np.ones(len(gene_names)))).T
                        #cur_huge_gene_covariates = np.vstack((sim_gene_prob_causal, np.ones(len(gene_names)))).T
                        #cur_huge_gene_covariates = np.vstack((right_dist, left_dist, right_left_sum, num_small, num_large, gene_size, gene_num_indices, np.ones(len(gene_names)))).T
                        #cur_huge_gene_covariates = np.vstack((np.maximum(right_dist, left_dist), right_left_sum, np.minimum(num_right_small, num_left_small), np.minimum(num_right_large, num_left_large), gene_size, gene_num_indices, np.ones(len(gene_names)))).T

                        if self.huge_gene_covariates is None:

                            self.huge_gene_covariates = cur_huge_gene_covariates
                            self.huge_gene_covariate_names = ["right_left_sum_inv", "num_right_%s" % large_dist, "num_left_%s" % large_dist, "gene_num_indices", "sim_prob_causal", "intercept"]

                            #OLD ONES
                            #self.huge_gene_covariate_names = ["right_dist", "left_dist", "right_left_sum_inv", "num_right_%s" % large_dist, "num_left_%s" % large_dist, "sim_prob_causal", "sim_prob_causal_norm", "intercept"]
                            #self.huge_gene_covariate_names = ["right_dist", "left_dist", "right_left_sum_inv", "num_right_%s" % large_dist, "num_left_%s" % large_dist, "gene_size", "gene_num_indices", "sim_prob_causal", "sim_prob_causal_norm", "intercept"]
                            #self.huge_gene_covariate_names = ["sim_prob_causal", "intercept"]
                            #self.huge_gene_covariate_names = ["max_dist", "right_left_sum_inv", "min_num_%s" % small_dist, "min_num_%s" % large_dist, "gene_size", "gene_num_indices", "intercept"]

                            self.huge_gene_covariate_intercept_index = len(self.huge_gene_covariate_names) - 1

                        else:
                            self.huge_gene_covariates = np.vstack((self.huge_gene_covariates, cur_huge_gene_covariates))

                        gene_covariate_genes += list(gene_names)

                    #now onto variants

                   
                    #Z-score based one:
                    #K=-0.439
                    #np.sqrt(1 + K) * np.exp(-np.square(var_z) / 2 * (K) / (1 + K))
                    #or, for which sample size doesn't matter:
                    #K=-0.439 / np.mean(var_n)
                    #np.sqrt(1 + var_n * K) * np.exp(-np.square(var_z) / 2 * (var_n * K) / (1 + var_n * K))

                    #print("CHANGING ALLELIC VAR")
                    #var_log_bf = np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(var_z) * allelic_var_k / (1 + allelic_var_k)
                    var_log_bf = -np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(var_z) * allelic_var_k / (1 + allelic_var_k)
                    var_log_bf_detect = -np.log(np.sqrt(1 + allelic_var_k_detect)) + 0.5 * np.square(var_z) * allelic_var_k_detect / (1 + allelic_var_k_detect)

                    #now calculate the posteriors
                    var_posterior = var_log_bf + np.log(gwas_prior_odds)
                    var_posterior_detect = var_log_bf_detect + np.log(gwas_prior_odds_detect)

                    max_log = 15
                    for cur_var_posterior in [var_posterior, var_posterior_detect]:
                        max_mask = cur_var_posterior < max_log
                        cur_var_posterior[~max_mask] = 1
                        cur_var_posterior[max_mask] = np.exp(cur_var_posterior[max_mask])
                        cur_var_posterior[max_mask] = cur_var_posterior[max_mask] / (1 + cur_var_posterior[max_mask])

                    variants_keep = np.full(len(var_pos), True)
                    qc_fail = 1 / var_se2 < min_n_ratio * mean_n
                    variants_keep[qc_fail] = False

                    #make sure to add in additional credible set ids
                    if not learn_params and chrom in added_chrom_pos:
                        for cur_pos in added_chrom_pos[chrom]:
                            variants_keep[var_pos == cur_pos] = True

                    #filter down for efficiency
                    var_pos = var_pos[variants_keep]
                    var_p = var_p[variants_keep]
                    var_beta = var_beta[variants_keep]
                    var_se = var_se[variants_keep]
                    var_se2 = var_se2[variants_keep]
                    var_log_bf = var_log_bf[variants_keep]
                    var_posterior = var_posterior[variants_keep]
                    var_posterior_detect = var_posterior_detect[variants_keep]

                    var_logp = -np.log(var_p) / np.log(10)

                    var_freq = None
                    if freq_col is not None:
                        var_freq = np.array(vars_zipped[4], dtype=float)[variants_keep]
                        var_freq[var_freq > 0.5] = 1 - var_freq[var_freq > 0.5]

                    variants_left = np.full(len(var_pos), True)
                    cs_ignore = np.full(len(var_pos), False)
                    while np.sum(variants_left) > 0:

                        cond_prob = None
                        if not learn_params and chrom in input_credible_set_info and len(input_credible_set_info[chrom].keys()) > 0:

                            cur_cs_id = list(input_credible_set_info[chrom].keys())[0]
                            cur_cs_vars = input_credible_set_info[chrom][cur_cs_id]

                            region_vars = np.full(len(var_pos), False)
                            cond_prob = np.zeros(len(var_pos))
                            for pos_ppa in cur_cs_vars:
                                pos = pos_ppa[0]
                                ppa = pos_ppa[1]
                                mask = var_pos == pos
                                if np.sum(mask) > 0:
                                    region_vars[mask] = True
                                    cond_prob[mask] = ppa

                            if np.sum(cond_prob) > 0:
                                cond_prob /= np.sum(cond_prob)
                            i = np.argmax(cond_prob)
                            cond_prob = cond_prob[region_vars]
                            cs_ignore = np.logical_or(cs_ignore, np.logical_and(var_pos > np.min(var_pos[mask]) - credible_set_span, var_pos < np.max(var_pos[mask]) + credible_set_span))
                            del input_credible_set_info[chrom][cur_cs_id]
                        else:
                            if not learn_params:
                                variants_left = np.logical_and(variants_left, ~cs_ignore)

                            #get the lowest p-value remaining variant
                            variants_left_inds = np.where(variants_left)[0]
                            i = variants_left_inds[np.argmin(var_p[variants_left_inds])]

                            #get all variants in the region
                            #TODO: here is where we would ideally clump either by
                            #1. reading in LD (if we have it)

                            #find variants within 100kb, extending as needed if we have some above 
                            #log("Processing variant %s:%d" % (chrom, var_pos[i]), TRACE)

                            region_vars = np.logical_and(var_pos >= var_pos[i] - signal_window_size, var_pos <= var_pos[i] + signal_window_size)

                            region_inds = np.where(region_vars)[0]
                            assert(len(region_inds) > 0)

                            #extend the region until the distance to the last significant snp is greater than the signal_min_sep

                            increase_ratio = 1.3
                            self._record_param("p_value_increase_ratio_for_sep_signal", increase_ratio)

                            region_ind = region_inds[0] - 1
                            last_significant_snp = region_inds[0]
                            while region_ind > 0 and np.abs(var_pos[region_ind] - var_pos[last_significant_snp]) < signal_min_sep:
                                if var_p[region_ind] < max_signal_p:

                                    if var_p[region_ind] < var_p[last_significant_snp]:
                                        #check if it starts to increase after it
                                        cur_block = np.logical_and(np.logical_and(var_pos >= var_pos[region_ind], var_pos < var_pos[region_ind] + signal_min_sep), var_p < max_signal_p)

                                        prev_block = np.logical_and(np.logical_and(var_pos >= var_pos[region_ind] + signal_min_sep, var_pos < var_pos[region_ind] + 2 * signal_min_sep), var_p < max_signal_p)

                                        #if the mean p-value of significant SNPs decreases relative to the previous one, stop extending
                                        if np.sum(prev_block) == 0 or np.sum(cur_block) == 0 or np.mean(var_logp[cur_block]) > increase_ratio * np.mean(var_logp[prev_block]):
                                            break

                                    last_significant_snp = region_ind

                                    region_vars[region_ind:region_inds[0]] = True

                                region_ind -= 1

                            region_ind = region_inds[-1] + 1
                            last_significant_snp = region_inds[0]
                            while region_ind < len(var_pos) and np.abs(var_pos[region_ind] - var_pos[last_significant_snp]) < signal_min_sep:
                                if var_p[region_ind] < max_signal_p:
                                    #if this is increasing, check to see if we can break
                                    if var_p[region_ind] < var_p[last_significant_snp]:
                                        cur_block = np.logical_and(np.logical_and(var_pos <= var_pos[region_ind], var_pos > var_pos[region_ind] - signal_min_sep), var_p < max_signal_p)
                                        prev_block = np.logical_and(np.logical_and(var_pos <= var_pos[region_ind] - signal_min_sep, var_pos > var_pos[region_ind] - 2 * signal_min_sep), var_p < max_signal_p)

                                        #if the mean p-value of significant SNPs decreases relative to the previous one, stop extending
                                        if np.sum(prev_block) == 0 or np.sum(cur_block) == 0 or np.mean(var_logp[cur_block]) > increase_ratio * np.mean(var_logp[prev_block]):
                                            break

                                    last_significant_snp = region_ind

                                    region_vars[region_inds[-1]:region_ind] = True
                                region_ind += 1

                            #if we have MAF, approximate LD by MAF
                            if var_freq is not None:

                                #maximum LD occurs when genotype carriers completely overlap (one is subset of another)
                                #(E[XY] - E[X]E[Y]) / sqrt((E[X^2] - E[X]^2)(E[Y^2] - E[Y]^2))
                                #(MAF_MIN - MAF_MAX * MAF_MIN) / sqrt(MAF_MAX * (1 - MAF_MAX) * MAF_MIN * (1 - MAF_MIN)
                                #MAF_MIN * (1 - MAF_MAX) / sqrt(MAF_MAX * (1 - MAF_MAX) * MAF_MIN * (1 - MAF_MIN)
                                #sqrt(MAF_MIN * (1 - MAF_MAX)) / sqrt((1 - MAF_MAX) * MAF_MIN)

                                max_ld = np.sqrt((var_freq[i] * (1 - var_freq)) / (var_freq * (1 - var_freq[i])))
                                max_ld[var_freq[i] > var_freq] = 1.0 / max_ld[var_freq[i] > var_freq]

                                int_mask = np.logical_and(region_vars, max_ld < max_clump_ld)
                                if np.sum(int_mask) > 0:
                                    argminp = np.argmin(var_p[int_mask])

                                #for variants with frequencies that imply they cannot have LD above max_clump_ld with index var,
                                #remove them from this clump
                                region_vars[max_ld < max_clump_ld] = False

                            if signal_max_logp_ratio is not None:
                                region_vars[var_logp/var_logp[i] < signal_max_logp_ratio] = False

                        #now remove all of the variants that have been seen
                        left_mask = variants_left[region_vars]
                        region_vars = np.logical_and(region_vars, variants_left)
                        #set these to not be seen again
                        variants_left[region_vars] = False

                        index_var_chrom_pos_ps[chrom].append((var_pos[i], var_p[i]))

                        sig_posterior = var_posterior[i]
                        sig_posterior_detect = var_posterior_detect[i]

                        min_pos = np.min(var_pos[region_vars])
                        max_pos = np.max(var_pos[region_vars])
                        #print("%d-%d (%d)" % (min_pos, max_pos, max_pos - min_pos))
                        #log("Index SNP %d=%d; region=%d-%d; logp=%.3g-%.3g" % (i,var_pos[i], np.min(var_pos[region_vars]), np.max(var_pos[region_vars]), np.min(var_logp[region_vars]), np.max(var_logp[region_vars])), TRACE)
                        #print("Variant:",var_pos[i],"P:",var_p[i],"POST:",sig_posterior,"MIN_POS:",min_pos,"MAX_POS:",max_pos,"NUM:",np.sum(region_vars))
                        #m = np.where(var_pos == 84279410.0)[0]
                        #print(region_vars[m],var_pos[m])


                        if cond_prob is None:
                            #now find the conditional posteriors of all of the variants in the region
                            #use log sum exp trick

                            c = np.max(var_log_bf[region_vars])

                            log_sum_bf = c + np.log(np.sum(np.exp(var_log_bf[region_vars] - c)))
                            log_rel_bf = var_log_bf[region_vars] - log_sum_bf

                            cond_prob = log_rel_bf
                            cond_prob[cond_prob > max_log] = 1
                            cond_prob[cond_prob < max_log] = np.exp(cond_prob[cond_prob < max_log])


                        #this is the final posterior probability of association for all variants in the region
                        full_prob = cond_prob * sig_posterior
                        full_prob_detect = cond_prob * sig_posterior_detect

                        if not learn_params:
                            
                            #calculate the posteriors
                            
                            #find out how many indices to look to the left and right of the nearest one before assigning 0 linkage probability

                            #first get all of the gene indices within the max window of each variant on each size
                            gene_index_ranges = __get_closest_gene_indices(np.vstack((var_pos[region_vars], var_pos[region_vars] - max_closest_gene_dist, var_pos[region_vars] + max_closest_gene_dist)))

                            max_num_indices = np.maximum(np.max(gene_index_ranges[0,:] - gene_index_ranges[1,:]), np.max(gene_index_ranges[2,:] - gene_index_ranges[0,:]))

                            (cur_gene_prob_causal, cur_gene_indices, cur_gene_po, cur_gene_prob_causal_norm, cur_gene_indices_norm) = __get_gene_posterior(var_pos[region_vars], full_prob, window_fun_slope, window_fun_intercept, exon_interval_tree=exon_interval_tree, interval_to_gene=interval_to_gene, pos_to_gene_prob=pos_to_gene_prob, max_offset=max_num_indices)


                            #print("TEMP TEMP")
                            #if "UBE2E2" in gene_names[cur_gene_indices]:
                            #    print(gene_names[cur_gene_indices])
                            #    print(cur_gene_prob_causal)
                            #    for i in range(len(cur_gene_indices)):
                            #        if gene_names[cur_gene_indices[i]] == "UBE2E2":
                            #            print(i,gene_names[cur_gene_indices[i]],cur_gene_prob_causal[i], cur_gene_po[i] / self.background_bf)


                            gene_prob_rows += list(len(gene_prob_genes) + cur_gene_indices)
                            gene_prob_cols += ([gene_prob_col_num] * len(cur_gene_indices))

                            gene_bf_data += list(cur_gene_po  / self.background_bf)

                            #the total posterior (for use in scale)


                            self.huge_signals.append((chrom, var_pos[i], var_p[i]))
                            self.huge_signal_posteriors.append(sig_posterior)
                            self.huge_signal_posteriors_for_regression.append(sig_posterior_detect)

                            #store the marginal bayes factor
                            cur_gene_cond_prob_causal = cur_gene_prob_causal / sig_posterior
                            #the sum of the conditional probabilities (after taking out sig posterior)
                            sum_cond_prob = np.sum(cur_gene_cond_prob_causal)

                            self.huge_signal_sum_gene_cond_probabilities.append(sum_cond_prob if sum_cond_prob < 1 else 1)

                            #the mean of the conditional BFs
                            mean_cond_po = np.sum(cur_gene_cond_prob_causal / (1 - cur_gene_cond_prob_causal))
                            self.huge_signal_mean_gene_pos.append(mean_cond_po)
                            gene_prob_col_num += 1

                            #now record them
                            #for i in range(len(gene_pos)):
                            #   gene_index = gene_index_to_name_index[i]
                            #    gene_name = gene_names[gene_index]
                            #    if gene_name not in gene_output_data:
                            #        gene_output_data[gene_name] = gene_prob_causal[gene_index]
                            #        total_prob_causal += gene_prob_causal[gene_index]
                            #    else:
                            #        #sanity check: same gene name should have same probability
                            #        assert(gene_prob_causal[gene_index] == gene_output_data[gene_name])

                            gene_prob_genes += list(gene_names)

                if learn_params:

                    #first update units if needed
                    unit_scale_factor = None
                    if gwas_units is not None:
                        unit_scale_factor = np.square(gwas_units)

                    index_var_ps = []
                    for chrom in index_var_chrom_pos_ps:
                        cur_pos = np.array(list(zip(*index_var_chrom_pos_ps[chrom]))[0])
                        cur_ps = np.array(list(zip(*index_var_chrom_pos_ps[chrom]))[1])
                        #now filter the variants
                        indep_window = 1e6
                        tree = IntervalTree([(x - indep_window, x + indep_window) for x in cur_pos])
                        start_to_index = dict([(cur_pos[i] - indep_window, i) for i in range(len(cur_pos))])
                        (ind_with_overlap_inds, overlapping_interval_starts, overlapping_interval_stops) = tree.find(cur_pos, cur_pos)
                        #ind_with_overlap_inds is the indices that had an overlap
                        #overlapping_interval_starts is start position of the overlapping interval; we can map these to indices by adding window size
                        assert(np.isclose(overlapping_interval_stops - overlapping_interval_starts - 2 * indep_window, np.zeros(len(overlapping_interval_stops))).all())

                        overlapping_inds = [start_to_index[i] for i in overlapping_interval_starts]

                        var_p = cur_ps[ind_with_overlap_inds]
                        overlap_var_p = cur_ps[overlapping_inds]
                        #this is a mask of the indices that are nearby another stronger variant
                        var_not_best_mask = overlap_var_p < var_p
                        #this is the list of indices

                        indep_mask = np.full(len(cur_pos), True)
                        indep_mask[ind_with_overlap_inds[var_not_best_mask]] = False

                        index_var_ps += list(cur_ps[indep_mask])

                    index_var_ps.sort()

                    index_var_ps = np.array(index_var_ps)
                    num_below_low_p = np.sum(index_var_ps < gwas_low_p)

                    self._record_param("num_below_initial_low_p", num_below_low_p)

                    log(" (%d variants below p=%.4g)" % (num_below_low_p, gwas_low_p))

                    if detect_high_power is not None or detect_low_power is not None:
                        target_max_num_variants = detect_high_power
                        target_min_num_variants = detect_low_power

                        old_low_p = gwas_low_p
                        high_or_low = None
                        if target_max_num_variants is not None and num_below_low_p > target_max_num_variants:
                            gwas_low_p = index_var_ps[target_max_num_variants]
                            high_or_low = "high"
                        if target_min_num_variants is not None and num_below_low_p < target_min_num_variants:
                            if len(index_var_ps) > target_min_num_variants:
                                gwas_low_p = index_var_ps[target_min_num_variants]
                            elif len(index_var_ps) > 0:
                                gwas_low_p = np.min(index_var_ps)
                            else:
                                gwas_low_p = 0.05
                            high_or_low = "low"

                        if high_or_low is not None:
                            self._record_param("gwas_low_p", gwas_low_p)

                            log("Detected %s power (%d variants below p=%.4g); adjusting --gwas-low-p to %.4g" % (high_or_low, num_below_low_p, old_low_p, gwas_low_p))
                            (allelic_var_k_detect, gwas_prior_odds_detect) = self.compute_allelic_var_and_prior(gwas_high_p, gwas_high_p_posterior, gwas_low_p, gwas_low_p_posterior)

                            if detect_adjust_huge:
                                #we have to adjust both for regression and the values used for huge scores
                                (allelic_var_k, gwas_prior_odds) = (allelic_var_k_detect, gwas_prior_odds_detect)
                                log("Using k=%.3g, po=%.3g for regression and huge scores" % (allelic_var_k_detect, gwas_prior_odds_detect))
                                self._record_params({"gwas_allelic_var_k": allelic_var_k, "gwas_prior_odds": gwas_prior_odds})
                            else:
                                log("Using k=%.3g, po=%.3g for regression only" % (allelic_var_k_detect, gwas_prior_odds_detect))
                                self._record_params({"gwas_allelic_var_k_detect": allelic_var_k_detect, "gwas_prior_odds_detect": gwas_prior_odds_detect})

                    log("Using k=%.3g, po=%.3g" % (allelic_var_k, gwas_prior_odds))
                    self._record_params({"gwas_allelic_var_k": allelic_var_k, "gwas_prior_odds": gwas_prior_odds})
                           
                    if learn_window:

                        use_logistic_window_function = False
                        if use_logistic_window_function:

                            #run this a few times
                            num_samples = 5
                            window_fun_slope = 0
                            window_fun_intercept = 0

                            for i in range(num_samples):
                                sample = np.random.random(len(closest_dist_Y)) < closest_dist_Y
                                closest_dist_Y_sample = copy.copy(closest_dist_Y)
                                closest_dist_Y_sample[sample > closest_dist_Y] = 1
                                closest_dist_Y_sample[sample <= closest_dist_Y] = 0

                                (cur_window_fun_slope, se, z, p, se_inflation_factor, cur_window_fun_intercept, diverged) = self._compute_logistic_beta_tildes(closest_dist_X[:,np.newaxis], closest_dist_Y_sample, 1, 0, resid_correlation_matrix=None, convert_to_dichotomous=False, log_fun=lambda x, y=0: 1)
                                window_fun_slope += cur_window_fun_slope
                                window_fun_intercept += cur_window_fun_intercept

                            window_fun_slope /= num_samples
                            window_fun_intercept /= num_samples
                        else:

                            mean_closest_dist_X = np.mean(closest_dist_X[closest_dist_Y == closest_gene_prob])
                            mean_non_closest_dist_X = np.mean(closest_dist_X[closest_dist_Y != closest_gene_prob])
                            mean_non_closest_dist_Y = np.mean(closest_dist_Y[closest_dist_Y != closest_gene_prob])
                            window_fun_slope  = (np.log(closest_gene_prob / (1 - closest_gene_prob)) - np.log(mean_non_closest_dist_Y / (1 - mean_non_closest_dist_Y))) / (mean_closest_dist_X - mean_non_closest_dist_X)
                            window_fun_intercept = np.log(closest_gene_prob / (1 - closest_gene_prob)) - window_fun_slope * mean_closest_dist_X

                        if window_fun_slope >= 0:
                            warn("Could not fit decaying linear window function slope for max-closest-gene-dist=%.4g and closest-gene_prob=%.4g; using default" % (max_closest_gene_dist, closest_gene_prob))
                            window_fun_slope = -6.983e-06
                            window_fun_intercept = -1.934

                        log("Fit function %.4g * x + %.4g for closest gene probability" % (window_fun_slope, window_fun_intercept))

                    else:
                        if max_closest_gene_dist < 3e5:
                            window_fun_slope = -5.086e-05
                            window_fun_intercept = 2.988
                        else:
                            window_fun_slope = -5.152e-05
                            window_fun_intercept = 4.854
                        log("Using %.4g * x + %.4g for closest gene probability" % (window_fun_slope, window_fun_intercept))

                    self._record_params({"window_fun_slope": window_fun_slope, "window_fun_intercept": window_fun_intercept})

            #now iterate through all significant variants

            log("Done reading --gwas-in", DEBUG)

            exomes_positive_controls_prior_log_bf = None

            if self.genes is not None:
                genes = self.genes
                gene_to_ind = self.gene_to_ind
            else:
                genes = list(gene_to_chrom.keys())
                gene_to_ind = self._construct_map_to_ind(genes)

            #need to remap the indices
            extra_genes = []
            extra_gene_to_ind = {}
            for i in range(len(gene_prob_rows)):
                cur_gene = gene_prob_genes[gene_prob_rows[i]]
                    
                if cur_gene in gene_to_ind:
                    new_ind = gene_to_ind[cur_gene]
                elif cur_gene in extra_gene_to_ind:
                    new_ind = extra_gene_to_ind[cur_gene]
                else:
                    new_ind = len(extra_genes) + len(genes)
                    extra_genes.append(cur_gene)
                    extra_gene_to_ind[cur_gene] = new_ind
                gene_prob_rows[i] = new_ind

            #add in any genes that were missed
            for cur_gene in list(gene_to_chrom.keys()) + gene_prob_genes:
                if cur_gene not in gene_to_ind and cur_gene not in extra_gene_to_ind:
                    new_ind = len(extra_genes) + len(genes)
                    extra_genes.append(cur_gene)
                    extra_gene_to_ind[cur_gene] = new_ind

            gene_prob_gene_list = genes + extra_genes

            #sort the covariate file; initially populate it with mean value in case some genes are missing from it
            
            sorted_huge_gene_covariates = np.tile(np.nanmean(self.huge_gene_covariates, axis=0), len(gene_prob_gene_list)).reshape((len(gene_prob_gene_list), self.huge_gene_covariates.shape[1]))

            for i in range(len(gene_covariate_genes)):
                cur_gene = gene_covariate_genes[i]
                assert(cur_gene in gene_to_ind or cur_gene in extra_gene_to_ind)

                if cur_gene in gene_to_ind:
                    new_ind = gene_to_ind[cur_gene]
                elif cur_gene in extra_gene_to_ind:
                    new_ind = extra_gene_to_ind[cur_gene]
                noninf_mask = ~np.isnan(self.huge_gene_covariates[i,:])
                sorted_huge_gene_covariates[new_ind,noninf_mask] = self.huge_gene_covariates[i,noninf_mask]

            self.huge_gene_covariates = sorted_huge_gene_covariates


            if self.Y_exomes is not None:
                assert(len(genes) == len(self.Y_exomes))
                exomes_positive_controls_prior_log_bf = np.append(self.Y_exomes, np.zeros(len(extra_genes)))
            if self.Y_positive_controls is not None:
                assert(len(genes) == len(self.Y_positive_controls))
                positive_controls_prior_log_bf = np.append(self.Y_positive_controls, np.zeros(len(extra_genes)))
                if exomes_positive_controls_prior_log_bf is None:
                    exomes_positive_controls_prior_log_bf = positive_controls_prior_log_bf
                else:
                    exomes_positive_controls_prior_log_bf += positive_controls_prior_log_bf
                    

            #add in the extra genes

            #this is the normalizing constant between huge_signal_bfs and the BFs
            #PO = BF * huge_signal_posteriors
            self.huge_signal_posteriors = np.array(self.huge_signal_posteriors)
            self.huge_signal_posteriors_for_regression = np.array(self.huge_signal_posteriors_for_regression)
            self.huge_signal_max_closest_gene_prob = max_closest_gene_prob
            self.huge_cap_region_posterior = cap_region_posterior
            self.huge_scale_region_posterior = scale_region_posterior
            self.huge_phantom_region_posterior = phantom_region_posterior
            self.huge_allow_evidence_of_absence = allow_evidence_of_absence
            self.huge_correct_huge = correct_huge


            #from Maller et al, these are proportional to BFs (but not necessarily equal)
            self.huge_signal_bfs = sparse.csc_matrix((gene_bf_data, (gene_prob_rows, gene_prob_cols)), shape=(len(gene_prob_gene_list), gene_prob_col_num))
            self.huge_signal_sum_gene_cond_probabilities = np.array(self.huge_signal_sum_gene_cond_probabilities)
            self.huge_signal_mean_gene_pos = np.array(self.huge_signal_mean_gene_pos)


            #construct the matrix

            if self.huge_correct_huge:
                #remove outliers on any of the metrics. We don't want to use them to fit the linear model
                huge_covariates_means = np.mean(self.huge_gene_covariates, axis=0)
                huge_covariates_sds = np.std(self.huge_gene_covariates, axis=0)
                #make sure intercept not excluded
                huge_covariates_sds[huge_covariates_sds == 0] = 1

                self.huge_gene_covariates_mask = np.all(self.huge_gene_covariates < huge_covariates_means + 5 * huge_covariates_sds, axis=1)

                self.huge_gene_covariates_mat_inv = np.linalg.inv(self.huge_gene_covariates[self.huge_gene_covariates_mask,:].T.dot(self.huge_gene_covariates[self.huge_gene_covariates_mask,:]))
                huge_gene_covariate_sds = np.std(self.huge_gene_covariates, axis=0)
                huge_gene_covariate_sds[huge_gene_covariate_sds == 0] = 1
                self.huge_gene_covariate_zs = (self.huge_gene_covariates - np.mean(self.huge_gene_covariates, axis=0)) / huge_gene_covariate_sds
            else:
                self.huge_gene_covariates = None
                self.huge_gene_covariates_mask = None
                self.huge_gene_covariate_names = None
                self.huge_gene_covariate_intercept_index = None
                self.huge_gene_covariate_betas = None
                self.huge_gene_covariates_mat_inv = None
                self.huge_gene_covariate_zs = None


            #print("DELETE THE IND MAP!!!")
            #gene_to_ind = self._construct_map_to_ind(gene_prob_genes)
            #print("JASON 1")
            #print(self.huge_signal_bfs[gene_to_ind["ZAN"],:])

            (huge_results_for_regression, huge_results_uncorrected_for_regression, absent_genes_for_regression, absent_log_bf_for_regression) = self._distill_huge_signal_bfs(self.huge_signal_bfs, self.huge_signal_posteriors_for_regression, self.huge_signal_sum_gene_cond_probabilities, self.huge_signal_mean_gene_pos, self.huge_signal_max_closest_gene_prob, self.huge_cap_region_posterior, self.huge_scale_region_posterior, self.huge_phantom_region_posterior, self.huge_allow_evidence_of_absence, self.huge_correct_huge, self.huge_gene_covariates, self.huge_gene_covariates_mask, self.huge_gene_covariates_mat_inv, gene_prob_gene_list, total_genes=self.genes, rel_prior_log_bf=exomes_positive_controls_prior_log_bf)

            (huge_results, huge_results_uncorrected, absent_genes, absent_log_bf) = self._distill_huge_signal_bfs(self.huge_signal_bfs, self.huge_signal_posteriors, self.huge_signal_sum_gene_cond_probabilities, self.huge_signal_mean_gene_pos, self.huge_signal_max_closest_gene_prob, self.huge_cap_region_posterior, self.huge_scale_region_posterior, self.huge_phantom_region_posterior, self.huge_allow_evidence_of_absence, self.huge_correct_huge, self.huge_gene_covariates, self.huge_gene_covariates_mask, self.huge_gene_covariates_mat_inv, gene_prob_gene_list, total_genes=self.genes, rel_prior_log_bf=exomes_positive_controls_prior_log_bf)

            #print("CALL 2")
            #temp_priors = np.zeros(self.huge_signal_bfs.shape[0])
            #temp_priors[gene_to_ind['ADAMTSL3']] = 5
            #(huge_results, huge_results_uncorrected, absent_genes, absent_log_bf) = self._distill_huge_signal_bfs(self.huge_signal_bfs, self.huge_signal_posteriors, self.huge_signal_sum_gene_cond_probabilities, self.huge_signal_mean_gene_pos, self.huge_signal_max_closest_gene_prob, self.cap_region_posterior, self.scale_region_posterior, self.phantom_region_posterior, self.allow_evidence_of_absence, self.huge_correct_huge, self.huge_gene_covariates, self.huge_gene_covariates_mask, self.huge_gene_covariates_mat_inv, gene_prob_gene_list, rel_prior_log_bf=temp_priors)

            #calculate
            #
            #for i in range(huge_signal_bfs.shape[0]):
            #    if huge_results[i] > 0:
            #        print(gene_prob_gene_list[i], huge_results[i], gene_output_data[gene_prob_gene_list[i]])
            
            #for gene in gene_output_data:
            #    gene_prob = gene_output_data[gene] * norm_constant
            #    gene_output_data[gene] = np.log(gene_prob / (1 - gene_prob)) - self.background_log_bf

            if self.genes is not None:
                gene_bf = np.array([np.nan] * len(self.genes))
                gene_bf_for_regression = np.array([np.nan] * len(self.genes))
            else:
                gene_bf = np.array([])
                gene_bf_for_regression = np.array([])

            extra_gene_bf = []
            extra_gene_bf_for_regression = []
            extra_genes = []
            self.gene_to_gwas_huge_score = {}
            self.gene_to_gwas_huge_score_uncorrected = {}

            for i in range(len(gene_prob_gene_list)):
                gene = gene_prob_gene_list[i]
                bf = huge_results[i]
                bf_for_regression = huge_results_for_regression[i]
                bf_uncorrected = huge_results_uncorrected[i]
                self.gene_to_gwas_huge_score[gene] = bf
                self.gene_to_gwas_huge_score_uncorrected[gene] = bf_uncorrected
                if self.genes is not None and gene in self.gene_to_ind:
                    #if self.gene_to_ind[gene] != i:
                        #print(gene,self.gene_to_ind[gene],i)
                    assert(self.gene_to_ind[gene] == i)
                    gene_bf[self.gene_to_ind[gene]] = bf
                    gene_bf_for_regression[self.gene_to_ind[gene]] = bf_for_regression
                else:
                    extra_gene_bf.append(bf)
                    extra_gene_bf_for_regression.append(bf_for_regression)
                    extra_genes.append(gene)
            for gene in absent_genes:
                bf = absent_log_bf
                self.gene_to_gwas_huge_score[gene] = bf
                self.gene_to_gwas_huge_score_uncorrected[gene] = bf
                if self.genes is not None and gene in self.gene_to_ind:
                    gene_bf[self.gene_to_ind[gene]] = bf
                    gene_bf_for_regression[self.gene_to_ind[gene]] = bf
                else:
                    extra_gene_bf.append(bf)
                    extra_gene_bf_for_regression.append(bf)
                    extra_genes.append(gene)
            extra_gene_bf = np.array(extra_gene_bf)

            self.combine_huge_scores()

            return (gene_bf, extra_genes, extra_gene_bf, gene_bf_for_regression, extra_gene_bf_for_regression)
                
    def calculate_huge_scores_exomes(self, exomes_in, exomes_gene_col=None, exomes_p_col=None, exomes_beta_col=None, exomes_se_col=None, exomes_n_col=None, exomes_n=None, exomes_units=None, allelic_var=0.36, exomes_low_p=2.5e-6, exomes_high_p=0.05, exomes_low_p_posterior=0.95, exomes_high_p_posterior=0.10, hold_out_chrom=None, gene_loc_file=None, **kwargs):
        if exomes_in is None:
            bail("Require --exomes-in for this operation")

        if hold_out_chrom is not None and self.gene_to_chrom is None:
            print("READING IT BITCH")
            (self.gene_chrom_name_pos, self.gene_to_chrom, self.gene_to_pos) = self._read_loc_file(gene_loc_file)

        self._record_params({"exomes_low_p": exomes_low_p, "exomes_high_p": exomes_high_p, "exomes_low_p_posterior": exomes_low_p_posterior, "exomes_high_p_posterior": exomes_high_p_posterior})

        if exomes_gene_col is None:
            need_columns = True

        has_se = exomes_se_col is not None or exomes_n_col is not None or exomes_n is not None
        if exomes_gene_col is not None and ((exomes_p_col is not None and exomes_beta_col is not None) or (exomes_p_col is not None and has_se) or (exomes_beta_col is not None and has_se)):
            need_columns = False
        else:
            need_columns = True

        if need_columns:
            (possible_gene_id_cols, possible_var_id_cols, possible_chrom_cols, possible_pos_cols, possible_locus_cols, possible_p_cols, possible_beta_cols, possible_se_cols, possible_freq_cols, possible_n_cols, header) = self._determine_columns(exomes_in)

            #now recompute
            if exomes_gene_col is None:
                if len(possible_gene_id_cols) == 1:
                    exomes_gene_col = possible_gene_id_cols[0]
                    log("Using %s for gene_id column; change with --exomes-gene-col if incorrect" % exomes_gene_col)
                else:
                    bail("Could not determine gene_id column from header %s; specify with --exomes-gene-col" % header)

            if exomes_p_col is None:
                if len(possible_p_cols) == 1:
                    exomes_p_col = possible_p_cols[0]
                    log("Using %s for p column; change with --exomes-p-col if incorrect" % exomes_p_col)
                else:
                    log("Could not determine p column from header %s; if desired specify with --exomes-p-col" % header)
            if exomes_se_col is None:
                if len(possible_se_cols) == 1:
                    exomes_se_col = possible_se_cols[0]
                    log("Using %s for se column; change with --exomes-se-col if incorrect" % exomes_se_col)
                else:
                    log("Could not determine se column from header %s; if desired specify with --exomes-se-col" % header)
            if exomes_beta_col is None:
                if len(possible_beta_cols) == 1:
                    exomes_beta_col = possible_beta_cols[0]
                    log("Using %s for beta column; change with --exomes-beta-col if incorrect" % exomes_beta_col)
                else:
                    log("Could not determine beta column from header %s; if desired specify with --exomes-beta-col" % header)

            if exomes_n_col is None:
                if len(possible_n_cols) == 1:
                    exomes_n_col = possible_n_cols[0]
                    log("Using %s for N column; change with --exomes-n-col if incorrect" % exomes_n_col)
                else:
                    log("Could not determine N column from header %s; if desired specify with --exomes-n-col" % header)

            has_se = exomes_se_col is not None or exomes_n_col is not None or exomes_n is not None
            if (exomes_p_col is not None and exomes_beta_col is not None) or (exomes_p_col is not None and has_se) or (exomes_beta_col is not None and has_se):
                pass
            else:
                bail("Require information about at least two of p-value, se, and beta; specify with --exomes-p-col, --exomes-beta-col, and --exomes-se-col")

        (allelic_var_k, exomes_prior_odds) = self.compute_allelic_var_and_prior(exomes_high_p, exomes_high_p_posterior, exomes_low_p, exomes_low_p_posterior)

        self._record_params({"exomes_allelic_var_k": allelic_var_k, "exomes_prior_odds": exomes_prior_odds})

        log("Using exomes k=%.3g, po=%.3g" % (allelic_var_k, exomes_prior_odds))

        log("Calculating exomes HuGE scores")

        log("Reading --exomes-in file %s" % exomes_in, INFO)

        seen_genes = set()
        genes = []
        gene_ps = []
        gene_betas = []
        gene_ses = []

        with open_gz(exomes_in) as exomes_fh:
            header_cols = exomes_fh.readline().strip().split()
            gene_col = self._get_col(exomes_gene_col, header_cols)

            p_col = None
            if exomes_p_col is not None:
                p_col = self._get_col(exomes_p_col, header_cols)

            beta_col = None
            if exomes_beta_col is not None:
                beta_col = self._get_col(exomes_beta_col, header_cols)

            n_col = None
            se_col = None
            if exomes_n_col is not None:
                n_col = self._get_col(exomes_n_col, header_cols)
            if exomes_se_col is not None:
                se_col = self._get_col(exomes_se_col, header_cols)
            
            chrom_pos_p_se = {}

            #read in the exomes associations
            total_num_genes = 0

            for line in exomes_fh:

                cols = line.strip().split()
                if gene_col > len(cols) or (p_col is not None and p_col > len(cols)) or (se_col is not None and se_col > len(cols)) or (beta_col is not None and beta_col > len(cols)) or (n_col is not None and n_col > len(cols)):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[gene_col]

                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]

                if hold_out_chrom is not None and gene in self.gene_to_chrom and self.gene_to_chrom[gene] == hold_out_chrom:
                    continue

                p = None
                beta = None
                se = None
                
                if p_col is not None:
                    try:
                        p = float(cols[p_col])
                    except ValueError:
                        if not cols[p_col] == "NA":
                            warn("Skipping unconvertible p value %s" % (cols[p_col]))
                        continue

                    min_p = 1e-250
                    if p < min_p:
                        p = min_p

                    if p <= 0 or p > 1:
                        warn("Skipping invalid p value %s" % (p))
                        continue

                if beta_col is not None:
                    try:
                        beta = float(cols[beta_col])
                    except ValueError:
                        if not cols[beta_col] == "NA":
                            warn("Skipping unconvertible beta value %s" % (cols[beta_col]))
                        continue

                if se_col is not None:
                    try:
                        se = float(cols[se_col])
                    except ValueError:
                        if not cols[se_col] == "NA":
                            warn("Skipping unconvertible se value %s" % (cols[se_col]))
                        continue
                elif n_col is not None:
                    try:
                        n = float(cols[n_col])
                    except ValueError:
                        if not cols[n_col] == "NA":
                            warn("Skipping unconvertible n value %s" % (cols[n_col]))
                        continue
                        
                    if n <= 0:
                        warn("Skipping invalid N value %s" % (n))
                        continue
                    se = 1 / np.sqrt(n)
                elif exomes_n is not None:
                    if exomes_n <= 0:
                        bail("Invalid exomes-n value: %s" % (exomesa_n))
                        continue
                    n = exomes_n
                    se = 1 / np.sqrt(n)

                total_num_genes += 1

                if gene in seen_genes:
                    warn("Gene %s has been seen before; skipping all but first occurrence" % gene)
                    continue
                
                seen_genes.add(gene)
                genes.append(gene)
                gene_ps.append(p)
                gene_betas.append(beta)
                gene_ses.append(se)

            #determine scale_factor
          
            gene_ps = np.array(gene_ps, dtype=float)
            gene_betas = np.array(gene_betas, dtype=float)
            gene_ses = np.array(gene_ses, dtype=float)

            (gene_ps, gene_betas, gene_ses) = self._complete_p_beta_se(gene_ps, gene_betas, gene_ses)

            gene_zs = gene_betas / gene_ses

            gene_ses2 = np.square(gene_ses)

            log("Done reading --exomes-in", DEBUG)

            #adjust units of beta if beta column was passed in
            #if exomes_units is not None:
            #    allelic_var *= np.square(exomes_units)
            #    log("Scaling allelic variance %.3g-fold to be %.4g" % (np.square(exomes_units), allelic_var))
            #else:
            #    #get the empirical variance of the betas for variants in a range of p=0.05
            #    p05mask = np.abs(np.abs(gene_betas/gene_ses) - 1.95) <= 0.07
            #    if np.sum(p05mask) > 100:
            #        emp_beta_var = np.mean(np.square(gene_betas[p05mask]) - gene_ses2[p05mask])
            #        #this is roughly what we observe for a dichotomous trait in this range. Larger than for gwas by about 10x
            #        ref_emp_beta_var = 0.1
            #        if emp_beta_var > 0 and (emp_beta_var / ref_emp_beta_var > 5 or emp_beta_var / ref_emp_beta_var < 0.2):
            #            allelic_var *= (emp_beta_var / 0.1)
            #            log("Scaling allelic variance %.3g-fold to be %.4g" % (emp_beta_var / 0.1, allelic_var))

            #print("CHANGING ALLELIC VAR")
            #gene_log_bfs = np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(gene_zs) * allelic_var_k / (1 + allelic_var_k)

            gene_log_bfs = -np.log(np.sqrt(1 + allelic_var_k)) + 0.5 * np.square(gene_zs) * allelic_var_k / (1 + allelic_var_k)

            max_log = 15
            gene_log_bfs[gene_log_bfs > max_log] = max_log

            #set lower bound not here but below; otherwise it gets inflated above background
            #gene_log_bfs[gene_log_bfs < 0] = 0

            gene_post = np.exp(gene_log_bfs + np.log(exomes_prior_odds))
            gene_probs = gene_post / (gene_post + 1)
            gene_probs[gene_probs < self.background_prior] = self.background_prior

            #gene_probs_sum = np.sum(gene_probs)

            absent_genes = set()
            if self.genes is not None:
                #have to account for these
                absent_genes = set(self.genes) - set(genes)
            #gene_probs_sum += self.background_prior * len(absent_genes)

            norm_constant = 1
            #norm_constant = (self.background_prior * (len(gene_probs) + len(absent_genes))) / gene_probs_sum
            #need at least 1000 genes
            #if len(gene_probs) < 1000:
            #    norm_constant = 1
            #gene_probs *= norm_constant


            gene_log_bfs = np.log(gene_probs / (1 - gene_probs)) - self.background_log_bf

            absent_prob = self.background_prior * norm_constant
            absent_log_bf = np.log(absent_prob / (1 - absent_prob)) - self.background_log_bf

            if self.genes is not None:
                gene_bf = np.array([np.nan] * len(self.genes))
            else:
                gene_bf = np.array([])

            extra_gene_bf = []
            extra_genes = []
            self.gene_to_exomes_huge_score = {}

            for i in range(len(genes)):
                gene = genes[i]
                bf = gene_log_bfs[i]
                self.gene_to_exomes_huge_score[gene] = bf
                if self.genes is not None and gene in self.gene_to_ind:
                    gene_bf[self.gene_to_ind[gene]] = bf
                else:
                    extra_gene_bf.append(bf)
                    extra_genes.append(gene)
            for gene in absent_genes:
                bf = absent_log_bf
                self.gene_to_exomes_huge_score[gene] = bf
                if self.genes is not None and gene in self.gene_to_ind:
                    gene_bf[self.gene_to_ind[gene]] = bf
                else:
                    extra_gene_bf.append(bf)
                    extra_genes.append(gene)
            extra_gene_bf = np.array(extra_gene_bf)

            self.combine_huge_scores()
            return (gene_bf, extra_genes, extra_gene_bf)

    def read_positive_controls(self, positive_controls_in, positive_controls_id_col=None, positive_controls_prob_col=None, hold_out_chrom=None, gene_loc_file=None, **kwargs):
        if positive_controls_in is None:
            bail("Require --positive-controls-in for this operation")

        if hold_out_chrom is not None and self.gene_to_chrom is None:
            (self.gene_chrom_name_pos, self.gene_to_chrom, self.gene_to_pos) = self._read_loc_file(gene_loc_file)

        log("Reading --positive-controls-in file %s" % positive_controls_in, INFO)
        self.gene_to_positive_controls = {}
        id_col = 0
        prob_col = None

        with open_gz(positive_controls_in) as positive_controls_fh:
            read_header = False
            for line in positive_controls_fh:
                cols = line.strip().split()
                if not read_header:
                    read_header = True
                    if len(cols) > 1 or positive_controls_prob_col is not None:
                        if positive_controls_id_col is None:
                            bail("--positive-controls-id-col required for positive control files with more than one gene")
                        else:
                            id_col = self._get_col(positive_controls_id_col, cols)
                        if positive_controls_prob_col is not None:
                            prob_col = self._get_col(positive_controls_prob_col, cols)
                        continue

                if id_col >= len(cols):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue
                
                gene = cols[id_col]
                
                if gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]

                if hold_out_chrom is not None and gene in self.gene_to_chrom and self.gene_to_chrom[gene] == hold_out_chrom:
                    continue

                prob = 0.99
                if prob_col is not None and prob_col >= len(cols):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                if prob_col is not None:
                    try:
                        prob = float(cols[prob_col])
                        if prob <= 0 or prob >= 1:
                            warn("Probabilities must be in (0,1); observed %s for %s" % (prob, gene))
                            continue
                    except ValueError:
                        if not cols[prob_col] == "NA":
                            warn("Skipping unconvertible prob value %s for gene %s" % (cols[prob_col], gene))
                        continue

                log_bf = np.log(prob / (1 - prob)) - self.background_log_bf
                self.gene_to_positive_controls[gene] = log_bf

        if self.genes is not None:
            genes = self.genes
            gene_to_ind = self.gene_to_ind
        else:
            genes = []
            gene_to_ind = {}

        positive_controls = np.array([np.nan] * len(genes))
        
        extra_positive_controls = []
        extra_genes = []
        for gene in self.gene_to_positive_controls:
            log_bf = self.gene_to_positive_controls[gene]
            if gene in gene_to_ind:
                positive_controls[gene_to_ind[gene]] = log_bf
            else:
                extra_positive_controls.append(log_bf)
                extra_genes.append(gene)

        return (positive_controls, extra_genes, np.array(extra_positive_controls))


    def compute_allelic_var_and_prior(self, high_p, high_p_posterior, low_p, low_p_posterior):

        if high_p < low_p:
            warn("Swapping high_p and low_p")
            temp = high_p
            high_p = low_p
            low_p = temp

        if high_p == low_p:
            high_p = low_p * 2

        if high_p_posterior >= 1:
            po_high = 0.99/0.01
        elif high_p_posterior <=0 :
            po_high = 0.001/0.999
        else:
            po_high = high_p_posterior / (1 - high_p_posterior)

        if low_p_posterior >= 1:
            po_low = 0.99/0.01
        elif low_p_posterior <=0 :
            po_low = 0.001/0.999
        else:
            po_low = low_p_posterior / (1 - low_p_posterior)

        z_high = np.abs(scipy.stats.norm.ppf(high_p/2))
        z_low = np.abs(scipy.stats.norm.ppf(low_p/2))
        ratio = po_low / po_high

        allelic_var_k = 2 * np.log(ratio) / (np.square(z_low) - np.square(z_high))

        if allelic_var_k > 1:
            #reset high_p_posterior
            max_allelic_var_k = 0.99;
            po_high = po_low / np.exp(max_allelic_var_k * (np.square(z_low) - np.square(z_high)) / 2)
            log("allelic_var_k overflow; adjusting --high-p-posterior to %.4g" % (po_high/(1+po_high)))
            ratio = po_low / po_high
            allelic_var_k = 2 * np.log(ratio) / (np.square(z_low) - np.square(z_high))

        allelic_var_k = allelic_var_k / (1 - allelic_var_k)

        #print("CHANGING ALLELIC VAR K")
        prior_odds = po_low / (np.sqrt(1 / (1 + allelic_var_k)) * np.exp(0.5 * np.square(z_low) * (allelic_var_k / (1 + allelic_var_k))))
        
        return (allelic_var_k, prior_odds)


    def combine_huge_scores(self):
        #combine the huge scores if needed
        if self.gene_to_gwas_huge_score is not None and self.gene_to_exomes_huge_score is not None:
            self.gene_to_huge_score = {}
            genes = list(set().union(self.gene_to_gwas_huge_score, self.gene_to_exomes_huge_score))
            for gene in genes:
                self.gene_to_huge_score[gene] = 0
                if gene in self.gene_to_gwas_huge_score:
                    self.gene_to_huge_score[gene] += self.gene_to_gwas_huge_score[gene]
                if gene in self.gene_to_exomes_huge_score:
                    self.gene_to_huge_score[gene] += self.gene_to_exomes_huge_score[gene]

    def read_gene_set_statistics(self, stats_in, stats_id_col=None, stats_exp_beta_tilde_col=None, stats_beta_tilde_col=None, stats_p_col=None, stats_se_col=None, stats_beta_col=None, stats_beta_uncorrected_col=None, ignore_negative_exp_beta=False, max_gene_set_p=None, min_gene_set_beta=None, min_gene_set_beta_uncorrected=None, return_only_ids=False):

        if stats_in is None:
            bail("Require --stats-in for this operation")

        log("Reading --stats-in file %s" % stats_in, INFO)
        subset_mask = None
        need_to_take_log = False

        read_ids = set()

        with open_gz(stats_in) as stats_fh:
            header_cols = stats_fh.readline().strip().split()
            id_col = self._get_col(stats_id_col, header_cols)
            beta_tilde_col = None

            if stats_beta_tilde_col is not None:
                beta_tilde_col = self._get_col(stats_beta_tilde_col, header_cols, False)
            if beta_tilde_col is not None:
                log("Using col %s for beta_tilde values" % stats_beta_tilde_col)
            elif stats_exp_beta_tilde_col is not None:
                beta_tilde_col = self._get_col(stats_exp_beta_tilde_col, header_cols)
                need_to_take_log = True
                if beta_tilde_col is not None:
                    log("Using %s for exp(beta_tilde) values" % stats_exp_beta_tilde_col)
                else:
                    bail("Could not find beta_tilde column %s or %s in header: %s" % (stats_beta_tilde_col, stats_exp_beta_tilde_col, "\t".join(header_cols)))

            p_col = None
            if stats_p_col is not None:
                p_col = self._get_col(stats_p_col, header_cols, False)            

            se_col = None
            if stats_se_col is not None:
                se_col = self._get_col(stats_se_col, header_cols, False)            

            beta_col = None
            if stats_beta_col is not None:
                beta_col = self._get_col(stats_beta_col, header_cols, True)
            else:
                beta_col = self._get_col("beta", header_cols, False)
                
            beta_uncorrected_col = None
            if stats_beta_uncorrected_col is not None:
                beta_uncorrected_col = self._get_col(stats_beta_uncorrected_col, header_cols, True)
            else:
                beta_uncorrected_col = self._get_col("beta_uncorrected", header_cols, False)

            if se_col is None and p_col is None and beta_tilde_col is None and beta_col is None and beta_uncorrected_col is None:
                bail("Require at least something to read from --gene-set-stats-in")

            if not return_only_ids:

                if self.gene_sets is not None:
                    if beta_tilde_col is not None:
                        self.beta_tildes = np.zeros(len(self.gene_sets))
                    if p_col is not None or se_col is not None:
                        self.p_values = np.zeros(len(self.gene_sets))
                        self.ses = np.zeros(len(self.gene_sets))
                        self.z_scores = np.zeros(len(self.gene_sets))
                    if beta_col is not None:
                        self.betas = np.zeros(len(self.gene_sets))
                    if beta_uncorrected_col is not None:
                        self.betas_uncorrected = np.zeros(len(self.gene_sets))

                    subset_mask = np.array([False] * len(self.gene_sets))
                else:
                    if beta_tilde_col is not None:
                        self.beta_tildes = []
                    if p_col is not None or se_col is not None:
                        self.p_values = []
                        self.ses = []
                        self.z_scores = []
                    if beta_col is not None:
                        self.betas = []
                    if beta_uncorrected_col is not None:
                        self.betas_uncorrected = []

            gene_sets = []
            gene_set_to_ind = {}

            ignored = 0

            for line in stats_fh:
                beta_tilde = None
                alpha_tilde = None
                p = None
                se = None
                z = None
                beta = None
                beta_uncorrected = None

                cols = line.strip().split()
                if id_col > len(cols) or (beta_tilde_col is not None and beta_tilde_col > len(cols)) or (p_col is not None and p_col > len(cols)) or (se_col is not None and se_col > len(cols)):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene_set = cols[id_col]
                if gene_set in gene_set_to_ind:
                    warn("Already seen gene set %s; only considering first instance" % (gene_set))

                if beta_tilde_col is not None:
                    try:
                        beta_tilde = float(cols[beta_tilde_col])
                    except ValueError:
                        if not cols[beta_tilde_col] == "NA":
                            warn("Skipping unconvertible beta_tilde value %s for gene_set %s" % (cols[beta_tilde_col], gene_set))
                        continue

                    if need_to_take_log:
                        if beta_tilde < 0:
                            if ignore_negative_exp_beta:
                                continue
                            bail("Exp(beta) value %s for gene set %s is < 0; did you mean to specify --stats-beta-col? Otherwise, specify --ignore-negative-exp-beta to ignore these" % (beta_tilde, gene_set))
                        beta_tilde = np.log(beta_tilde)
                    alpha_tilde = 0

                if se_col is not None:
                    try:
                        se = float(cols[se_col])
                    except ValueError:
                        if not cols[se_col] == "NA":
                            warn("Skipping unconvertible se value %s for gene_set %s" % (cols[se_col], gene_set))
                        continue

                    if beta_tilde_col is not None:
                        z = beta_tilde / se
                        p = 2*scipy.stats.norm.cdf(-np.abs(z))
                        if max_gene_set_p is not None and p > max_gene_set_p:
                            continue
                elif p_col is not None:
                    try:
                        p = float(cols[p_col])
                        if max_gene_set_p is not None and p > max_gene_set_p:
                            continue
                    except ValueError:
                        if not cols[p_col] == "NA":
                            warn("Skipping unconvertible p value %s for gene_set %s" % (cols[p_col], gene_set))
                        continue

                    z = np.abs(scipy.stats.norm.ppf(p/2))
                    if z == 0:
                        warn("Skipping gene_set %s due to 0 z-score" % (gene_set))
                        continue

                    if beta_tilde_col is not None:
                        se = np.abs(beta_tilde) / z

                if beta_col is not None:
                    try:
                        beta = float(cols[beta_col])
                        if min_gene_set_beta is not None and beta < min_gene_set_beta:
                            continue

                    except ValueError:
                        if not cols[beta_col] == "NA":
                            warn("Skipping unconvertible beta value %s for gene_set %s" % (cols[beta_col], gene_set))
                        continue

                if beta_uncorrected_col is not None:
                    try:
                        beta_uncorrected = float(cols[beta_uncorrected_col])
                        if min_gene_set_beta_uncorrected is not None and beta_uncorrected < min_gene_set_beta_uncorrected:
                            continue

                    except ValueError:
                        if not cols[beta_uncorrected_col] == "NA":
                            warn("Skipping unconvertible beta_uncorrected value %s for gene_set %s" % (cols[beta_uncorrected_col], gene_set))
                        continue


                gene_set_ind = None

                if self.gene_sets is not None:
                    if gene_set not in self.gene_set_to_ind:
                        ignored += 1
                        continue

                    if return_only_ids:
                        read_ids.add(gene_set)
                        continue

                    gene_set_ind = self.gene_set_to_ind[gene_set]
                    if gene_set_ind is not None:
                        if beta_tilde_col is not None:
                            self.beta_tildes[gene_set_ind] = beta_tilde * self.scale_factors[gene_set_ind]
                        if p_col is not None or se_col is not None:
                            self.p_values[gene_set_ind] = p
                            self.z_scores[gene_set_ind] = z
                            self.ses[gene_set_ind] = se * self.scale_factors[gene_set_ind]
                        if beta_col is not None:
                            self.betas[gene_set_ind] = beta
                        if beta_uncorrected_col is not None:
                            self.betas_uncorrected[gene_set_ind] = beta_uncorrected
                        subset_mask[gene_set_ind] = True
                else:
                    if return_only_ids:
                        read_ids.add(gene_set)
                        continue
                    

                    if beta_tilde_col is not None:
                        self.beta_tildes.append(beta_tilde)
                    if p_col is not None or se_col is not None:
                        self.p_values.append(p)
                        self.z_scores.append(z)
                        self.ses.append(se)
                    if beta_col is not None:
                        self.betas.append(beta)
                    if beta_uncorrected_col is not None:
                        self.betas_uncorrected.append(beta_uncorrected)

                    #store these in all cases to be able to check for duplicate gene sets in the input
                    gene_set_to_ind[gene_set] = len(gene_sets)
                    gene_sets.append(gene_set)

            log("Done reading --stats-in-file", DEBUG)

        if return_only_ids:
            return read_ids

        if self.gene_sets is not None:
            log("Subsetting matrices", DEBUG)
            #need to subset existing matrices
            if ignored > 0:
                warn("Ignored %s values from --stats-in file because absent from previously loaded files" % ignored)
            if sum(subset_mask) != len(subset_mask):
                warn("Excluding %s values from previously loaded files because absent from --stats-in file" % (len(subset_mask) - sum(subset_mask)))
                if not need_to_take_log and sum(self.beta_tildes < 0) == 0:
                    warn("All beta_tilde values are positive. Are you sure that the values in column %s are not exp(beta_tilde)?" % stats_beta_col)
                self._subset_gene_sets(subset_mask, keep_missing=True)
            log("Done subsetting matrices", DEBUG)
        else:
            self.X_orig_missing_gene_sets = None
            self.mean_shifts_missing = None
            self.scale_factors_missing = None
            self.is_dense_gene_set_missing = None
            self.ps_missing = None
            self.sigma2s_missing = None

            self.beta_tildes_missing = None
            self.p_values_missing = None
            self.ses_missing = None
            self.z_scores_missing = None

            self.beta_tildes = np.array(self.beta_tildes)
            self.p_values = np.array(self.p_values)
            self.z_scores = np.array(self.z_scores)
            self.ses = np.array(self.ses)
            self.gene_sets = gene_sets
            self.gene_set_to_ind = gene_set_to_ind

            if beta_col is not None:
                self.betas = np.array(self.betas)
            if beta_uncorrected_col is not None:
                self.betas_uncorrected = np.array(self.betas_uncorrected)

            self.total_qc_metrics_missing = None
            self.mean_qc_metrics_missing = None

        #self.max_gene_set_p = max_gene_set_p
        self.is_logistic = False
        #make sure we are doing the normalization
        self._set_X(self.X_orig, self.genes, self.gene_sets, skip_N=True)

    def calculate_gene_set_statistics(self, gwas_in=None, exomes_in=None, positive_controls_in=None, gene_bfs_in=None, gene_percentiles_in=None, gene_zs_in=None, Y=None, show_progress=True, max_gene_set_p=None, run_gls=False, run_logistic=False, run_corrected_ols=False, correct_betas=True, gene_loc_file=None, gene_cor_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, skip_V=False, **kwargs):
        if self.X_orig is None:
            bail("Error: X is required")
        #now calculate the betas and p-values

        log("Calculating gene set statistics", INFO)

        if Y is None:
            Y = self.Y_for_regression

        if self.Y_for_regression is None:
            if gwas_in is None and exomes_in is None and gene_bfs_in is None and gene_percentiles_in is None and gene_zs_in is None:
                bail("Need --gwas-in or --exomes-in or --gene-bfs-in or --gene-percentiles-in or --gene-zs-in")

            log("Reading Y within calculate_gene_set_statistics; parameters may not be honored")
            self.read_Y(gwas_in=gwas_in, exomes_in=exomes_in, positive_controls_in=positive_controls_in, gene_bfs_in=gene_bfs_in, gene_percentiles_in=gene_percentiles_in, gene_zs_in=gene_zs_in, **kwargs)


        #FIXME: need to make this so don't always read in correlations, and add priors where needed
        #can move this inside of the Y is None loop
        #but -- if compute correlation distance function is true and gene_cor_file is none and gene_loc file is not None, then we need to redo this
        #and: if gene_cor_file is not none, then we need to update the correlation matrix to account for the priors
        #To decrease correlation, we first convert cor to covaraince (multiply by np.var(Y)) then divide by np.var(Y) + np.var(prior). For np.sd prior, we can either use a fixed value (the sd of priors across all genes) or we can use the actual y values (and thus do np.sqrt(np.var(Y) + prior1)np.sqrt(np.var(Y) + prior2).
        #and, finally: we always need to call set Y here
        if run_gls:
            run_corrected_ols = False

        if (run_gls or run_corrected_ols) and self.y_corr is None:
            correlation_m = self._read_correlations(gene_cor_file, gene_loc_file, gene_cor_file_gene_col=gene_cor_file_gene_col, gene_cor_file_cor_start_col=gene_cor_file_cor_start_col)

            #convert X and Y to their new values
            min_correlation = 0.05
            self._set_Y(self.Y, self.Y_for_regression, self.Y_exomes, self.Y_positive_controls, Y_corr_m=correlation_m, store_cholesky=run_gls, store_corr_sparse=run_corrected_ols, skip_V=True, skip_scale_factors=True, min_correlation=min_correlation)

        #subset gene sets to remove empty ones first
        #number of gene sets in each gene set
        col_sums = self.get_col_sums(self.X_orig, num_nonzero=True)
        self._subset_gene_sets(col_sums > 0, keep_missing=False, skip_V=True, skip_scale_factors=True)

        #CAN REMOVE
        #mean of Y is now zero
        #self.beta_tildes = self.scale_factors * ((self.X_orig.T.dot(Y_clean) / len(Y_clean)) - (self.mean_shifts * np.mean(Y_clean))) / variances
        self._set_scale_factors()

        if run_logistic:

            Y_to_use = self.Y_for_regression

            Y = np.exp(Y_to_use + self.background_log_bf) / (1 + np.exp(Y_to_use + self.background_log_bf))

            #handy option in case we want to see what sampling looks like outside of gibbs
            use_sampling_for_betas = None
            if use_sampling_for_betas is not None and use_sampling_for_betas > 0:
                
                avg_beta_tildes = np.zeros(len(self.gene_sets))
                avg_z_scores = np.zeros(len(self.gene_sets))
                tot_its = 0
                for iteration_num in range(use_sampling_for_betas):
                    log("Sampling iteration %d..." % (iteration_num+1))
                    p_sample_m = np.zeros(Y.shape)
                    p_sample_m[np.random.random(Y.shape) < Y] = 1
                    Y = p_sample_m

                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors, alpha_tildes, diverged) = self._compute_logistic_beta_tildes(self.X_orig, Y, self.scale_factors, self.mean_shifts, resid_correlation_matrix=self.y_corr_sparse)

                    avg_beta_tildes += beta_tildes
                    avg_z_scores += z_scores
                    tot_its += 1
                    
                self.beta_tildes = avg_beta_tildes / tot_its
                self.z_scores = avg_z_scores / tot_its

                self.p_values = 2*scipy.stats.norm.cdf(-np.abs(self.z_scores))
                self.ses = np.full(self.beta_tildes.shape, 100.0)
                self.ses[self.z_scores != 0] = np.abs(self.beta_tildes[self.z_scores != 0] / self.z_scores[self.z_scores != 0])

            else:
                (self.beta_tildes, self.ses, self.z_scores, self.p_values, self.se_inflation_factors, alpha_tildes, diverged) = self._compute_logistic_beta_tildes(self.X_orig, Y, self.scale_factors, self.mean_shifts, resid_correlation_matrix=self.y_corr_sparse)

            self.is_logistic = True
        else:
            if run_gls:
                #Y has already been whitened
                #dot_product = np.array([])
                y_var = self.y_fw_var
                Y = self.Y_fw
                #OLD CODE
                #as an optimization, multiply original X by fully whitened Y, rather than half by half
                #for X_b, begin, end, batch in self._get_X_blocks():
                #    #calculate mean shifts
                #    dot_product = np.append(dot_product, X_b.T.dot(self.Y_w) / len(self.Y_w))
                #dot_product = self.X_orig.T.dot(self.Y_fw) / len(self.Y_fw)
            else:
                #Technically, we could use the above code for this case, since X_blocks will returned unwhitened matrix
                #But, probably faster to keep sparse multiplication? Might be worth revisiting later to see if there actually is a performance gain
                #We can use original X here because we know that whitening will occur only for GLS
                #assert this to be sure
                assert(not self.scale_is_for_whitened)
                Y = copy.copy(self.Y_for_regression)

                y_var = self.y_var

                #OLD CODE
                #dot_product = self.X_orig.T.dot(self.Y) / len(self.Y)

            #variances = np.power(self.scale_factors, 2)
            #multiply by scale factors because we store beta_tilde in units of scaled X
            #self.beta_tildes = self.scale_factors * np.array(dot_product) / variances
            #self.ses = self.scale_factors * np.sqrt(y_var) / (np.sqrt(variances * len(self.Y)))
            #self.z_scores = self.beta_tildes / self.ses
            #self.p_values = 2*scipy.stats.norm.cdf(-np.abs(self.z_scores))

            (self.beta_tildes, self.ses, self.z_scores, self.p_values, self.se_inflation_factors) = self._compute_beta_tildes(self.X_orig, Y, y_var, self.scale_factors, self.mean_shifts, resid_correlation_matrix=self.y_corr_sparse, )
            self.is_logistic = False

        if correct_betas:
            (self.beta_tildes, self.ses, self.z_scores, self.p_values, self.se_inflation_factors) = self._correct_beta_tildes(self.beta_tildes, self.ses, self.se_inflation_factors, self.total_qc_metrics, self.mean_qc_metrics, add_missing=True)

        self.X_orig_missing_gene_sets = None
        self.mean_shifts_missing = None
        self.scale_factors_missing = None
        self.is_dense_gene_set_missing = None
        self.ps_missing = None
        self.sigma2s_missing = None

        self.beta_tildes_missing = None
        self.p_values_missing = None
        self.ses_missing = None
        self.z_scores_missing = None

        self.total_qc_metrics_missing = None
        self.mean_qc_metrics_missing = None

        if max_gene_set_p is not None:
            gene_set_mask = self.p_values <= max_gene_set_p
            if np.sum(gene_set_mask) == 0 and len(self.p_values) > 0:
                gene_set_mask = self.p_values == np.min(self.p_values)
            log("Keeping %d gene sets that passed threshold of p<%.3g" % (np.sum(gene_set_mask), max_gene_set_p))
            self._subset_gene_sets(gene_set_mask, keep_missing=True, skip_V=True)

            if len(self.gene_sets) < 1:
                log("No gene sets left!")
                return

        #self.max_gene_set_p = max_gene_set_p

    #FIXME: Update calls to use includes_non_missing and from_osc
    def set_p(self, p):
        #if p is None:
        #    log("Set p called with p=%s" % p, TRACE)
        #else:
        #    log("Set p called with p=%.3g" % p, TRACE)

        if p is not None:
            if p > 1:
                p = 1
            if p < 0:
                p = 0
        self.p = p

    def get_sigma2(self, convert_sigma_to_external_units=False):
        if self.sigma2 is not None and convert_sigma_to_external_units and self.sigma_power is not None:
            if self.scale_factors is not None:
                if self.is_dense_gene_set is not None and np.sum(~self.is_dense_gene_set) > 0:
                    return self.sigma2 * np.mean(np.power(self.scale_factors[~self.is_dense_gene_set], self.sigma_power - 2))
                else:
                    return self.sigma2 * np.mean(np.power(self.scale_factors, self.sigma_power - 2))
            else:
                return self.sigma2 * np.power(self.MEAN_MOUSE_SCALE, self.sigma_power - 2)

        else:
            return self.sigma2

    def get_scaled_sigma2(self, scale_factors, sigma2, sigma_power, sigma_threshold_k, sigma_threshold_xo):
        threshold = 1
        if sigma_threshold_k is not None and sigma_threshold_xo is not None:
            threshold =  1 / (1 + np.exp(-sigma_threshold_k * (scale_factors - sigma_threshold_xo)))
        return threshold * sigma2 * np.power(scale_factors, sigma_power)

    def set_sigma(self, sigma2, sigma_power, sigma2_osc=None, sigma2_se=None, sigma2_p=None, sigma2_scale_factors=None, convert_sigma_to_internal_units=False):

        #if sigma2 is None:
        #    log("Set sigma called with sigma2=%s" % sigma2, TRACE)
        #else:
        #    log("Set sigma called with sigma2=%.3g" % sigma2, TRACE)            

        #WARNING: sigma storage is not handled optimally right now
        #if self.sigma_power=None, then sigma2 is in units of internal beta (because it is constant internally)
        #so, sigma2 / np.square(self.scale_factors[i]) is the external beta unit
        #if self.sigma_power != None, then sigma2 is in units of external beta (because it is constant externally)
        #so, sigma2 * np.power(self.scale_factors[i], self.sigma_power) is the internal beta unit
        #This setter does not validate any of this, so
        #1. when getting sigma, you must convert to internal sigma units
        #2. when setting sigma, you must pass in external units if const sigma and internal units if not


        self.sigma_power = sigma_power
        if sigma_power is None:
            #default is to have constant sigma in external units of beta, so beta is 2
            sigma_power = 2

        if convert_sigma_to_internal_units:
            #we divide by expected value of scale ** (power - 2) because:
            #beta_internal_j = beta_j * scale_factors_j -> beta_j = beta_internal_j / scale_factors_j
            #beta_internal_j ~ N(0, sigma2 * scale_factors_j ** sigma_power) -> beta_j ~ N(0, sigma2 * scale_factors_j ** (sigma_power-2))
            #sigma2_ext = E[sigma2 * scale_factors_j ** (sigma_power - 2)] = sigma2 * E[scale_factors_j ** (sigma_power - 2)]
            #sigma2 = sigma2_ext / E[scale_factors_j ** (sigma_power - 2)]
            if self.scale_factors is not None:
                if self.is_dense_gene_set is not None and np.sum(~self.is_dense_gene_set) > 0:
                    self.sigma2 = sigma2 / np.mean(np.power(self.scale_factors[~self.is_dense_gene_set], self.sigma_power - 2))
                else:
                    self.sigma2 = sigma2 / np.mean(np.power(self.scale_factors, self.sigma_power - 2))
            else:
                self.sigma2 = sigma2 / np.power(self.MEAN_MOUSE_SCALE, self.sigma_power - 2)
        else:
            self.sigma2 = sigma2

        if sigma2_osc is not None:
            self.sigma2_osc = sigma2_osc

        if sigma2_scale_factors is None:
            sigma2_scale_factors = self.scale_factors

        if sigma2_se is not None:
            self.sigma2_se = sigma2_se
        if self.sigma2_p is not None:
            self.sigma2_p = sigma2_p

        if self.sigma2 is None and self.sigma2_osc is None:
            return

        sigma2_for_var = self.sigma2_osc if self.sigma2_osc is not None else self.sigma2

        if sigma2_for_var is not None and sigma2_scale_factors is not None:
            if self.sigma_power is None:
                self.sigma2_total_var = sigma2_for_var * len(sigma2_scale_factors)
            else:
                self.sigma2_total_var = sigma2_for_var * np.sum(np.square(sigma2_scale_factors))

        if self.sigma2_total_var is not None and self.sigma2_se is not None:
            self.sigma2_total_var_lower = self.sigma2_total_var * (sigma2_for_var - 1.96 * self.sigma2_se)/(sigma2_for_var)
            self.sigma2_total_var_upper = self.sigma2_total_var * (sigma2_for_var + 1.96 * self.sigma2_se)/(sigma2_for_var)

        #minimum bound
        if self.sigma2 is None:
            return

    def calculate_sigma(self, V, sigma_power=None, chisq_threshold=None, chisq_dynamic=False, desired_intercept_difference=1.3):
        if self.z_scores is None:
            bail("Cannot calculate sigma with no stats loaded!")
        if V is None:
            V = self._get_V()
        if len(self.z_scores) == 0:
            bail("No gene sets were in both V and stats!")
        self.sigma_power = sigma_power
       
        log("Calculating OSC", DEBUG)

        #generating batches are most expensive part here, so do each one only once
        self.osc = np.zeros(self.X_orig.shape[1])
        if self.X_orig_missing_gene_sets is not None:
            self.osc_missing = np.zeros(self.X_orig_missing_gene_sets.shape[1])

        for X_b1, begin1, end1, batch1 in self._get_X_blocks(whiten=False, full_whiten=True):
            #begin block for calculating OSC between non-missing gene sets and non-missing gene sets
            for X_b2, begin2, end2, batch2 in self._get_X_blocks(start_batch=batch1, whiten=False, full_whiten=False):
                self.osc[begin1:end1] = np.add(self.osc[begin1:end1], np.sum(np.power(self._compute_V(X_b1, self.mean_shifts[begin1:end1], self.scale_factors[begin1:end1], X_orig2=X_b2, mean_shifts2=self.mean_shifts[begin2:end2], scale_factors2=self.scale_factors[begin2:end2]), 2), axis=1))
                if not batch1 == batch2:
                    self.osc[begin2:end2] = np.add(self.osc[begin2:end2], np.sum(np.power(self._compute_V(X_b2, self.mean_shifts[begin2:end2], self.scale_factors[begin2:end2], X_orig2=X_b1, mean_shifts2=self.mean_shifts[begin1:end1], scale_factors2=self.scale_factors[begin1:end1]), 2), axis=1))
            #end block for calculating OSC between non-missing gene sets and non-missing gene sets

            if self.X_orig_missing_gene_sets is not None:
                for X_m_b1, m_begin1, m_end1, m_batch1 in self._get_X_blocks(get_missing=True, whiten=False, full_whiten=False):
                    #since we have the missing blocks, do the missing/missing osc as well
                    #but only want to do this once
                    #osc between non-missing and missing
                    self.osc[begin1:end1] = np.add(self.osc[begin1:end1], np.sum(np.power(self._compute_V(X_b1, self.mean_shifts[begin1:end1], self.scale_factors[begin1:end1], X_orig2=X_m_b1, mean_shifts2=self.mean_shifts_missing[m_begin1:m_end1], scale_factors2=self.scale_factors_missing[m_begin1:m_end1]), 2), axis=1))
                    #osc between missing and non-missing
                    self.osc_missing[m_begin1:m_end1] = np.add(self.osc_missing[m_begin1:m_end1], np.sum(np.power(self._compute_V(X_m_b1, self.mean_shifts_missing[m_begin1:m_end1], self.scale_factors_missing[m_begin1:m_end1], X_orig2=X_b1, mean_shifts2 = self.mean_shifts[begin1:end1], scale_factors2 = self.scale_factors[begin1:end1]), 2), axis=1))

        if self.X_orig_missing_gene_sets is not None:
            for X_m_b1, m_begin1, m_end1, m_batch1 in self._get_X_blocks(get_missing=True, whiten=False, full_whiten=True):
                for X_m_b2, m_begin2, m_end2, m_batch2 in self._get_X_blocks(get_missing=True, whiten=False, full_whiten=False, start_batch=m_batch1):
                    self.osc_missing[m_begin1:m_end1] = np.add(self.osc_missing[m_begin1:m_end1], np.sum(np.power(self._compute_V(X_m_b1, self.mean_shifts_missing[m_begin1:m_end1], self.scale_factors_missing[m_begin1:m_end1], X_orig2=X_m_b2, mean_shifts2=self.mean_shifts_missing[m_begin2:m_end2], scale_factors2=self.scale_factors_missing[m_begin2:m_end2]), 2), axis=1))
                    if not m_batch1 == m_batch2:
                        self.osc_missing[m_begin2:m_end2] = np.add(self.osc_missing[m_begin2:m_end2], np.sum(np.power(self._compute_V(X_m_b2, self.mean_shifts_missing[m_begin2:m_end2], self.scale_factors_missing[m_begin2:m_end2], X_orig2=X_m_b1, mean_shifts2=self.mean_shifts_missing[m_begin1:m_end1], scale_factors2=self.scale_factors_missing[m_begin1:m_end1]), 2), axis=1))

        #X_osc is in units of standardized X
        self.X_osc = self.osc/np.square(self.ses)
        if self.X_orig_missing_gene_sets is not None:
            self.X_osc_missing = self.osc_missing/np.square(self.ses_missing)
            osc = np.append(self.osc, self.osc_missing)
            denominator = np.square(np.append(self.ses, self.ses_missing))
            scale_factors = np.append(self.scale_factors, self.scale_factors_missing)
            Y_chisq=np.square(np.append(self.z_scores, self.z_scores_missing))
        else:
            self.osc_missing = None
            self.X_osc_missing = None
            osc = self.osc
            denominator = np.square(self.ses)
            scale_factors = self.scale_factors
            Y_chisq=np.square(self.z_scores)

        X_osc = osc/denominator
        if self.sigma_power is not None:
            #all of the X_osc have been scaled by 1/scale_factor**2 (because each X is scaled by 1/scale_factor)
            #so we need to multiply by scale_factor
            if np.sum(~self.is_dense_gene_set) > 0:
                X_osc[~self.is_dense_gene_set] = X_osc[~self.is_dense_gene_set] * np.power(scale_factors[~self.is_dense_gene_set], self.sigma_power)
                X_osc[self.is_dense_gene_set] = X_osc[self.is_dense_gene_set] * np.power(np.mean(scale_factors[~self.is_dense_gene_set]), self.sigma_power)
                self.X_osc[~self.is_dense_gene_set] = self.X_osc[~self.is_dense_gene_set] * np.power(self.scale_factors[~self.is_dense_gene_set], self.sigma_power)
                self.X_osc[self.is_dense_gene_set] = self.X_osc[self.is_dense_gene_set] * np.power(np.mean(self.scale_factors[~self.is_dense_gene_set]), self.sigma_power)

                if self.X_orig_missing_gene_sets is not None:
                    self.X_osc_missing[~self.is_dense_gene_set_missing] = self.X_osc_missing[~self.is_dense_gene_set_missing] * np.power(self.scale_factors_missing[~self.is_dense_gene_set_missing], self.sigma_power)
                    self.X_osc_missing[self.is_dense_gene_set_missing] = self.X_osc_missing[self.is_dense_gene_set_missing] * np.power(np.mean(self.scale_factors[~self.is_dense_gene_set]), self.sigma_power)
            else:
                X_osc[self.is_dense_gene_set] = X_osc[self.is_dense_gene_set] * np.power(np.mean(scale_factors), self.sigma_power)
                self.X_osc[self.is_dense_gene_set] = self.X_osc[self.is_dense_gene_set] * np.power(np.mean(self.scale_factors), self.sigma_power)
                if self.X_orig_missing_gene_sets is not None:
                    self.X_osc_missing[self.is_dense_gene_set_missing] = self.X_osc_missing[self.is_dense_gene_set_missing] * np.power(np.mean(self.scale_factors), self.sigma_power)

        #X_osc is N l_j
        #osc is l_j

        #OLD
        #osc_weights = 1/(np.square(1 + X_osc) * X_osc)
        #NEW
        tau = (np.power(np.mean(Y_chisq), 2) - 1) / (np.mean(X_osc))
        osc_weights = 1/(osc * np.square(1 + X_osc * tau))

        #OLD
        #self.osc_weights = 1/(np.square(1 + self.X_osc) * self.X_osc)
        #NEW
        self.osc_weights = 1/(self.osc * np.square(1 + tau * self.X_osc))

        if self.X_orig_missing_gene_sets is not None:
            #OLD
            #self.osc_weights_missing = 1/(np.square(1 + self.X_osc_missing) * self.X_osc_missing)
            #NEW
            self.osc_weights_missing = 1/(self.osc_missing * np.square(1 + tau * self.X_osc_missing))
        else:
            self.osc_weights_missing = None

        orig_mask = ~(np.isnan(Y_chisq) | np.isinf(Y_chisq) | np.isnan(X_osc) | np.isinf(X_osc) | np.isnan(osc_weights) | np.isinf(osc_weights))

        if chisq_dynamic:
            cur_chisq_threshold = max(Y_chisq)
            best_intercept_difference = None
            best_chisq_threshold = None
            min_chisq_threshold = 5
        elif chisq_threshold is not None:
            cur_chisq_threshold = chisq_threshold
        else:
            cur_chisq_threshold = np.inf

        log("Running OSC regressions", DEBUG)

        while True:

            mask = np.logical_and(orig_mask, Y_chisq < cur_chisq_threshold)

            #run to get intercept
            constant_term = np.ones(len(X_osc[mask]))
            X = np.vstack((X_osc[mask], constant_term))
            Y = Y_chisq[mask] 
            #add weights for WLS
            X[0,] = X[0,] * np.sqrt(osc_weights[mask])
            X[1,] = X[1,] * np.sqrt(osc_weights[mask])
            Y = Y * np.sqrt(osc_weights[mask])
            try:
                mat_inv = np.linalg.inv(X.dot(X.T) + np.eye)
            except np.linalg.LinAlgError:
                mat_inv = np.linalg.inv(X.dot(X.T) + 0.2 * np.eye(X.shape[0]))

            result = mat_inv.dot(X.dot(Y))
            result_ses = np.sqrt(np.diag(np.var(Y - result.dot(X)) * mat_inv))
            cur_beta = result[0]
            cur_beta_se = result_ses[0]
            cur_beta_z = cur_beta / cur_beta_se
            cur_beta_p = 2*scipy.stats.norm.cdf(-np.abs(cur_beta_z))
            cur_intercept = result[1]
            cur_intercept_se = result_ses[1]
            cur_intercept_z = cur_intercept / cur_intercept_se
            cur_intercept_p = 2*scipy.stats.norm.cdf(-np.abs(cur_intercept_z))

            def __write_results(beta, beta_se, beta_p, intercept=None, intercept_se=None, intercept_p=None, level=INFO):
                log("=================================================", level=level)
                log("value        coef%s      std err      P" % ("" if beta > 0 and (intercept is None or intercept > 0) else " "), level=level)
                log("-------------------------------------------------", level=level)
                log("beta         %.4g%s    %.4g       %.4g" % (beta, " " if beta > 0 else "", beta_se, beta_p), level=level)
                if intercept is not None:
                    log("intercept    %.4g%s    %.4g       %.3g" % (intercept, " " if intercept > 0 else "", intercept_se, intercept_p), level=level)
                log("================================================", level=level)

            log("Results from full regression (chisq-threshold=%.3g):" % cur_chisq_threshold, TRACE)
            __write_results(cur_beta, cur_beta_se, cur_beta_p, cur_intercept, cur_intercept_se, cur_intercept_p, TRACE)

            #log("Results from regression with pinned intercept at 1:")

            #X = X[0,:]
            #Y = (Y_chisq[mask] - 1) * np.sqrt(osc_weights[mask])
            #beta = X.dot(Y) / X.dot(X)
            #beta_se = np.sqrt(np.var(Y - beta * X) / X.dot(X))
            #beta_z = beta / beta_se
            #beta_p = 2*scipy.stats.norm.cdf(-np.abs(beta_z))
            #__write_results(beta, beta_se, beta_p)

            #first log the results with the intercept
            if not chisq_dynamic:
                best_chisq_threshold = cur_chisq_threshold
                intercept = cur_intercept
                beta = cur_beta
                beta_se = cur_beta_se
                beta_p = cur_beta_p
                break
            else:
                if best_intercept_difference is None or np.abs(cur_intercept - 1) < best_intercept_difference:
                    best_intercept_difference = np.abs(cur_intercept - 1)
                    best_chisq_threshold=cur_chisq_threshold
                    intercept = cur_intercept
                    beta = cur_beta
                    beta_se = cur_beta_se
                    beta_p = cur_beta_p
                if cur_chisq_threshold < min_chisq_threshold or best_intercept_difference < desired_intercept_difference:
                    break
                else:
                    cur_chisq_threshold /= 1.5

        log("Results from full regression (chisq-threshold=%.3g):" % best_chisq_threshold, INFO)
        __write_results(cur_beta, cur_beta_se, cur_beta_p, cur_intercept, cur_intercept_se, cur_intercept_p, INFO)
        log("Final sigma results:")
        self.intercept = intercept
        self.set_sigma(beta, self.sigma_power, sigma2_osc=beta, sigma2_se=beta_se, sigma2_p=beta_p, sigma2_scale_factors=scale_factors)
        #self.write_params(None)

        #now log the results with the pinned
        #log("Results with pinned intercept:")
        #self.set_sigma(beta, self.sigma_power, beta_se, beta_p, sigma2_scale_factors=scale_factors)
        #self.write_sigma(None)

    def write_params(self, output_file):
        if output_file is not None:
            log("Writing params to %s" % output_file, INFO)
            params_fh = open(output_file, 'w')

            params_fh.write("Parameter\tVersion\tValue\n")
            for param in self.param_keys:
                if type(self.params[param]) == list:
                    values = self.params[param]
                else:
                    values = [self.params[param]]
                for i in range(len(values)):
                    params_fh.write("%s\t%s\t%s\n" % (param, i + 1, values[i]))
                        
            params_fh.close()

    def read_betas(self, betas_in):

        betas_format = "<gene_id> <beta>"

        if self.betas_in is None:
            bail("Operation requires --beta-in\nformat: %s" % (self.betas_format))

        log("Reading --betas-in file %s" % self.betas_in, INFO)

        with open_gz(betas_in) as betas_fh:
            id_col = 0
            beta_col = 1

            if self.gene_sets is not None:
                self.betas = np.zeros(len(self.gene_sets))
                subset_mask = np.array([False] * len(self.gene_sets))
            else:
                self.betas = []

            gene_sets = []
            gene_set_to_ind = {}

            ignored = 0
            for line in betas_fh:
                cols = line.strip().split()
                if id_col > len(cols) or beta_col > len(cols):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene_set = cols[id_col]
                if gene_set in gene_set_to_ind:
                    warn("Already seen gene set %s; only considering first instance" % (gene_set))
                try:
                    beta = float(cols[beta_col])
                except ValueError:
                    if not cols[beta_col] == "NA":
                        warn("Skipping unconvertible beta value %s for gene_set %s" % (cols[beta_col], gene_set))
                    continue
                
                gene_set_ind = None
                if self.gene_sets is not None:
                    if gene_set not in self.gene_set_to_ind:
                        ignored += 1
                        continue
                    gene_set_ind = self.gene_set_to_ind[gene_set]
                    if gene_set_ind is not None:
                        self.betas[gene_set_ind] = beta
                        subset_mask[gene_set_ind] = True
                else:
                    self.betas.append(beta)
                    #store these in all cases to be able to check for duplicate gene sets in the input
                    gene_set_to_ind[gene_set] = len(gene_sets)
                    gene_sets.append(gene_set)

            if self.gene_sets is not None:
                #need to subset existing marices
                if ignored > 0:
                    warn("Ignored %s values from --betas-in file because absent from previously loaded files" % ignored)
                if sum(subset_mask) != len(subset_mask):
                    warn("Excluding %s values from previously loaded files because absent from --betas-in file" % (len(subset_mask) - sum(subset_mask)))
                    self._subset_gene_sets(subset_mask, keep_missing=False)
            else:
                self.gene_sets = gene_sets
                self.gene_set_to_ind = gene_set_to_ind
                self.betas = np.array(self.betas).flatten()

            if self.normalize_betas:
                self.betas -= np.mean(self.betas)

    def calculate_inf_betas(self, update_hyper_sigma=True, max_num_iter=20, eps=0.01):
        #catch the "death spiral"
        orig_sigma2 = self.sigma2
        orig_inf_betas = None
        significant_decrease = 0
        total = 0
        converged = False

        V = self._get_V()

        if self.y_corr_sparse is not None:
            V_cor = self._calculate_V_internal(self.X_orig, None, self.mean_shifts, self.scale_factors, y_corr_sparse=self.y_corr_sparse)
            V_inv = self._invert_sym_matrix(V)
        else:
            V_cor = None
            V_inv = None

        for i in range(max_num_iter):
            inf_betas = self._calculate_inf_betas(V_cor=V_cor, V_inv=V_inv, se_inflation_factors=self.se_inflation_factors)

            if not update_hyper_sigma:
                break

            if orig_inf_betas is None:
                orig_inf_betas = inf_betas

            h2 = inf_betas.dot(V).dot(inf_betas)
            if self.sigma_power is not None:
                #np.sum(sigma2 * np.square(self.scale_factors)) = h2
                new_sigma2 = h2 / np.sum(np.power(self.scale_factors, self.sigma_power))
            else:
                new_sigma2 = h2 / len(inf_betas)
            if abs(new_sigma2 - self.sigma2) / self.sigma2 < eps:
                converged = True
                break
            log("Updating sigma to: %.4g" % new_sigma2, TRACE)

            total += 1
            if new_sigma2 < self.sigma2:
                significant_decrease += 1
            self.set_sigma(new_sigma2, self.sigma_power)
            if new_sigma2 == 0:
                break

        #don't degrade it too much
        if total > 0 and not converged and float(significant_decrease) / float(total) == 1:
            log("Reverting to original sigma=%.4g due to convergence to 0" % orig_sigma2, TRACE)
            inf_betas = orig_inf_betas
            self.set_sigma(orig_sigma2, self.sigma_power)

        if self.betas is None or self.betas is self.inf_betas:
            self.betas = inf_betas

        self.inf_betas = inf_betas

        if self.gene_sets_missing is not None:
            self.betas_missing = np.zeros(len(self.gene_sets_missing))
            self.betas_uncorrected_missing = np.zeros(len(self.gene_sets_missing))
            self.inf_betas_missing = np.zeros(len(self.gene_sets_missing))

    def calculate_non_inf_betas(self, p, max_num_burn_in=1000, max_num_iter=1100, min_num_iter=10, num_chains=100, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, max_frac_sem=0.01, gauss_seidel=False, update_hyper_sigma=True, update_hyper_p=True, sparse_solution=False, pre_filter_batch_size=None, pre_filter_small_batch_size=500, sparse_frac_betas=None, betas_trace_out=None, **kwargs):

        (avg_betas_uncorrected_v, avg_postp_uncorrected_v) = self._calculate_non_inf_betas(p, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, assume_independent=True, V=None, **kwargs)

        avg_betas_v = np.zeros(len(self.gene_sets))
        avg_postp_v = np.zeros(len(self.gene_sets))

        initial_run_mask = avg_betas_uncorrected_v > 0
        run_mask = copy.copy(initial_run_mask)

        if pre_filter_batch_size is not None and np.sum(initial_run_mask) > pre_filter_batch_size:
            self._record_param("pre_filter_batch_size_orig", pre_filter_batch_size)

            num_batches = self._get_num_X_blocks(self.X_orig[:,initial_run_mask], batch_size=pre_filter_small_batch_size)
            if num_batches > 1:
                #try to run with small batches to see if we can zero out more
                gene_set_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_orig[:,initial_run_mask], mean_shifts=self.mean_shifts[initial_run_mask], scale_factors=self.scale_factors[initial_run_mask], find_correlated_instead=pre_filter_small_batch_size)
                if len(gene_set_masks) > 0:
                    if np.sum(gene_set_masks[-1]) == 1 and len(gene_set_masks) > 1:
                        #merge singletons at the end into the one before
                        gene_set_masks[-2][gene_set_masks[-1]] = True
                        gene_set_masks = gene_set_masks[:-1]
                    if np.sum(gene_set_masks[0]) > 1:
                        V_data = []
                        V_rows = []
                        V_cols = []
                        for gene_set_mask in gene_set_masks:
                            V_block = self._calculate_V_internal(self.X_orig[:,initial_run_mask][:,gene_set_mask], self.y_corr_cholesky, self.mean_shifts[initial_run_mask][gene_set_mask], self.scale_factors[initial_run_mask][gene_set_mask])
                            orig_indices = np.where(gene_set_mask)[0]
                            V_rows += list(np.repeat(orig_indices, V_block.shape[0]))
                            V_cols += list(np.tile(orig_indices, V_block.shape[0]))
                            V_data += list(V_block.ravel())
                            
                        V_sparse = sparse.csc_matrix((V_data, (V_rows, V_cols)), shape=(np.sum(initial_run_mask), np.sum(initial_run_mask)))

                        log("Running %d blocks to check for zeros..." % len(gene_set_masks), DEBUG)
                        (avg_betas_half_corrected_v, avg_postp_half_corrected_v) = self._calculate_non_inf_betas(p, V=V_sparse, X_orig=None, scale_factors=self.scale_factors[initial_run_mask], mean_shifts=self.mean_shifts[initial_run_mask], is_dense_gene_set=self.is_dense_gene_set[initial_run_mask], ps=self.ps[initial_run_mask], sigma2s=self.sigma2s[initial_run_mask], max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=update_hyper_sigma, update_hyper_p=update_hyper_p, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, **kwargs)

                        add_zero_mask = avg_betas_half_corrected_v == 0

                        if np.any(add_zero_mask):
                            #need to convert these to the original gene sets
                            map_to_full = np.where(initial_run_mask)[0]
                            #get rows and then columns in subsetted
                            set_to_zero_full = np.where(add_zero_mask)
                            #map columns in subsetted to original
                            set_to_zero_full = map_to_full[set_to_zero_full]
                            orig_zero = np.sum(run_mask)
                            run_mask[set_to_zero_full] = False
                            new_zero = np.sum(run_mask)
                            log("Found %d additional zero gene sets" % (orig_zero - new_zero),DEBUG)

            if np.sum(~run_mask) > 0:
                log("Set additional %d gene sets to zero based on half corrected betas" % np.sum(~run_mask))

        #for i in range(len(self.gene_sets)):
        #    print("3\t%s\t%.3g\t%.3g" % (self.gene_sets[i], avg_betas_uncorrected_v[i], avg_postp_uncorrected_v[i]))

        (avg_betas_v[run_mask], avg_postp_v[run_mask]) = self._calculate_non_inf_betas(p, beta_tildes=self.beta_tildes[run_mask], ses=self.ses[run_mask], X_orig=self.X_orig[:,run_mask], scale_factors=self.scale_factors[run_mask], mean_shifts=self.mean_shifts[run_mask], V=None, ps=self.ps[run_mask] if self.ps is not None else None, sigma2s=self.sigma2s[run_mask] if self.sigma2s is not None else None, is_dense_gene_set=self.is_dense_gene_set[run_mask], max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=update_hyper_sigma, update_hyper_p=update_hyper_p, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, betas_trace_out=betas_trace_out, betas_trace_gene_sets=[self.gene_sets[i] for i in range(len(self.gene_sets)) if run_mask[i]], debug_gene_sets=[self.gene_sets[i] for i in range(len(self.gene_sets)) if run_mask[i]], **kwargs)

        if len(avg_betas_v.shape) == 2:
            avg_betas_v = np.mean(avg_betas_v, axis=0)
            avg_postp_v = np.mean(avg_postp_v, axis=0)

        self.betas = copy.copy(avg_betas_v)
        self.betas_uncorrected = copy.copy(avg_betas_uncorrected_v)

        self.non_inf_avg_postps = copy.copy(avg_postp_v)
        self.non_inf_avg_cond_betas = copy.copy(avg_betas_v)
        self.non_inf_avg_cond_betas[avg_postp_v > 0] /= avg_postp_v[avg_postp_v > 0]

        if self.gene_sets_missing is not None:
            self.betas_missing = np.zeros(len(self.gene_sets_missing))
            self.betas_uncorrected_missing = np.zeros(len(self.gene_sets_missing))
            self.non_inf_avg_postps_missing = np.zeros(len(self.gene_sets_missing))
            self.non_inf_avg_cond_betas_missing = np.zeros(len(self.gene_sets_missing))

    def calculate_priors(self, max_gene_set_p=None, num_gene_batches=None, correct_betas=True, gene_loc_file=None, gene_cor_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, p_noninf=None, tag="", **kwargs):
        if self.X_orig is None:
            bail("X is required for this operation")
        if self.betas is None:
            bail("betas are required for this operation")

        use_X = False

        assert(self.gene_sets is not None)
        max_num_gene_batches_together = 10000
        #if 0, don't use any V
        num_gene_batches_parallel = int(max_num_gene_batches_together / len(self.gene_sets))
        if num_gene_batches_parallel == 0:
            use_X = True
            log("Using low memory X instead of V in priors", TRACE)
            num_gene_batches_parallel = 1

        loco = False
        if num_gene_batches is None:
            log("Doing leave-one-chromosome-out cross validation for priors computation")
            loco = True

        if num_gene_batches is not None and num_gene_batches < 2:
            #this calculates the values for the non missing genes
            #use original X matrix here because we are rescaling betas back to those units
            priors = np.array(self.X_orig.dot(self.betas / self.scale_factors) - np.sum(self.mean_shifts * self.betas / self.scale_factors)).flatten()
            self.combined_prior_Ys = None
            self.combined_prior_Ys_for_regression = None
            self.combined_prior_Ys_adj = None
            self.combined_prior_Y_ses = None
            self.combined_Ds = None
            self.batches = None
        else:

            if loco:
                if gene_loc_file is None:
                    bail("Need --gene-loc-file for --loco")

                gene_chromosomes = {}
                batches = set()
                log("Reading gene locations")
                if self.gene_to_chrom is None:
                    self.gene_to_chrom = {}
                if self.gene_to_pos is None:
                    self.gene_to_pos = {}

                with open_gz(gene_loc_file) as gene_loc_fh:
                    for line in gene_loc_fh:
                        cols = line.strip().split()
                        if len(cols) != 6:
                            bail("Format for --gene-loc-file is:\n\tgene_id\tchrom\tstart\tstop\tstrand\tgene_name\nOffending line:\n\t%s" % line)
                        gene_name = cols[5]
                        if gene_name not in self.gene_to_ind:
                            continue

                        chrom = cols[1]
                        pos1 = int(cols[2])
                        pos2 = int(cols[3])

                        self.gene_to_chrom[gene_name] = chrom
                        self.gene_to_pos[gene_name] = (pos1,pos2)

                        batches.add(chrom)
                        gene_chromosomes[gene_name] = chrom
                batches = sorted(batches)
                num_gene_batches = len(batches)
            else:
                #need sorted genes and correlation matrix to batch genes
                if self.y_corr is None:
                    correlation_m = self._read_correlations(gene_cor_file, gene_loc_file, gene_cor_file_gene_col=gene_cor_file_gene_col, gene_cor_file_cor_start_col=gene_cor_file_cor_start_col)
                    self._set_Y(self.Y, self.Y_for_regression, self.Y_exomes, self.Y_positive_controls, Y_corr_m=correlation_m, skip_V=True, store_cholesky=False, skip_scale_factors=True, min_correlation=None)
                batches = range(num_gene_batches)

            gene_batch_size = int(len(self.genes) / float(num_gene_batches) + 1)
            self.batches = [None] * len(self.genes)
            priors = np.zeros(len(self.genes))

            #store a matrix of all beta_tildes across all batches
            full_matrix_shape = (len(batches), len(self.gene_sets) + (len(self.gene_sets_missing) if self.gene_sets_missing is not None else 0))
            full_beta_tildes_m = np.zeros(full_matrix_shape)
            full_ses_m = np.zeros(full_matrix_shape)
            full_z_scores_m = np.zeros(full_matrix_shape)
            full_se_inflation_factors_m = np.zeros(full_matrix_shape)
            full_p_values_m = np.zeros(full_matrix_shape)
            full_scale_factors_m = np.zeros(full_matrix_shape)
            full_ps_m = None
            if self.ps is not None:
                full_ps_m = np.zeros(full_matrix_shape)                
            full_sigma2s_m = None
            if self.sigma2s is not None:
                full_sigma2s_m = np.zeros(full_matrix_shape)                

            full_is_dense_gene_set_m = np.zeros(full_matrix_shape, dtype=bool)
            full_mean_shifts_m = np.zeros(full_matrix_shape)
            full_include_mask_m = np.zeros((len(batches), len(self.genes)), dtype=bool)
            full_priors_mask_m = np.zeros((len(batches), len(self.genes)), dtype=bool)

            #combine X_orig and X_orig missing
            revert_subset_mask = None
            if self.gene_sets_missing is not None:
                revert_subset_mask = self._unsubset_gene_sets(skip_V=True)

            for batch_ind in range(len(batches)):
                batch = batches[batch_ind]

                #specify:
                # (a) include_mask: the genes that are used for calculating beta tildes and betas for this batch
                # (b) priors_mask: the genes that we will calculate priors for
                #these are not exact complements because we may need to exlude some genes for both (i.e. a buffer)
                if loco:
                    include_mask = np.array([True] * len(self.genes))
                    priors_mask = np.array([False] * len(self.genes))
                    for i in range(len(self.genes)):
                        if self.genes[i] not in gene_chromosomes:
                            include_mask[i] = False
                            priors_mask[i] = True
                        elif gene_chromosomes[self.genes[i]] == batch:
                            include_mask[i] = False
                            priors_mask[i] = True
                        else:
                            include_mask[i] = True
                            priors_mask[i] = False
                    log("Batch %s: %d genes" % (batch, np.sum(priors_mask)))
                else:
                    begin = batch * gene_batch_size
                    end = (batch + 1) * gene_batch_size
                    if end > len(self.genes):
                        end = len(self.genes)
                    end = end - 1
                    log("Batch %d: genes %d - %d" % (batch+1, begin, end))


                    #include only genes not correlated with any in the current batch
                    include_mask = np.array([True] * len(self.genes))

                    include_mask_begin = begin - 1
                    while include_mask_begin > 0 and (begin - include_mask_begin) < len(self.y_corr) and self.y_corr[begin - include_mask_begin][include_mask_begin] > 0:
                        include_mask_begin -= 1
                    include_mask_begin += 1

                    include_mask_end = end + 1
                    while (include_mask_end - end) < len(self.y_corr) and self.y_corr[include_mask_end - end][end] > 0:
                        include_mask_end += 1
                    include_mask[include_mask_begin:include_mask_end] = False
                    include_mask_end -= 1

                    priors_mask = np.array([False] * len(self.genes))
                    priors_mask[begin:(end+1)] = True


                for i in range(len(self.genes)):
                    if priors_mask[i]:
                        self.batches[i] = batch

                #now subset Y
                Y = copy.copy(self.Y_for_regression)
                y_corr = None
                y_corr_cholesky = None
                y_corr_sparse = None

                if self.y_corr is not None:
                    y_corr = copy.copy(self.y_corr)
                    if not loco:
                        #we cannot rely on chromosome boundaries to zero out correlations, so manually do this
                        for i in range(include_mask_begin - 1, include_mask_begin - y_corr.shape[0], -1):
                            y_corr[include_mask_begin - i:,i] = 0
                    #don't need to zero out anything for include_mask_end because correlations between after end and removed are all stored inside of the removed indices
                    y_corr = y_corr[:,include_mask]

                    if self.y_corr_cholesky is not None:
                        Y = copy.copy(self.Y_fw)
                        #this is the correlation matrix we will use this batch
                        #it is a subsetted version of the self.y_corr but with the correlations with the removed genes zeroed out
                        y_corr_cholesky = self._get_y_corr_cholesky(y_corr)
                    elif self.y_corr_sparse is not None:
                        y_corr_sparse = self.y_corr_sparse[include_mask,:][:,include_mask]
                
                Y = Y[include_mask]
                y_var = np.var(Y)

                #DO WE NEED THIS??
                #y_mean = np.mean(Y)
                #Y = Y - y_mean

                (mean_shifts, scale_factors) = self._calc_X_shift_scale(self.X_orig[include_mask,:], y_corr_cholesky)

                #if some gene sets became empty!
                assert(not np.any(np.logical_and(mean_shifts != 0, scale_factors == 0)))
                mean_shifts[mean_shifts == 0] = 0
                scale_factors[scale_factors == 0] = 1

                ps = self.ps
                sigma2s = self.sigma2s
                is_dense_gene_set = self.is_dense_gene_set

                #max_gene_set_p = self.max_gene_set_p if self.max_gene_set_p is not None else 1

                #compute special beta tildes here
                if self.is_logistic:
                    Y_to_use = Y
                    Y = np.exp(Y_to_use + self.background_log_bf) / (1 + np.exp(Y_to_use + self.background_log_bf))
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors, alpha_tildes, diverged) = self._compute_logistic_beta_tildes(self.X_orig[include_mask,:], Y, scale_factors, mean_shifts, resid_correlation_matrix=y_corr_sparse)
                else:
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = self._compute_beta_tildes(self.X_orig[include_mask,:], Y, y_var, scale_factors, mean_shifts, resid_correlation_matrix=y_corr_sparse)

                if correct_betas:
                    (beta_tildes, ses, z_scores, p_values, se_inflation_factors) = self._correct_beta_tildes(beta_tildes, ses, se_inflation_factors, self.total_qc_metrics, self.mean_qc_metrics, add_missing=True)

                #now determine those that have too many genes removed to be accurate
                mean_reduction = float(num_gene_batches - 1) / float(num_gene_batches)
                sd_reduction = np.sqrt(mean_reduction * (1 - mean_reduction))
                reduction = mean_shifts / self.mean_shifts
                ignore_mask = reduction < mean_reduction - 3 * sd_reduction
                if sum(ignore_mask) > 0:
                    log("Ignoring %d gene sets because there are too many genes are missing from this batch" % sum(ignore_mask))
                    for ind in np.array(range(len(ignore_mask)))[ignore_mask]:
                        log("%s: %.4g remaining (vs. %.4g +/- %.4g expected)" % (self.gene_sets[ind], reduction[ind], mean_reduction, sd_reduction), TRACE)
                #also zero out anything above the p-value threshold; this is a convenience for below
                #note that p-values are still preserved though for below
                ignore_mask = np.logical_or(ignore_mask, p_values > max_gene_set_p)

                beta_tildes[ignore_mask] = 0
                ses[ignore_mask] = max(self.ses) * 100

                full_beta_tildes_m[batch_ind,:] = beta_tildes
                full_ses_m[batch_ind,:] = ses
                full_z_scores_m[batch_ind,:] = z_scores
                full_se_inflation_factors_m[batch_ind,:] = se_inflation_factors
                full_p_values_m[batch_ind,:] = p_values
                full_scale_factors_m[batch_ind,:] = scale_factors
                full_mean_shifts_m[batch_ind,:] = mean_shifts
                if full_ps_m is not None:
                    full_ps_m[batch_ind,:] = ps
                if full_sigma2s_m is not None:
                    full_sigma2s_m[batch_ind,:] = sigma2s

                full_is_dense_gene_set_m[batch_ind,:] = is_dense_gene_set
                full_include_mask_m[batch_ind,:] = include_mask
                full_priors_mask_m[batch_ind,:] = priors_mask

            #now calculate everything
            if p_noninf is None or p_noninf >= 1:
                num_gene_batches_parallel = 1
            num_calculations = int(np.ceil(num_gene_batches / num_gene_batches_parallel))
            for calc in range(num_calculations):
                begin = calc * num_gene_batches_parallel
                end = (calc + 1) * num_gene_batches_parallel
                if end > num_gene_batches:
                    end = num_gene_batches
                
                log("Running calculations for batches %d-%d" % (begin, end))

                #ensure there is at least one gene set remaining
                max_gene_set_p_v = np.min(full_p_values_m[begin:end,:], axis=1)
                #max_gene_set_p_v[max_gene_set_p_v < (self.max_gene_set_p if self.max_gene_set_p is not None else 1)] = (self.max_gene_set_p if self.max_gene_set_p is not None else 1)
                max_gene_set_p_v[max_gene_set_p_v < (max_gene_set_p if max_gene_set_p is not None else 1)] = (max_gene_set_p if max_gene_set_p is not None else 1)

                #get the include mask; any batch has p <= threshold
                new_gene_set_mask = np.max(full_p_values_m[begin:end,:].T <= max_gene_set_p_v, axis=1)
                num_gene_set_mask = np.sum(new_gene_set_mask)

                #we unsubset genes to aid in batching; this caused sigma and p to be affected
                fraction_non_missing = np.mean(new_gene_set_mask)
                missing_scale_factor = self._get_fraction_non_missing() / fraction_non_missing
                if missing_scale_factor > 1 / self.p:
                    #threshold this here. otherwise set_p will cap p but set_sigma won't cap sigma
                    missing_scale_factor = 1 / self.p
                
                #orig_sigma2 = self.sigma2
                #orig_p = self.p
                #self.set_sigma(self.sigma2 * missing_scale_factor, self.sigma_power, sigma2_osc=self.sigma2_osc)
                #self.set_p(self.p * missing_scale_factor)

                #construct the V matrix
                if not use_X:
                    V_m = np.zeros((end-begin, num_gene_set_mask, num_gene_set_mask))
                    for i,j in zip(range(begin, end),range(end-begin)):
                        include_mask = full_include_mask_m[i,:]

                        V_m[j,:,:] = self._calculate_V_internal(self.X_orig[include_mask,:][:,new_gene_set_mask], y_corr_cholesky, full_mean_shifts_m[i,new_gene_set_mask], full_scale_factors_m[i,new_gene_set_mask])
                else:
                    V_m = None

                cur_beta_tildes = full_beta_tildes_m[begin:end,:][:,new_gene_set_mask]
                cur_ses = full_ses_m[begin:end,:][:,new_gene_set_mask]
                cur_se_inflation_factors = full_se_inflation_factors_m[begin:end,:][:,new_gene_set_mask]
                cur_scale_factors = full_scale_factors_m[begin:end,:][:,new_gene_set_mask]
                cur_mean_shifts = full_mean_shifts_m[begin:end,:][:,new_gene_set_mask]
                cur_is_dense_gene_set = full_is_dense_gene_set_m[begin:end,:][:,new_gene_set_mask]
                cur_ps = None
                if full_ps_m is not None:
                    cur_ps = full_ps_m[begin:end,:][:,new_gene_set_mask]
                cur_sigma2s = None
                if full_sigma2s_m is not None:
                    cur_sigma2s = full_sigma2s_m[begin:end,:][:,new_gene_set_mask]

                #only non inf now
                (betas, avg_postp) = self._calculate_non_inf_betas(None, beta_tildes=cur_beta_tildes, ses=cur_ses, V=V_m, X_orig=self.X_orig[include_mask,:][:,new_gene_set_mask], scale_factors=cur_scale_factors, mean_shifts=cur_mean_shifts, is_dense_gene_set=cur_is_dense_gene_set, ps=cur_ps, sigma2s=cur_sigma2s, update_hyper_sigma=False, update_hyper_p=False, num_missing_gene_sets=int((1 - fraction_non_missing) * len(self.gene_sets)), **kwargs)
                if len(betas.shape) == 1:
                    betas = betas[np.newaxis,:]


                #if do inf:
                #    V = V_m[0,:,:]
                #    if self.y_corr_cholesky is None and self.y_corr_sparse is not None:
                #        V_cor = self._calculate_V_internal(self.X_orig[full_include_mask_m[begin,:],:][:,new_gene_set_mask], None, cur_mean_shifts, cur_scale_factors, y_corr_sparse=y_corr_sparse)
                #        V_inv = self._invert_sym_matrix(V)
                #    else:
                #        V_cor = None
                #        V_inv = None
                #    betas = self._calculate_inf_betas(beta_tildes=cur_beta_tildes, ses=cur_ses, V=V, V_cor=V_cor, V_inv=V_inv, se_inflation_factors=cur_se_inflation_factors, scale_factors=cur_scale_factors, is_dense_gene_set=cur_is_dense_gene_set)
                #    betas = betas[np.newaxis,:]

                for i,j in zip(range(begin, end),range(end-begin)):

                    priors[full_priors_mask_m[i,:]] = np.array(self.X_orig[full_priors_mask_m[i,:],:][:,new_gene_set_mask].dot(betas[j,:] / cur_scale_factors[j,:]))

                
                #ind = self.gene_to_ind['GIP']
                #if full_priors_mask_m[i,ind]:
                #    print("TEMP JASON!!!")
                #    print(ind)
                #    print(self.X_orig[ind,:])
                #    print(self.gene_sets[4])
                #    print(self.X_orig[:,4])
                #    print(self.X_orig[ind,4])
                #    print(betas)
                #    print(cur_scale_factors)
                #    print(priors[ind])


                #now restore the p and sigma
                #self.set_sigma(orig_sigma2, self.sigma_power, sigma2_osc=self.sigma2_osc)
                #self.set_p(orig_p)

            #now restore previous subsets
            self._subset_gene_sets(revert_subset_mask, keep_missing=True, skip_V=True)

        #now for the genes that were not included in X
        if self.X_orig_missing_genes is not None:
            #these can use the original betas because they were never included
            self.priors_missing = np.array(self.X_orig_missing_genes.dot(self.betas / self.scale_factors) - np.sum(self.mean_shifts * self.betas / self.scale_factors))
        else:
            self.priors_missing = np.array([])

        #store in member variable
        total_mean = np.mean(np.concatenate((priors, self.priors_missing)))
        self.priors = priors - total_mean
        self.priors_missing -= total_mean

        #do the regression
        gene_N = self.get_gene_N()
        gene_N_missing = self.get_gene_N(get_missing=True)
        all_gene_N = gene_N
        if self.genes_missing is not None:
            assert(gene_N_missing is not None)
            all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

        total_priors = np.concatenate((self.priors, self.priors_missing))
        priors_slope = np.cov(total_priors, all_gene_N)[0,1] / np.var(all_gene_N)
        priors_intercept = np.mean(total_priors - all_gene_N * priors_slope)

        log("Adjusting priors with slope %.4g" % priors_slope)
        self.priors_adj = self.priors - priors_slope * gene_N - priors_intercept
        if self.genes_missing is not None:
            self.priors_adj_missing = self.priors_missing - priors_slope * gene_N_missing


    def do_gibbs(self, min_num_iter=2, max_num_iter=100, num_chains=10, num_mad=3, r_threshold_burn_in=1.01, max_frac_sem=0.01, use_max_r_for_convergence=True, p_noninf=None, increase_hyper_if_betas_below=None, update_huge_scores=True, top_gene_prior=None, min_num_burn_in=10, max_num_burn_in=None, max_num_iter_betas=1100, min_num_iter_betas=10, num_chains_betas=4, r_threshold_burn_in_betas=1.01, use_max_r_for_convergence_betas=True, max_frac_sem_betas=0.01, use_mean_betas=True, sparse_frac_gibbs=0.01, sparse_solution=False, sparse_frac_betas=None, pre_filter_batch_size=None, pre_filter_small_batch_size=500, max_allowed_batch_correlation=None, gauss_seidel_betas=False, gauss_seidel=False, num_gene_batches=None, num_batches_parallel=10, max_mb_X_h=100, initial_linear_filter=True, correct_betas=True, gene_set_stats_trace_out=None, gene_stats_trace_out=None, betas_trace_out=None, eps=0.01):

        passed_in_max_num_burn_in = max_num_burn_in
        if max_num_burn_in is None:
            max_num_burn_in = int(max_num_iter * .25)
        if max_num_burn_in >= max_num_iter:
            max_num_burn_in = int(max_num_iter * .25)


        elif num_chains < 2:
            num_chains = 2

        self._record_params({"num_chains": num_chains, "num_chains_betas": num_chains_betas, "use_mean_betas": use_mean_betas, "sparse_solution": sparse_solution, "sparse_frac": sparse_frac_gibbs, "sparse_frac_betas": sparse_frac_betas, "pre_filter_batch_size": pre_filter_batch_size, "max_allowed_batch_correlation": max_allowed_batch_correlation, "initial_linear_filter": initial_linear_filter, "correct_betas": correct_betas})

        log("Running Gibbs")

        #save all of the old values
        self.beta_tildes_orig = copy.copy(self.beta_tildes)
        self.p_values_orig = copy.copy(self.p_values)
        self.ses_orig = copy.copy(self.ses)
        self.z_scores_orig = copy.copy(self.z_scores)
        self.beta_tildes_missing_orig = copy.copy(self.beta_tildes_missing)
        self.p_values_missing_orig = copy.copy(self.p_values_missing)
        self.ses_missing_orig = copy.copy(self.ses_missing)
        self.z_scores_missing_orig = copy.copy(self.z_scores_missing)

        self.betas_orig = copy.copy(self.betas)
        self.betas_uncorrected_orig = copy.copy(self.betas_uncorrected)
        self.inf_betas_orig = copy.copy(self.inf_betas)
        self.non_inf_avg_cond_betas_orig = copy.copy(self.non_inf_avg_cond_betas)
        self.non_inf_avg_postps_orig = copy.copy(self.non_inf_avg_postps)
        self.betas_missing_orig = copy.copy(self.betas_missing)
        self.betas_uncorrected_missing_orig = copy.copy(self.betas_uncorrected_missing)
        self.inf_betas_missing_orig = copy.copy(self.inf_betas_missing)
        self.non_inf_avg_cond_betas_missing_orig = copy.copy(self.non_inf_avg_cond_betas_missing)
        self.non_inf_avg_postps_missing_orig = copy.copy(self.non_inf_avg_postps_missing)

        self.Y_orig = copy.copy(self.Y)
        self.Y_for_regression_orig = copy.copy(self.Y_for_regression)
        self.Y_w_orig = copy.copy(self.Y_w)
        self.Y_fw_orig = copy.copy(self.Y_fw)
        self.priors_orig = copy.copy(self.priors)
        self.priors_adj_orig = copy.copy(self.priors_adj)
        self.priors_missing_orig = copy.copy(self.priors_missing)
        self.priors_adj_missing_orig = copy.copy(self.priors_adj_missing)


        #we always update correlation relative to the original one
        y_var_orig = np.var(self.Y_for_regression)

        #set up constants throughout the loop

        Y_to_use = self.Y_for_regression_orig
        bf_orig = np.exp(Y_to_use)

        bf_orig_raw = np.exp(self.Y_orig)

        #conditional variance of Y given beta: calculate residuals given priors
        priors_guess = np.array(self.X_orig.dot(self.betas / self.scale_factors) - np.sum(self.mean_shifts * self.betas / self.scale_factors))

        Y_resid = np.var(self.Y_for_regression_orig - priors_guess)
        Y_cond_var = Y_resid

        if top_gene_prior is not None:
            if top_gene_prior <= 0 or top_gene_prior >= 1:
                bail("--top-gene-prior needs to be in (0,1)")
            Y_total_var = self.convert_prior_to_var(top_gene_prior, len(self.genes))
            Y_cond_var = Y_total_var - self.get_sigma2(convert_sigma_to_external_units=True) * np.mean(self.get_gene_N())
            if Y_cond_var < 0:
                #minimum value
                Y_cond_var = 0.1
            log("Setting Y cond var=%.4g (total var = %.4g) given top gene prior of %.4g" % (Y_cond_var, Y_total_var, top_gene_prior))

        Y_cond_sd = np.sqrt(Y_cond_var)

        #FIXME FIXME FIXME
        #print("THE VARIANCES NEED TO BE WORKED THROUGH!!! -- should be residuals estimated at the start")
        #the variance of Y conditional upon the prior is the remaining variance of Y after subtracting out the variance for the prior
        #for i in range(len(self.Y_orig)):
        #    print("%s\t%.3g\t%.3g\t%.3g\t%.3g" % (self.genes[i], self.Y_orig[i], priors_guess[i], self.Y_orig[i] - priors_guess[i], (self.Y_orig[i] - priors_guess[i])**2))

        #THIS DOESN'T SEEM TO WORK

        ##first calculate sum of sigmas for the gene set that the gene is in
        ##this is sum(Xi^2*sigmai^2)
        #if self.sigma_power:
            #sigma2_v = self.sigma2 * np.square(self.scale_factors)
        #else:
            #sigma2_v = self.sigma2
        #Y_explained_var = self.X_orig.dot(np.square(self.scale_factors) * sigma2_v)
        #print("EXPLAINED Y VAR %s" % Y_explained_var)

        #Y_cond_var = Y_orig_var - Y_explained_var
        #Y_cond_sd = np.sqrt(Y_cond_var)
        #Y_cond_sd_m = np.tile(Y_cond_sd, num_chains).reshape(num_chains, len(bf_orig))

        #print("COND Y VAR %s" % Y_cond_var)

        #temp_genes = ["LDLR", "APOE", "TM6SF2", "SLC30A8", "TP53", "CDKN1A", "IL6","POLR2A"]
        #temp_genes = [x for x in temp_genes if x in self.gene_to_ind]

        #for gene in temp_genes:
        #    print("%s: %s" % (gene, self.Y[self.gene_to_ind[gene]]))
        #print("Mean: %s" % (np.mean(self.Y)))

        #this is the density of the relative (log) prior odds

        bf_orig_m = np.tile(bf_orig, num_chains).reshape(num_chains, len(bf_orig))
        log_bf_m = np.log(bf_orig_m)

        bf_orig_raw_m = np.tile(bf_orig_raw, num_chains).reshape(num_chains, len(bf_orig_raw))
        log_bf_raw_m = np.log(bf_orig_raw_m)

        compute_Y_raw = np.any(~np.isclose(log_bf_m, log_bf_raw_m))

        #we will adjust this to preserve the original probabilities if requested
        cur_background_log_bf_v = np.tile(self.background_log_bf, num_chains)

        def __density_fun(x, loc, scale, bf=bf_orig, background_log_bf=self.background_log_bf, do_expected=False):
            if type(x) == np.ndarray:
                prob = np.ones(x.shape)
                okay_mask = x < 10
                #we need absolute odds (not relative) for this calculation so add in background_log_bf
                x_odds = np.exp(x[okay_mask] + background_log_bf)
                prob[okay_mask] =  x_odds / (1 + x_odds)
            else:
                if x < 10:
                    x_odds = np.exp(x + background_log_bf)
                    prob = x_odds / (1 + x_odds)
                else:
                    prob = 1
        
            density = (bf * prob + (1 - prob)) * scipy.stats.norm.pdf(x, loc=loc, scale=scale)
            if do_expected:
                return np.dstack((density.T, x * density.T, np.square(x) * density.T)).T
            else:
                return density

        def __outlier_resistant_mean(sum_m, num_sum_m, outlier_mask_m=None):
            if outlier_mask_m is None:

                self._record_param("mad_threshold", num_mad)

                #1. calculate mean values for each chain (divide by number -- make sure it is correct; may not be num_avg_Y)
                chain_means_m = sum_m / num_sum_m

                #2. calculate median values across chains (one number per gene set/gene)
                medians_v = np.median(chain_means_m, axis=0)

                #3. calculate abs(difference) between each chain and median (one value per chain/geneset)
                mad_m = np.abs(chain_means_m - medians_v)

                #4. calculate median of abs(difference) across chains (one number per gene set/gene)
                mad_median_v = np.median(mad_m, axis=0)

                #5. mask any chain that is more than 3 median(abs(difference)) from median
                outlier_mask_m = chain_means_m > medians_v + num_mad * mad_median_v

            #6. take average only across chains that are not outliers
            num_sum_v = np.sum(~outlier_mask_m, axis=0)

            #should never happen but just in case
            num_sum_v[num_sum_v == 0] = 1

            #7. to do this, zero out outlier chains, then sum them, then divide by number of outliers
            copy_sum_m = copy.copy(sum_m)
            copy_sum_m[outlier_mask_m] = 0
            avg_v = np.sum(copy_sum_m / num_sum_m, axis=0) / num_sum_v
            
            return (outlier_mask_m, avg_v)


        #initialize Y

        if self.y_corr_cholesky is not None:
            bail("GLS not implemented yet for Gibbs sampling!")

        #dimensions of matrices are (num_chains, num_gene_sets)

        num_full_gene_sets = len(self.gene_sets)
        if self.gene_sets_missing is not None:
            num_full_gene_sets += len(self.gene_sets_missing)

        beta_tilde_outlier_z_threshold = None

        #this loop checks if the gibbs loop was successful

        max_num_restarts = 20
        num_p_increases = 0

        for num_restarts in range(0,max_num_restarts+1):

            #by default it succeeded
            gibbs_good = True

            #for increasing p option
            p_scale_factor = 1 - np.log(self.p)/(2 * np.log(10))
            num_before_checking_p_increase = max(min_num_iter, min_num_burn_in)
            if increase_hyper_if_betas_below is not None and num_before_checking_p_increase > min_num_iter:
                #make sure that we always trigger this check before breaking
                min_num_iter = num_before_checking_p_increase

            self._record_param("num_gibbs_restarts", num_restarts, overwrite=True)
            if num_restarts > 0:
                log("Gibbs restart %d" % num_restarts)

            num_restarts += 1

            burn_in_phase_beta_v = np.full(num_full_gene_sets, True)

            #set_to_zero_v = np.zeros(num_full_gene_sets)
            #avg_full_betas_sample_v = np.zeros(num_full_gene_sets)
            #avg_full_postp_sample_v = np.zeros(num_full_gene_sets)


            #sum of values for each chain
            #TODO: add in values for everything that has sum

            full_betas_m_shape = (num_chains, num_full_gene_sets)
            sum_betas_m = np.zeros(full_betas_m_shape)
            sum_betas2_m = np.zeros(full_betas_m_shape)
            sum_betas_uncorrected_m = np.zeros(full_betas_m_shape)
            sum_postp_m = np.zeros(full_betas_m_shape)
            sum_beta_tildes_m = np.zeros(full_betas_m_shape)
            sum_z_scores_m = np.zeros(full_betas_m_shape)
            num_sum_beta_m = np.zeros(full_betas_m_shape)

            Y_m_shape = (num_chains, len(self.Y_for_regression))
            burn_in_phase_Y_v = np.full(Y_m_shape[1], True)
            sum_Ys_m = np.zeros(Y_m_shape)
            sum_Ys2_m = np.zeros(Y_m_shape)
            sum_Y_raws_m = np.zeros(Y_m_shape)
            sum_log_pos_m = np.zeros(Y_m_shape)
            sum_log_pos2_m = np.zeros(Y_m_shape)
            sum_log_po_raws_m = np.zeros(Y_m_shape)
            sum_priors_m = np.zeros(Y_m_shape)
            sum_Ds_m = np.zeros(Y_m_shape)
            sum_D_raws_m = np.zeros(Y_m_shape)
            sum_bf_orig_m = np.zeros(Y_m_shape)
            sum_bf_orig_raw_m = np.zeros(Y_m_shape)
            num_sum_Y_m = np.zeros(Y_m_shape)

            #sums across all iterations, not just converged
            all_sum_betas_m = np.zeros(full_betas_m_shape)
            all_sum_betas2_m = np.zeros(full_betas_m_shape)
            all_sum_z_scores_m = np.zeros(full_betas_m_shape)
            all_sum_z_scores2_m = np.zeros(full_betas_m_shape)
            all_num_sum_m = np.zeros(full_betas_m_shape)

            all_sum_Ys_m = np.zeros(Y_m_shape)
            all_sum_Ys2_m = np.zeros(Y_m_shape)

            #num_sum = 0

            #sum_Ys_post_m = np.zeros(Y_m_shape)
            #sum_Ys2_post_m = np.zeros(Y_m_shape)
            #num_sum_post = 0

            #sum across all chains

            #avg_betas = np.zeros(num_full_gene_sets)
            #avg_betas2 = np.zeros(num_full_gene_sets)
            #avg_betas_uncorrected = np.zeros(num_full_gene_sets)

            #avg_postp = np.zeros(num_full_gene_sets)
            #avg_beta_tildes = np.zeros(num_full_gene_sets)
            #avg_z_scores = np.zeros(num_full_gene_sets)
            #avg_Ys = np.zeros(len(self.Y))
            #avg_Ys2 = np.zeros(len(self.Y))
            #avg_log_pos = np.zeros(len(self.Y))
            #avg_log_pos2 = np.zeros(len(self.Y))
            #avg_priors = np.zeros(len(self.Y))
            #avg_Ds = np.zeros(len(self.Y))
            #avg_bf_orig = np.zeros(len(self.Y))

            #num_avg_beta = np.zeros(num_full_gene_sets)
            #num_avg_Y = np.zeros(len(self.Y))

            #initialize the priors
            priors_sample_m = np.zeros(Y_m_shape)
            priors_mean_m = np.zeros(Y_m_shape)

            priors_percentage_max_sample_m = np.zeros(Y_m_shape)
            priors_percentage_max_mean_m = np.zeros(Y_m_shape)
            priors_adjustment_sample_m = np.zeros(Y_m_shape)
            priors_adjustment_mean_m = np.zeros(Y_m_shape)

            priors_for_Y_m = priors_sample_m
            priors_percentage_max_for_Y_m = priors_percentage_max_sample_m
            priors_adjustment_for_Y_m = priors_adjustment_sample_m
            if use_mean_betas:
                priors_for_Y_m = priors_mean_m                
                priors_percentage_max_for_Y_m = priors_percentage_max_mean_m
                priors_adjustment_for_Y_m = priors_adjustment_mean_m

            num_genes_missing = 0
            if self.genes_missing is not None:
                num_genes_missing = len(self.genes_missing)

            sum_priors_missing_m = np.zeros((num_chains, num_genes_missing))
            sum_Ds_missing_m = np.zeros((num_chains, num_genes_missing))

            #avg_priors_missing = np.zeros(num_genes_missing)
            #avg_Ds_missing = np.zeros(num_genes_missing)

            priors_missing_sample_m = np.zeros(sum_priors_missing_m.shape)
            priors_missing_mean_m = np.zeros(sum_priors_missing_m.shape)
            num_sum_priors_missing_m = np.zeros(sum_priors_missing_m.shape)

            if gene_set_stats_trace_out is not None:

                gene_set_stats_trace_fh = open_gz(gene_set_stats_trace_out, 'w')
                gene_set_stats_trace_fh.write("It\tChain\tGene_Set\tbeta_tilde\tP\tZ\tSE\tbeta_uncorrected\tbeta\tpostp\tbeta_tilde_outlier_z\tR\tSEM\n")
            if gene_stats_trace_out is not None:
                gene_stats_trace_fh = open_gz(gene_stats_trace_out, 'w')
                gene_stats_trace_fh.write("It\tChain\tGene\tprior\tcombined\tlog_bf\tD\tpercent_top\tadjust\n")

            #TEMP STUFF
            only_genes = None
            only_gene_sets = None

            if self.gene_sets_missing is not None:
                revert_subset_mask = self._unsubset_gene_sets(skip_V=True)

            prev_Ys_m = None
            #cache this
            X_hstacked = None
            stack_batch_size = num_chains + 1
            if num_chains > 1:

                X_size_mb = self._get_X_size_mb()
                X_h_size_mb =  num_chains * X_size_mb
                if X_h_size_mb <= max_mb_X_h:
                    X_hstacked = sparse.hstack([self.X_orig] * num_chains)
                else:
                    stack_batch_size = int(max_mb_X_h / X_size_mb)
                    if stack_batch_size == 0:
                        stack_batch_size = 1
                    log("Not building X_hstacked, size would be %d > %d; will instead run %d chains at a time" % (X_h_size_mb, max_mb_X_h, stack_batch_size))
                    X_hstacked = sparse.hstack([self.X_orig] * stack_batch_size)
            else:
                X_hstacked = self.X_orig

            num_stack_batches = int(np.ceil(num_chains / stack_batch_size))

            X_all = self.X_orig
            if self.genes_missing is not None:
                X_all = sparse.vstack((self.X_orig, self.X_orig_missing_genes))

            for iteration_num in range(max_num_iter):

                log("Beginning Gibbs iteration %d" % (iteration_num+1))
                self._record_param("num_gibbs_iter", iteration_num, overwrite=True)

                log("Sampling new Ys")

                log("Setting logistic Ys", TRACE)

                Y_sample_m = priors_for_Y_m + log_bf_m
                Y_raw_sample_m = priors_for_Y_m + log_bf_raw_m
                y_var = np.var(Y_sample_m, axis=1)

                #if adjust_background_prior:
                #    #get the original mean bf
                #    background_log_prior_scale_factor = np.mean(Y_sample_m, axis=1) - np.mean(log_bf_orig_m, axis=1)
                #    cur_background_log_bf_v = self.background_log_bf + background_log_prior_scale_factor
                #    cur_background_bf_v = np.exp(cur_background_log_bf_v)
                #    log("Adjusting background priors to %.4g-%.4g" % (np.min(cur_background_bf_v / (1 + cur_background_bf_v)), np.max(cur_background_bf_v / (1 + cur_background_bf_v))))

                #threshold in case things go off the rails
                max_log = 15

                cur_log_bf_m = Y_sample_m.T + cur_background_log_bf_v
                cur_log_bf_m[cur_log_bf_m > max_log] = max_log
                bf_sample_m = np.exp(cur_log_bf_m).T

                cur_log_bf_raw_m = Y_raw_sample_m.T + cur_background_log_bf_v
                cur_log_bf_raw_m[cur_log_bf_raw_m > max_log] = max_log
                bf_raw_sample_m = np.exp(cur_log_bf_raw_m).T

                #FIXME: do we really need to add in background_log_bf and then subtract it below? Surely there must be a way to do these calculations independent of background_log_bf? Can check by seeing if results are invariant to background_log_bf
                max_D = 1-1e-5
                min_D = 1e-5

                D_sample_m = bf_sample_m / (1 + bf_sample_m)
                D_sample_m[D_sample_m > max_D] = max_D
                D_sample_m[D_sample_m < min_D] = min_D
                log_po_sample_m = np.log(D_sample_m/(1-D_sample_m))

                D_raw_sample_m = bf_raw_sample_m / (1 + bf_raw_sample_m)
                D_raw_sample_m[D_raw_sample_m > max_D] = max_D
                D_raw_sample_m[D_raw_sample_m < min_D] = min_D
                log_po_raw_sample_m = np.log(D_raw_sample_m/(1-D_raw_sample_m))

                #if center_combined:
                #    #recenter the log_pos and Ds as well
                #    #the "combined missing" is just the priors
                #    log_po_sample_total_m = np.hstack((log_po_sample_m, priors_missing_sample_m))
                #    total_po_mean_v = np.mean(log_po_sample_total_m, axis=1)
                #    log_po_sample_m = (log_po_sample_m.T - total_po_mean_v).T
                #    bf_sample_m = np.exp(log_po_sample_m.T + cur_background_log_bf_v).T
                #    D_sample_m = bf_sample_m / (1 + bf_sample_m)
                #else:
                #    log("Not centering combined")


                #for gene in temp_genes:
                #    print("D for %s: %s" % (gene, np.mean(D_sample_m[:,self.gene_to_ind[gene]])))

                #for gene in temp_genes:
                #    print("Y for %s: %s" % (gene, np.mean(Y_sample_m[:,self.gene_to_ind[gene]])))

                #for gene in temp_genes:
                #    print("D\t%s\t%s" % (gene, "\t".join([str(x) for x in D_sample_m[:,self.gene_to_ind[gene]]])))
                #    print("Y\t%s\t%s" % (gene, "\t".join([str(x) for x in Y_sample_m[:,self.gene_to_ind[gene]]])))

                #We must normalize Y_sample_m for the compute beta tildes!
                #FIXME: this led to a bug and should be updated to prevent errors in the future
                #TESTING removal of this standardization
                #Y_sample_m = (Y_sample_m.T - np.mean(Y_sample_m, axis=1)).T

                #var(Y) = E[var(Y|S,beta)] + var(E[Y|S,beta])
                #First term can be estimated from the gibbs samples
                #Second term is just Y_cond_var (to a first approximation), or more accurately the term in the integral from gauss
                #Third term is the term we estimate from the Gauss seidel regression
                #So: if we use
                #y_var = np.var(Y_sample_m, axis=1)
                #the term we want, var(Y|S,beta) is being overestimated for gibbs, underestimated for gauss seidel
                #Let's first try Gauss seidel with correction, then try Gibbs with the first approximation (see the difference)
                #This is what is implemented above in Y_var_m -- gives conditional variance of each Y
                #Taking mean of this is our other estimate

                #sample from beta

                #combine X_orig and X_orig missing?

                #TODO: y_corr_sparse needs to be reduced due to y_var (it is larger here than it is above)
                #TODO: if decide calculations depend on chain, then also need to update compute_beta_tildes to return matrix of se_inflation_factors
                y_corr_sparse = None
                if self.y_corr_sparse is not None:

                    log("Adjusting correlation matrix")

                    y_corr_sparse = copy.copy(self.y_corr_sparse)

                    #lower the correlation to account for the 
                    y_corr_sparse = y_corr_sparse.multiply(y_var_orig)

                    #new variances
                    new_y_sd = np.sqrt(np.square(np.mean(priors_for_Y_m, axis=0)) + y_var_orig)[np.newaxis,:]

                    y_corr_sparse = y_corr_sparse.multiply(1/new_y_sd.T)
                    y_corr_sparse = y_corr_sparse.multiply(1/new_y_sd)
                    y_corr_sparse.setdiag(1)

                    y_corr_sparse = y_corr_sparse.tocsc()


                #NOW ONTO GENE SETS

                def __get_gene_set_mask(uncorrected_betas_mean_m, uncorrected_betas_sample_m, p_values_m, sparse_frac=0.01):
                    #if desired, add back in option to set to sample
                    #uncorrected_betas_m = uncorrected_betas_sample_m

                    uncorrected_betas_m = uncorrected_betas_mean_m

                    gene_set_mask_m = uncorrected_betas_m != 0

                    if sparse_frac is not None:
                        #this triggers three things
                        #1. Only gene sets above this threshold are considered for full analysis
                        gene_set_mask_m = np.logical_and(gene_set_mask_m, (np.abs(uncorrected_betas_m).T >= sparse_frac * np.max(np.abs(uncorrected_betas_m), axis=1)).T)
                        #2. The uncorrected values for sampling next iteration are also zeroed out
                        uncorrected_betas_sample_m[~gene_set_mask_m] = 0
                        #3. The mean values (which are added to the estimate) are also zeroed out
                        uncorrected_betas_mean_m[~gene_set_mask_m] = 0

                    if np.sum(gene_set_mask_m) == 0:
                        gene_set_mask_m = p_values_m <= np.min(p_values_m)
                    return gene_set_mask_m


                full_scale_factors_m = np.tile(self.scale_factors, num_chains).reshape((num_chains, len(self.scale_factors)))
                full_mean_shifts_m = np.tile(self.mean_shifts, num_chains).reshape((num_chains, len(self.mean_shifts)))
                full_is_dense_gene_set_m = np.tile(self.is_dense_gene_set, num_chains).reshape((num_chains, len(self.is_dense_gene_set)))
                full_ps_m = None
                if self.ps is not None:
                    full_ps_m = np.tile(self.ps, num_chains).reshape((num_chains, len(self.ps)))
                full_sigma2s_m = None
                if self.sigma2s is not None:
                    full_sigma2s_m = np.tile(self.sigma2s, num_chains).reshape((num_chains, len(self.sigma2s)))

                #we have to keep local replicas here because unsubset does not restore the original order, which would break full_beta_tildes and full_betas

                p_sample_m = copy.copy(Y_sample_m)

                pre_gene_set_filter_mask = None
                full_z_cur_beta_tildes_m = np.zeros(full_betas_m_shape)

                if self.is_logistic:
                    if not gauss_seidel:
                        log("Sampling Ds for logistic", TRACE)
                        p_sample_m = np.zeros(D_sample_m.shape)
                        p_sample_m[np.random.random(D_sample_m.shape) < D_sample_m] = 1

                    else:
                        log("Setting Ds to mean probabilities", TRACE)
                        p_sample_m = D_sample_m

                    if initial_linear_filter:
                        (linear_beta_tildes_m, linear_ses_m, linear_z_scores_m, linear_p_values_m, linear_se_inflation_factors_m) = self._compute_beta_tildes(self.X_orig, Y_sample_m, y_var, self.scale_factors, self.mean_shifts, resid_correlation_matrix=y_corr_sparse)
                        (linear_uncorrected_betas_sample_m, linear_uncorrected_postp_sample_m, linear_uncorrected_betas_mean_m, linear_uncorrected_postp_mean_m) = self._calculate_non_inf_betas(assume_independent=True, initial_p=None, beta_tildes=linear_beta_tildes_m, ses=linear_ses_m, V=None, X_orig=None, scale_factors=full_scale_factors_m, mean_shifts=full_mean_shifts_m, is_dense_gene_set=full_is_dense_gene_set_m, ps=full_ps_m, sigma2s=full_sigma2s_m, return_sample=True, max_num_burn_in=passed_in_max_num_burn_in, max_num_iter=max_num_iter_betas, min_num_iter=min_num_iter_betas, num_chains=num_chains_betas, r_threshold_burn_in=r_threshold_burn_in_betas, use_max_r_for_convergence=use_max_r_for_convergence_betas, max_frac_sem=max_frac_sem_betas, max_allowed_batch_correlation=max_allowed_batch_correlation, gauss_seidel=gauss_seidel_betas, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, debug_gene_sets=self.gene_sets)
                        pre_gene_set_filter_mask_m = __get_gene_set_mask(linear_uncorrected_betas_mean_m, linear_uncorrected_betas_sample_m, linear_p_values_m, sparse_frac=sparse_frac_gibbs)
                        pre_gene_set_filter_mask = np.any(pre_gene_set_filter_mask_m, axis=0)

                        log("Filtered down to %d gene sets using linear pre-filtering" % np.sum(pre_gene_set_filter_mask))
                    else:
                        pre_gene_set_filter_mask = np.full(full_beta_tildes_m.shape[1], True)


                    full_beta_tildes_m = np.zeros(full_betas_m_shape)
                    full_ses_m = np.zeros(full_betas_m_shape)
                    full_z_scores_m = np.zeros(full_betas_m_shape)
                    full_p_values_m = np.zeros(full_betas_m_shape)
                    se_inflation_factors_m = np.zeros(full_betas_m_shape)
                    full_alpha_tildes_m = np.zeros(full_betas_m_shape)
                    diverged_m = np.full(full_betas_m_shape, False)

                    for batch in range(num_stack_batches):
                        begin = batch * stack_batch_size
                        end = (batch + 1) * stack_batch_size
                        if end > num_chains:
                            end = num_chains

                        log("Batch %d: chains %d-%d" % (batch, begin, end), TRACE)
                        num_cur_stack = (end - begin)
                        if num_cur_stack == stack_batch_size:
                            cur_X_hstacked = X_hstacked
                        else:
                            cur_X_hstacked = sparse.hstack([self.X_orig] * num_cur_stack)

                        stack_mask = np.tile(pre_gene_set_filter_mask, num_cur_stack)

                        (full_beta_tildes_m[begin:end,pre_gene_set_filter_mask], full_ses_m[begin:end,pre_gene_set_filter_mask], full_z_scores_m[begin:end,pre_gene_set_filter_mask], full_p_values_m[begin:end,pre_gene_set_filter_mask], init_se_inflation_factors_m, full_alpha_tildes_m[begin:end,pre_gene_set_filter_mask], diverged_m[begin:end,pre_gene_set_filter_mask]) = self._compute_logistic_beta_tildes(self.X_orig[:,pre_gene_set_filter_mask], p_sample_m[begin:end,:], self.scale_factors[pre_gene_set_filter_mask], self.mean_shifts[pre_gene_set_filter_mask], resid_correlation_matrix=y_corr_sparse, X_stacked=cur_X_hstacked[:,stack_mask])

                        full_ses_m[begin:end,~pre_gene_set_filter_mask] = 100
                        full_p_values_m[begin:end,~pre_gene_set_filter_mask] = 1

                        if init_se_inflation_factors_m is not None:
                            se_inflation_factors_m[begin:end,pre_gene_set_filter_mask] = init_se_inflation_factors_m
                        else:
                            se_inflation_factors_m = None

                    #old unconditional one; shouldn't be necessary
                    #else:
                    #    (full_beta_tildes_m, full_ses_m, full_z_scores_m, full_p_values_m, se_inflation_factors_m, full_alpha_tildes_m, diverged_m) = self._compute_logistic_beta_tildes(self.X_orig, p_sample_m, self.scale_factors, self.mean_shifts, resid_correlation_matrix=y_corr_sparse, X_stacked=X_hstacked)
                    #    pre_gene_set_filter_mask = np.full(full_beta_tildes_m.shape[1], True)

                    #calculate whether the sample was an outlier

                    if beta_tilde_outlier_z_threshold is not None:
                        self._record_param("beta_tilde_outlier_z_threshold", beta_tilde_outlier_z_threshold)

                        mean_for_outlier_m = np.zeros(all_sum_z_scores_m.shape)
                        mean2_for_outlier_m = np.zeros(all_sum_z_scores_m.shape)
                        se_for_outlier_m = np.zeros(all_sum_z_scores_m.shape)
                        z_for_outlier_m = np.zeros(all_sum_z_scores_m.shape)
                        num_for_outlier_m = np.zeros(all_sum_z_scores_m.shape)

                        #calculate mean_m as mean ignoring the current chain
                        mean_for_outlier_m = np.sum(all_sum_z_scores_m, axis=0) - all_sum_z_scores_m
                        mean2_for_outlier_m = np.sum(all_sum_z_scores2_m, axis=0) - all_sum_z_scores2_m
                        num_for_outlier_m = np.sum(all_num_sum_m, axis=0) - all_num_sum_m
                        num_for_outlier_non_zero_mask_m = num_for_outlier_m > 0
                        mean_for_outlier_m[num_for_outlier_non_zero_mask_m] = mean_for_outlier_m[num_for_outlier_non_zero_mask_m] / num_for_outlier_m[num_for_outlier_non_zero_mask_m]
                        mean2_for_outlier_m[num_for_outlier_non_zero_mask_m] = mean2_for_outlier_m[num_for_outlier_non_zero_mask_m] / num_for_outlier_m[num_for_outlier_non_zero_mask_m]

                        #TEMP TEMP TEMP
                        #if np.sum(mean2_for_outlier_m[num_for_outlier_non_zero_mask_m] < np.square(mean_for_outlier_m[num_for_outlier_non_zero_mask_m])) > 0:
                        #    print(mean2_for_outlier_m[num_for_outlier_non_zero_mask_m])
                        #    print(mean_for_outlier_m[num_for_outlier_non_zero_mask_m])
                        #    print(num_for_outlier_m[num_for_outlier_non_zero_mask_m])
                        #    print(np.square(mean_for_outlier_m[num_for_outlier_non_zero_mask_m]))

                        #    print(mean2_for_outlier_m[num_for_outlier_non_zero_mask_m])
                        #    print(mean_for_outlier_m[num_for_outlier_non_zero_mask_m])
                        #    print(num_for_outlier_m[num_for_outlier_non_zero_mask_m])
                        #    print(all_sum_z_scores2_m[num_for_outlier_non_zero_mask_m])
                        #    print(all_sum_z_scores_m[num_for_outlier_non_zero_mask_m])

                        #    mask = np.any(mean2_for_outlier_m < np.square(mean_for_outlier_m), axis=0)
                        #    print(mean2_for_outlier_m[:,mask])
                        #    print(mean_for_outlier_m[:,mask])
                        #    print(num_for_outlier_m[:,mask])
                        #    print(all_sum_z_scores2_m[:,mask])
                        #    print(all_sum_z_scores_m[:,mask])

                        #    print(mean2_for_outlier_m[:,mask][:,0])
                        #    print(mean_for_outlier_m[:,mask][:,0])
                        #    print(num_for_outlier_m[:,mask][:,0])
                        #    print(all_sum_z_scores2_m[:,mask][:,0])
                        #    print(all_sum_z_scores_m[:,mask][:,0])


                        se_for_outlier_m[num_for_outlier_non_zero_mask_m] = np.sqrt(mean2_for_outlier_m[num_for_outlier_non_zero_mask_m] - np.square(mean_for_outlier_m[num_for_outlier_non_zero_mask_m]))

                        se_for_outlier_mask_m = se_for_outlier_m > 0
                        full_z_cur_beta_tildes_m[se_for_outlier_mask_m] = (full_z_scores_m[se_for_outlier_mask_m] - mean_for_outlier_m[se_for_outlier_mask_m]) / se_for_outlier_m[se_for_outlier_mask_m]
                        outlier_mask_m = full_z_cur_beta_tildes_m > beta_tilde_outlier_z_threshold

                        if np.sum(outlier_mask_m) > 0:
                            log("Detected %d outlier gene sets: %s" % (np.sum(outlier_mask_m), ",".join([self.gene_sets[i] for i in np.where(np.any(outlier_mask_m, axis=0))[0]])),DEBUG)

                            outlier_control = False
                            if outlier_control:
                                #inflate them
                                #full_beta_tildes_m[outlier_mask_m] / full_ses_m[outlier_mask_m] - mean_for_outlier_m[outlier_mask_m] = beta_tilde_outlier_z_threshold * se_for_outlier_m[outlier_mask_m]
                                #full_beta_tildes_m[outlier_mask_m] / full_ses_m[outlier_mask_m] = mean_for_outlier_m[outlier_mask_m] + beta_tilde_outlier_z_threshold * se_for_outlier_m[outlier_mask_m]
                                #full_beta_tildes_m[outlier_mask_m] = full_ses_m[outlier_mask_m] * (mean_for_outlier_m[outlier_mask_m] + beta_tilde_outlier_z_threshold * se_for_outlier_m[outlier_mask_m])

                                new_ses_m = np.abs(full_beta_tildes_m[outlier_mask_m] / (mean_for_outlier_m[outlier_mask_m] + beta_tilde_outlier_z_threshold * se_for_outlier_m[outlier_mask_m]))
                                print("Inflated ses from %.3g - %.3g" % (np.min(new_ses_m / full_ses_m[outlier_mask_m]), np.max(new_ses_m / full_ses_m[outlier_mask_m])))

                                full_ses_m[outlier_mask_m] = new_ses_m
                                full_z_scores_m[outlier_mask_m] = full_beta_tildes_m[outlier_mask_m] / new_ses_m
                                full_p_values_m[outlier_mask_m] = 2*scipy.stats.norm.cdf(-np.abs(full_z_scores_m[outlier_mask_m]))

                            else:

                                #first check if we need to reset entire chains
                                num_outliers = np.sum(outlier_mask_m, axis=1)
                                frac_outliers = num_outliers / outlier_mask_m.shape[1]
                                chain_outlier_frac_threshold = 0.0005
                                outlier_chains = frac_outliers > chain_outlier_frac_threshold
                                for outlier_chain in np.where(outlier_chains)[0]:
                                    log("Detected entire chain %d as an outlier since it had %d (%.4g fraction) outliers" % (outlier_chain+1,num_outliers[outlier_chain], frac_outliers[outlier_chain]), DEBUG)
                                    if np.sum(~outlier_chains) > 0:
                                        replacement_chain = np.random.choice(np.where(~outlier_chains)[0], size=1)
                                        for matrix in [full_beta_tildes_m, full_ses_m, full_z_scores_m, full_p_values_m, se_inflation_factors_m, full_alpha_tildes_m, diverged_m]:
                                            if matrix is not None:
                                                matrix[outlier_chain,:] = matrix[replacement_chain,:]
                                        outlier_mask_m[outlier_chain,:] = False

                                    else:
                                        log("Everything was an outlier chain so doing nothing", DEBUG)

                                outlier_gene_sets = np.any(outlier_mask_m, axis=0)
                                for outlier_gene_set in np.where(outlier_gene_sets)[0]:
                                    non_outliers = ~outlier_mask_m[:,outlier_gene_set]
                                    if np.sum(non_outliers) > 0:
                                        log("Resetting %s chain %s; beta_tilde=%s and z=%s" % (self.gene_sets[outlier_gene_set], np.where(outlier_mask_m[:,outlier_gene_set])[0] + 1, full_beta_tildes_m[outlier_mask_m[:,outlier_gene_set],outlier_gene_set], full_z_cur_beta_tildes_m[outlier_mask_m[:,outlier_gene_set],outlier_gene_set]),DEBUG)

                                        replacement_chains = np.random.choice(np.where(non_outliers)[0], size=np.sum(outlier_mask_m[:,outlier_gene_set]))
                                        for matrix in [full_beta_tildes_m, full_ses_m, full_z_scores_m, full_p_values_m, se_inflation_factors_m, full_alpha_tildes_m, diverged_m]:
                                            if matrix is not None:
                                                matrix[outlier_mask_m[:,outlier_gene_set],outlier_gene_set] = matrix[replacement_chains,outlier_gene_set]

                                        #now make the Z threshold more lenient
                                        #beta_tilde_outlier_z_threshold[outlier_gene_set] = -scipy.stats.norm.ppf(scipy.stats.norm.cdf(-np.abs(beta_tilde_outlier_z_threshold[outlier_gene_set])) / 10)
                                        #log("New threshold is z=%.4g" % beta_tilde_outlier_z_threshold[outlier_gene_set], TRACE)

                                    else:
                                        log("Everything was an outlier for gene set %s so doing nothing" % (self.gene_sets[outlier_gene_set]), DEBUG)

                    else:
                        full_z_cur_beta_tildes_m = np.zeros(full_beta_tildes_m.shape)


                    if np.sum(diverged_m) > 0:
                        for c in range(diverged_m.shape[0]):
                            if np.sum(diverged_m[c,:] > 0):
                                for p in np.nditer(np.where(diverged_m[c,:])):
                                    log("Chain %d: gene set %s diverged" % (c+1, self.gene_sets[p]), DEBUG)
                else:
                    (full_beta_tildes_m, full_ses_m, full_z_scores_m, full_p_values_m, se_inflation_factors_m) = self._compute_beta_tildes(self.X_orig, Y_sample_m, y_var, self.scale_factors, self.mean_shifts, resid_correlation_matrix=y_corr_sparse)

                if correct_betas:
                    (full_beta_tildes_m, full_ses_m, full_z_scores_m, full_p_values_m, se_inflation_factors_m) = self._correct_beta_tildes(full_beta_tildes_m, full_ses_m, se_inflation_factors_m, self.total_qc_metrics, self.mean_qc_metrics, add_missing=True, fit=False)


                #now write the gene stats trace

                if gene_stats_trace_out is not None:
                    log("Writing gene stats trace", TRACE)
                    for chain_num in range(num_chains):
                        for i in range(len(self.genes)):
                            if only_genes is None or self.genes[i] in only_genes:
                                gene_stats_trace_fh.write("%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n" % (iteration_num+1, chain_num+1, self.genes[i], priors_for_Y_m[chain_num,i], Y_sample_m[chain_num,i], log_bf_m[chain_num,i], p_sample_m[chain_num,i], priors_percentage_max_for_Y_m[chain_num,i], priors_adjustment_for_Y_m[chain_num,i]))
                        #if self.genes_missing is not None:
                        #    for i in range(len(self.genes_missing)):
                        #        if only_genes is None or self.genes[i] in only_genes:
                        #            gene_stats_trace_fh.write("%d\t%d\t%s\t%.4g\t%s\t%s\t%s\n" % (iteration_num+1, chain_num+1, self.genes_missing[i], priors_missing_sample_m[chain_num,i], "NA", "NA", "NA"))

                    gene_stats_trace_fh.flush()

                (uncorrected_betas_sample_m, uncorrected_postp_sample_m, uncorrected_betas_mean_m, uncorrected_postp_mean_m) = self._calculate_non_inf_betas(assume_independent=True, initial_p=None, beta_tildes=full_beta_tildes_m, ses=full_ses_m, V=None, X_orig=None, scale_factors=full_scale_factors_m, mean_shifts=full_mean_shifts_m, is_dense_gene_set=full_is_dense_gene_set_m, ps=full_ps_m, sigma2s=full_sigma2s_m, return_sample=True, max_num_burn_in=passed_in_max_num_burn_in, max_num_iter=max_num_iter_betas, min_num_iter=min_num_iter_betas, num_chains=num_chains_betas, r_threshold_burn_in=r_threshold_burn_in_betas, use_max_r_for_convergence=use_max_r_for_convergence_betas, max_frac_sem=max_frac_sem_betas, max_allowed_batch_correlation=max_allowed_batch_correlation, gauss_seidel=gauss_seidel_betas, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, debug_gene_sets=self.gene_sets)

                #initial values to use
                #we will overwrite these with the corrected betas
                #but, if we decide to filter them out due to sparsity, we'll persist with the (small) uncorrected values
                default_betas_sample_m = copy.copy(uncorrected_betas_sample_m)
                default_postp_sample_m = copy.copy(uncorrected_postp_sample_m)
                default_betas_mean_m = copy.copy(uncorrected_betas_mean_m)
                default_postp_mean_m = copy.copy(uncorrected_postp_mean_m)

                #filter back down based on max p?
                gene_set_mask_m = np.full(full_p_values_m.shape, True)

                #no longer an option
                #if self.max_gene_set_p is not None and use_orig_gene_set_p:
                #    gene_set_mask_m = np.tile(self.p_values_orig <= self.max_gene_set_p, num_chains).reshape((num_chains, len(gene_set_mask)))

                gene_set_mask_m = __get_gene_set_mask(uncorrected_betas_mean_m, uncorrected_betas_sample_m, full_p_values_m, sparse_frac=sparse_frac_gibbs)
                any_gene_set_mask = np.any(gene_set_mask_m, axis=0)
                if pre_filter_batch_size is not None and np.sum(any_gene_set_mask) > pre_filter_batch_size:
                    num_batches = self._get_num_X_blocks(self.X_orig[:,any_gene_set_mask], batch_size=pre_filter_small_batch_size)
                    if num_batches > 1:
                        #try to run with small batches to see if we can zero out more
                        gene_set_block_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_orig[:,any_gene_set_mask], mean_shifts=self.mean_shifts[any_gene_set_mask], scale_factors=self.scale_factors[any_gene_set_mask], find_correlated_instead=pre_filter_small_batch_size)

                        if len(gene_set_block_masks) > 0:
                            if np.sum(gene_set_block_masks[-1]) == 1 and len(gene_set_block_masks) > 1:
                                #merge singletons at the end into the one before
                                gene_set_block_masks[-2][gene_set_block_masks[-1]] = True
                                gene_set_block_masks = gene_set_block_masks[:-1]
                            if len(gene_set_block_masks) > 1 and np.sum(gene_set_block_masks[0]) > 1:
                                #find map of indices to original indices
                                V_data = []
                                V_rows = []
                                V_cols = []
                                for gene_set_block_mask in gene_set_block_masks:
                                    V_block = self._calculate_V_internal(self.X_orig[:,any_gene_set_mask][:,gene_set_block_mask], self.y_corr_cholesky, self.mean_shifts[any_gene_set_mask][gene_set_block_mask], self.scale_factors[any_gene_set_mask][gene_set_block_mask])
                                    orig_indices = np.where(gene_set_block_mask)[0]
                                    V_rows += list(np.repeat(orig_indices, V_block.shape[0]))
                                    V_cols += list(np.tile(orig_indices, V_block.shape[0]))
                                    V_data += list(V_block.ravel())

                                V_sparse = sparse.csc_matrix((V_data, (V_rows, V_cols)), shape=(np.sum(any_gene_set_mask), np.sum(any_gene_set_mask)))
                                log("Running %d blocks to check for zeros..." % len(gene_set_block_masks),DEBUG)
                                (half_corrected_betas_sample_m, half_corrected_postp_sample_m, half_corrected_betas_mean_m, half_corrected_postp_mean_m) = self._calculate_non_inf_betas(initial_p=None, beta_tildes=full_beta_tildes_m[:,any_gene_set_mask], ses=full_ses_m[:,any_gene_set_mask], V=V_sparse, X_orig=None, scale_factors=full_scale_factors_m[:,any_gene_set_mask], mean_shifts=full_mean_shifts_m[:,any_gene_set_mask], is_dense_gene_set=full_is_dense_gene_set_m[:,any_gene_set_mask], ps=full_ps_m[:,any_gene_set_mask], sigma2s=full_sigma2s_m[:,any_gene_set_mask], return_sample=True, max_num_burn_in=passed_in_max_num_burn_in, max_num_iter=max_num_iter_betas, min_num_iter=min_num_iter_betas, num_chains=num_chains_betas, r_threshold_burn_in=r_threshold_burn_in_betas, use_max_r_for_convergence=use_max_r_for_convergence_betas, max_frac_sem=max_frac_sem_betas, max_allowed_batch_correlation=max_allowed_batch_correlation, gauss_seidel=gauss_seidel_betas, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas)

                                add_zero_mask_m = ~(__get_gene_set_mask(half_corrected_betas_mean_m, half_corrected_betas_sample_m, full_p_values_m, sparse_frac=sparse_frac_gibbs))

                                if np.any(add_zero_mask_m):
                                    #need to convert these to the original gene sets
                                    map_to_full = np.where(any_gene_set_mask)[0]
                                    #get rows and then columns in subsetted
                                    set_to_zero_full = np.where(add_zero_mask_m)
                                    #map columns in subsetted to original
                                    set_to_zero_full = (set_to_zero_full[0], map_to_full[set_to_zero_full[1]])
                                    orig_zero = np.sum(np.any(gene_set_mask_m, axis=0))
                                    gene_set_mask_m[set_to_zero_full] = False
                                    new_zero = np.sum(np.any(gene_set_mask_m, axis=0))
                                    log("Found %d additional zero gene sets" % (orig_zero - new_zero),DEBUG)

                                    #need to update uncorrected ones too
                                    default_betas_sample_m[set_to_zero_full] = half_corrected_betas_sample_m[add_zero_mask_m]
                                    default_postp_sample_m[set_to_zero_full] = half_corrected_postp_sample_m[add_zero_mask_m]
                                    default_betas_mean_m[set_to_zero_full] = half_corrected_betas_mean_m[add_zero_mask_m]
                                    default_postp_mean_m[set_to_zero_full] = half_corrected_postp_mean_m[add_zero_mask_m]

                num_non_missing_v = np.sum(gene_set_mask_m, axis=1)
                max_num_non_missing = np.max(num_non_missing_v)
                max_num_non_missing_idx = np.argmax(num_non_missing_v)
                log("Max number of gene sets to keep across all chains is %d" % (max_num_non_missing))
                log("Keeping %d gene sets that had non-zero uncorected betas" % (sum(np.any(gene_set_mask_m, axis=0))))
                for chain_num in range(gene_set_mask_m.shape[0]):
                    if num_non_missing_v[chain_num] < max_num_non_missing:
                        cur_num = 0
                        #add in gene sets that are in the max one to ensure things are square
                        for index in np.nonzero(gene_set_mask_m[max_num_non_missing_idx,:] & ~gene_set_mask_m[chain_num,:])[0]:
                            assert(gene_set_mask_m[chain_num,index] == False)
                            gene_set_mask_m[chain_num,index] = True
                            cur_num += 1
                            if cur_num >= max_num_non_missing - num_non_missing_v[chain_num]:
                                break

                #log("Keeping %d gene sets that passed threshold of p<%.3g" % (sum(gene_set_mask), self.max_gene_set_p))


                #Now call betas in batches
                #we are doing this only for memory reasons -- we have to create a V matrix for each chain
                #it is furthermore faster to create a V once for all gene sets across all chains, and then subset it for each chain, which
                #further increases memory
                #so, the strategy is to batch the chains, for each batch calculate a V for the superset of all gene sets, and then subset it

                #betas
                if options.debug_zero_sparse:
                    full_betas_mean_m = copy.copy(default_betas_mean_m)
                    full_betas_sample_m = copy.copy(default_betas_sample_m)
                    full_postp_mean_m = copy.copy(default_postp_mean_m)
                    full_postp_sample_m = copy.copy(default_postp_sample_m)
                else:
                    full_betas_mean_m = np.zeros(default_betas_mean_m.shape)
                    full_betas_sample_m = np.zeros(default_betas_sample_m.shape)
                    full_postp_mean_m = np.zeros(default_postp_mean_m.shape)
                    full_postp_sample_m = np.zeros(default_postp_sample_m.shape)

                num_calculations = int(np.ceil(num_chains / num_batches_parallel))
                #we will default all to the uncorrected sample, and then replace those below that are non-zero
                for calc in range(num_calculations):
                    begin = calc * num_batches_parallel
                    end = (calc + 1) * num_batches_parallel
                    if end > num_chains:
                        end = num_chains

                    #get the include mask; any batch has p <= threshold
                    cur_gene_set_mask = np.any(gene_set_mask_m[begin:end,:], axis=0)
                    num_gene_set_mask = np.sum(cur_gene_set_mask)
                    max_num_gene_set_mask = np.max(np.sum(gene_set_mask_m, axis=1))

                    #construct the V matrix
                    V_superset = self._calculate_V_internal(self.X_orig[:,cur_gene_set_mask], self.y_corr_cholesky, self.mean_shifts[cur_gene_set_mask], self.scale_factors[cur_gene_set_mask])


                    #empirically it is faster to do one V if the total is less than 5x the max
                    run_one_V = num_gene_set_mask < 5 * max_num_gene_set_mask

                    if run_one_V:
                        num_non_missing = np.sum(cur_gene_set_mask)
                    else:
                        num_non_missing = np.max(np.sum(gene_set_mask_m, axis=1))

                    num_missing = gene_set_mask_m.shape[1] - num_non_missing

                    #fraction_non_missing = float(num_non_missing) / gene_set_mask_m.shape[1]
                    #missing_scale_factor = self._get_fraction_non_missing() / fraction_non_missing

                    #if missing_scale_factor > 1 / self.p:
                    #    #threshold this here. otherwise set_p will cap p but set_sigma won't cap sigma
                    #    missing_scale_factor = 1 / self.p

                    if run_one_V:
                        beta_tildes_m = full_beta_tildes_m[begin:end,cur_gene_set_mask]
                        ses_m = full_ses_m[begin:end,cur_gene_set_mask]
                        V_m=V_superset
                        scale_factors_m = self.scale_factors[cur_gene_set_mask]
                        mean_shifts_m = self.mean_shifts[cur_gene_set_mask]
                        is_dense_gene_set_m = self.is_dense_gene_set[cur_gene_set_mask]
                        ps_m = None
                        if self.ps is not None:
                            ps_m = self.ps[cur_gene_set_mask]
                        sigma2s_m = None
                        if self.sigma2s is not None:
                            sigma2s_m = self.sigma2s[cur_gene_set_mask]


                        #beta_tildes_missing_m = full_beta_tildes_m[begin:end,~cur_gene_set_mask]
                        #ses_missing_m = full_ses_m[begin:end,~cur_gene_set_mask]
                        #scale_factors_missing_m = self.scale_factors[~cur_gene_set_mask]

                    else:
                        non_missing_matrix_shape = (num_chains, num_non_missing)
                        beta_tildes_m = full_beta_tildes_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
                        ses_m = full_ses_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
                        scale_factors_m = full_scale_factors_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
                        mean_shifts_m = full_mean_shifts_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
                        is_dense_gene_set_m = full_is_dense_gene_set_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
                        ps_m = None
                        if full_ps_m is not None:
                            ps_m = full_ps_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]
                        sigma2s_m = None
                        if full_sigma2s_m is not None:
                            sigma2s_m = full_sigma2s_m[gene_set_mask_m].reshape(non_missing_matrix_shape)[begin:end,:]

                        V_m = np.zeros((end-begin, beta_tildes_m.shape[1], beta_tildes_m.shape[1]))
                        for i,j in zip(range(begin, end),range(end-begin)):
                            #gene_set_mask_m[i,:] is the current batch mask, with dimensions num_gene_sets
                            #to index into V_superset, we need to subset it down to cur_gene_set_mask
                            gene_set_mask_subset = gene_set_mask_m[i,cur_gene_set_mask]
                            V_m[j,:,:] = V_superset[gene_set_mask_subset,:][:,gene_set_mask_subset]

                        #missing_matrix_shape = (num_chains, num_missing)
                        #beta_tildes_missing_m = full_beta_tildes_m[~gene_set_mask_m].reshape(missing_matrix_shape)[begin:end,:]
                        #ses_missing_m = full_ses_m[~gene_set_mask_m].reshape(missing_matrix_shape)[begin:end,:]
                        #scale_factors_missing_m = full_scale_factors_m[~gene_set_mask_m].reshape(missing_matrix_shape)[begin:end,:]

                    (cur_betas_sample_m, cur_postp_sample_m, cur_betas_mean_m, cur_postp_mean_m) = self._calculate_non_inf_betas(initial_p=None, beta_tildes=beta_tildes_m, ses=ses_m, V=V_m, scale_factors=scale_factors_m, mean_shifts=mean_shifts_m, is_dense_gene_set=is_dense_gene_set_m, ps=ps_m, sigma2s=sigma2s_m, return_sample=True, max_num_burn_in=passed_in_max_num_burn_in, max_num_iter=max_num_iter_betas, min_num_iter=min_num_iter_betas, num_chains=num_chains_betas, r_threshold_burn_in=r_threshold_burn_in_betas, use_max_r_for_convergence=use_max_r_for_convergence_betas, max_frac_sem=max_frac_sem_betas, max_allowed_batch_correlation=max_allowed_batch_correlation, gauss_seidel=gauss_seidel_betas, update_hyper_sigma=False, update_hyper_p=False, num_missing_gene_sets=num_missing, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, betas_trace_out=betas_trace_out, debug_gene_sets=[self.gene_sets[i] for i in range(len(self.gene_sets)) if gene_set_mask_m[0,i]])

                    #store the values with zeros appended in order to add to sum_betas_m below
                    if run_one_V:
                        full_betas_sample_m[begin:end,cur_gene_set_mask] = cur_betas_sample_m
                        full_postp_sample_m[begin:end,cur_gene_set_mask] = cur_postp_sample_m
                        full_betas_mean_m[begin:end,cur_gene_set_mask] = cur_betas_mean_m
                        full_postp_mean_m[begin:end,cur_gene_set_mask] = cur_postp_mean_m

                        #handy option for debugging
                        print_overlapping = None
                        if print_overlapping is not None:
                            gene_sets_run = [self.gene_sets[i] for i in range(len(self.gene_sets)) if cur_gene_set_mask[i]]
                            gene_set_to_ind = self._construct_map_to_ind(gene_sets_run)
                            for gene_set in print_overlapping:
                                if gene_set in gene_set_to_ind:
                                    print("For gene set %s" % (gene_set))
                                    ind = gene_set_to_ind[gene_set]
                                    values = V_m[ind,:] * (cur_betas_mean_m if use_mean_betas else cur_betas_sample_m)
                                    indices = np.argsort(values, axis=1)
                                    for chain in range(values.shape[0]):
                                        print("Chain %d (uncorrected beta=%.4g, corrected beta=%.4g)" % (chain, uncorrected_betas_mean_m[chain,self.gene_set_to_ind[gene_set]], (cur_betas_mean_m[chain,ind] if use_mean_betas else cur_betas_sample_m[chain,ind])))
                                        for i in indices[chain,::-1]:
                                            if values[chain,i] == 0:
                                                break
                                            print("%s, V=%.4g, beta=%.4g, prod=%.4g" % (gene_sets_run[i], V_m[ind,i], (cur_betas_mean_m[chain,i] if use_mean_betas else cur_betas_sample_m[chain,i]), values[chain,i]))

                    else:
                        #store the values with zeros appended in order to add to sum_betas_m below
                        full_betas_sample_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_betas_sample_m.ravel()
                        full_postp_sample_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_postp_sample_m.ravel()
                        full_betas_mean_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_betas_mean_m.ravel()
                        full_postp_mean_m[begin:end,:][gene_set_mask_m[begin:end,:]] = cur_postp_mean_m.ravel()

                    #see how many are set to zero
                    #set_to_zero_v += np.mean(np.logical_and(full_betas_sample_m == 0, uncorrected_betas_sample_m == 0).reshape(full_betas_sample_m.shape), axis=0)
                    #avg_full_betas_sample_v += np.mean(full_betas_sample_m, axis=0)
                    #avg_full_postp_sample_v += np.mean(full_postp_sample_m, axis=0)

                #now restore the p and sigma
                #self.set_sigma(orig_sigma2, self.sigma_power, sigma2_osc=self.sigma2_osc)
                #self.set_p(orig_p)

                #since the betas are a sample (rather than mean), we can sample from priors by just multiplying this sample

                #this is the (log) prior odds relative to the background_log_bf

                def __calc_priors(_X, _betas):
                    return np.array(_X.dot((_betas / self.scale_factors).T) - np.sum(self.mean_shifts * _betas / self.scale_factors, axis=1).T).T                
                priors_sample_m = __calc_priors(self.X_orig, full_betas_sample_m)
                priors_mean_m = __calc_priors(self.X_orig, full_betas_mean_m)
                if self.genes_missing is not None:
                    priors_missing_sample_m = __calc_priors(self.X_orig_missing_genes, full_betas_sample_m)
                    priors_missing_mean_m = __calc_priors(self.X_orig_missing_genes, full_betas_mean_m)

                def __adjust_max_prior_contribution(_X, _betas, _priors_m):
                    priors_max_contribution_m = np.zeros(_priors_m.shape)
                    #don't think it is possible to vectorize this due to sparse matrices maxxing out at two dimensions
                    for chain in range(priors_max_contribution_m.shape[0]):
                        priors_max_contribution_m[chain,:] = _X.multiply(np.abs(_betas[chain,:]) / self.scale_factors).max(axis=1).todense().A1
                    priors_naive_m = np.array(_X.dot((np.abs(_betas) / self.scale_factors).T)).T

                    priors_percentage_max_m = np.ones(_priors_m.shape)
                    non_zero_priors_mask = priors_naive_m != 0
                    priors_percentage_max_m[non_zero_priors_mask] = priors_max_contribution_m[non_zero_priors_mask] / priors_naive_m[non_zero_priors_mask]

                    #TEMP STUFF FOR PRINTING
                    #non_zero_priors_mask = np.logical_and(_priors_m > 0, priors_percentage_max_m < 1)
                    #sample_index = np.where(non_zero_priors_mask)
                    #if len(sample_index[0]) > 0:
                    #    sample_chain = sample_index[0][0]
                    #    sample_index = sample_index[1][0]
                    #    print("Sample chain:",sample_index)
                    #    print("Sample index:",sample_index)
                    #    print("Sample priors:",_priors_m[sample_chain,sample_index])
                    #    print(priors_percentage_max_m[sample_chain,sample_index])
                    #    print(sparse.csc_matrix(_X)[sample_index,:])
                    #    print(_betas[sample_chain,:])
                    #    bail("")
                    #ind = self.gene_to_ind['ZNF275']
                    #c = 0
                    #m = np.where(priors_percentage_max_m > 100)
                    #if len(m[0]) > 0:
                    #    c = m[0][0]
                    #    ind = m[1][0]
                    #    print("CHECKING")
                    #    print(priors_naive_m[c,ind])
                    #    print(sparse.csr_matrix(_X)[ind,:])
                    #    print(sparse.csr_matrix(_X.multiply(_betas[chain,:]))[ind,:])
                    #    print(priors_max_contribution_m[c,ind],priors_naive_m[c,ind],priors_max_contribution_m[c,ind]/priors_naive_m[c,ind])
                    #    print(priors_percentage_max_m[c,ind])
                    #END TEMP STUFF FOR PRINTING


                    priors_percentage_max_m[priors_percentage_max_m < 0] = 0
                    priors_percentage_max_m[priors_percentage_max_m > 1] = 1

                    #for each prior, sample one percentage
                    new_priors_percentage_max_m = copy.copy(priors_percentage_max_m)
                    max_allowed_percentage = 0.95
                    for chain in range(priors_max_contribution_m.shape[0]):
                        sample_mask = priors_percentage_max_m[chain,:] < max_allowed_percentage
                        num_allowed = np.sum(sample_mask)
                        if num_allowed > 0:
                            new_columns = np.random.randint(num_allowed, size=_priors_m.shape[1])
                            new_priors_percentage_max_m[chain,:] = priors_percentage_max_m[chain,sample_mask][new_columns]

                    #don't update those that were below the threshold or would increase
                    no_update_priors_mask = np.logical_or(priors_percentage_max_m < new_priors_percentage_max_m, priors_percentage_max_m < max_allowed_percentage)
                    new_priors_percentage_max_m[no_update_priors_mask] = priors_percentage_max_m[no_update_priors_mask]

                    #print("FIXME: TEMPORARILY DISABLING ADJUSTMENT")
                    #return (np.zeros(_priors_m.shape), priors_percentage_max_m)

                    priors_adjustment_m = -priors_max_contribution_m + new_priors_percentage_max_m * priors_naive_m
                    return (priors_adjustment_m, priors_percentage_max_m)

                #option not used right now
                #if see one gene set dominating top, consider adding back
                penalize_priors = False
                if penalize_priors:
                    self._record_param("penalize_priors", True)

                    #first calculate
                    priors_to_adjust_sample_m = priors_sample_m
                    priors_to_adjust_mean_m = priors_mean_m
                    if self.genes_missing is not None:
                        priors_to_adjust_sample_m = np.hstack((priors_sample_m, priors_missing_sample_m))
                        priors_to_adjust_mean_m = np.hstack((priors_mean_m, priors_missing_mean_m))

                    (priors_adjustment_sample_m, priors_percentage_max_sample_m) = __adjust_max_prior_contribution(X_all, full_betas_sample_m, priors_to_adjust_sample_m)
                    (priors_adjustment_mean_m, priors_percentage_max_mean_m) = __adjust_max_prior_contribution(X_all, full_betas_mean_m, priors_to_adjust_mean_m)

                    if self.genes_missing is not None:
                        priors_missing_sample_m += priors_adjustment_sample_m[:,-priors_missing_sample_m.shape[1]:]
                        priors_missing_mean_m += priors_adjustment_mean_m[:,-priors_missing_mean_m.shape[1]:]
                        priors_sample_m += priors_adjustment_sample_m[:,:priors_adjustment_sample_m.shape[1]-priors_missing_sample_m.shape[1]]
                        priors_mean_m += priors_adjustment_mean_m[:,:priors_adjustment_mean_m.shape[1]-priors_missing_mean_m.shape[1]]
                    else:
                        priors_sample_m += priors_adjustment_sample_m
                        priors_mean_m += priors_adjustment_mean_m

                if self.huge_signal_bfs is not None and update_huge_scores:
                    #Now update the BFs is we have huge scores
                    log("Updating HuGE scores")
                    if self.huge_signal_bfs is not None:
                        rel_prior_log_bf = priors_for_Y_m

                        (log_bf_m, log_bf_uncorrected_m, absent_genes, absent_log_bf) = self._distill_huge_signal_bfs(self.huge_signal_bfs, self.huge_signal_posteriors_for_regression, self.huge_signal_sum_gene_cond_probabilities, self.huge_signal_mean_gene_pos, self.huge_signal_max_closest_gene_prob, self.huge_cap_region_posterior, self.huge_scale_region_posterior, self.huge_phantom_region_posterior, self.huge_allow_evidence_of_absence, self.huge_correct_huge, self.huge_gene_covariates, self.huge_gene_covariates_mask, self.huge_gene_covariates_mat_inv, self.genes, rel_prior_log_bf=rel_prior_log_bf + (self.Y_exomes if self.Y_exomes is not None else 0) + (self.Y_positive_controls if self.Y_positive_controls is not None else 0))

                        if compute_Y_raw:
                            (log_bf_raw_m, log_bf_uncorrected_raw_m, absent_genes_raw, absent_log_bf_raw) = self._distill_huge_signal_bfs(self.huge_signal_bfs, self.huge_signal_posteriors, self.huge_signal_sum_gene_cond_probabilities, self.huge_signal_mean_gene_pos, self.huge_signal_max_closest_gene_prob, self.huge_cap_region_posterior, self.huge_scale_region_posterior, self.huge_phantom_region_posterior, self.huge_allow_evidence_of_absence, self.huge_correct_huge, self.huge_gene_covariates, self.huge_gene_covariates_mask, self.huge_gene_covariates_mat_inv, self.genes, rel_prior_log_bf=rel_prior_log_bf + (self.Y_exomes if self.Y_exomes is not None else 0) + (self.Y_positive_controls if self.Y_positive_controls is not None else 0))
                        else:
                            log_bf_raw_m = copy.copy(log_bf_m)

                        if self.Y_exomes is not None:
                            #add in the Y_exomes
                            #it was used in distill just to fine map the GWAS associations; it was then subtracted out
                            #the other component of the rel_prior_log_bf (priors) will be added back in next iteration
                            log_bf_m += self.Y_exomes
                            log_bf_raw_m += self.Y_exomes

                        if self.Y_positive_controls is not None:
                            #add in the Y_positive_controls
                            #it was used in distill just to fine map the GWAS associations; it was then subtracted out
                            #the other component of the rel_prior_log_bf (priors) will be added back in next iteration
                            log_bf_m += self.Y_positive_controls
                            log_bf_raw_m += self.Y_positive_controls

                        if len(absent_genes) > 0:
                            bail("Error: huge_signal_bfs was incorrectly set and contains extra genes")

                #if center_combined:
                #    priors_sample_total_m = np.hstack((priors_sample_m, priors_missing_sample_m))
                #    total_mean_v = np.mean(priors_sample_total_m, axis=1)
                #    priors_sample_m = (priors_sample_m.T - total_mean_v).T
                #    priors_missing_sample_m = (priors_missing_sample_m.T - total_mean_v).T
                #    priors_mean_m = (priors_mean_m.T - total_mean_v).T
                #    priors_missing_mean_m = (priors_missing_mean_m.T - total_mean_v).T

                #if self.is_logistic:
                #    print("Alpha tildes: ",np.mean(full_alpha_tildes_m, axis=1) - self.background_log_bf)
                #    print("Adjustment factor: ",np.mean(full_alpha_tildes_m, axis=1) - self.background_log_bf)

                #    priors_sample_m = (priors_sample_m.T + np.mean(full_alpha_tildes_m, axis=1) - self.background_log_bf).T
                #    priors_missing_sample_m = (priors_missing_sample_m.T + np.mean(full_alpha_tildes_m, axis=1) - self.background_log_bf).T

                #do the regression
                total_priors_m = np.hstack((priors_sample_m, priors_missing_sample_m))
                gene_N = self.get_gene_N()
                gene_N_missing = self.get_gene_N(get_missing=True)

                all_gene_N = gene_N
                if self.genes_missing is not None:
                    assert(gene_N_missing is not None)
                    all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

                priors_slope = total_priors_m.dot(all_gene_N) / (total_priors_m.shape[1] * np.var(all_gene_N))
                #no intercept since we just standardized above

                log("Adjusting priors with slopes ranging from %.4g-%.4g" % (np.min(priors_slope), np.max(priors_slope)), TRACE)
                priors_sample_m = priors_sample_m - np.outer(priors_slope, gene_N)
                priors_mean_m = priors_mean_m - np.outer(priors_slope, gene_N)

                if self.genes_missing is not None:
                    priors_missing_sample_m = priors_missing_sample_m - np.outer(priors_slope, gene_N_missing)
                    priors_missing_mean_m = priors_missing_mean_m - np.outer(priors_slope, gene_N_missing)

                priors_for_Y_m = priors_sample_m
                priors_percentage_max_for_Y_m = priors_percentage_max_sample_m
                priors_adjustment_for_Y_m = priors_adjustment_sample_m
                if use_mean_betas:
                    priors_for_Y_m = priors_mean_m                
                    priors_percentage_max_for_Y_m = priors_percentage_max_mean_m
                    priors_adjustment_for_Y_m = priors_adjustment_mean_m

                #print("Priors")
                #for gene in temp_genes:
                #    print("%s: %s" % (gene, np.mean(priors_sample_m[:,self.gene_to_ind[gene]], axis=0)))

                #print("Mean priors:", np.mean(priors_sample_m[:,self.gene_N > 0], axis=1))
                #print("Mean priors empty:", np.mean(priors_sample_m[:,self.gene_N == 0], axis=1))
                #print("Mean combined:", np.mean(log_po_sample_m[:,self.gene_N > 0], axis=1))
                #print("Mean combined empty:", np.mean(log_po_sample_m[:,self.gene_N == 0], axis=1))
                #print("Mean D:", np.mean(D_sample_m[:,self.gene_N > 0], axis=1))
                #print("Mean D empty:", np.mean(D_sample_m[:,self.gene_N == 0], axis=1))

                #only add non-outliers to mean/sd
                #non_outlier_mask = np.full(sum_z_scores2_m.shape, True)
                #non_outlier_mask[num_sum_outlier_m > 10] = np.abs(full_z_scores_m[num_sum_outlier_m > 10]) < -scipy.stats.norm.ppf(0.5 * 0.05 / (num_sum_outlier_m[num_sum_outlier_m > 10] * num_chains_betas))
                #if pre_gene_set_filter_mask is not None:
                #    non_outlier_mask[:,~pre_gene_set_filter_mask] = False
                #sum_z_scores2_m[non_outlier_mask] = np.add(sum_z_scores2_m[non_outlier_mask], np.power(full_z_scores_m[non_outlier_mask], 2))
                #sum_z_scores_m[non_outlier_mask] = np.add(sum_z_scores_m[non_outlier_mask], full_z_scores_m[non_outlier_mask])
                #num_sum_outlier_m[non_outlier_mask] += 1

                all_sum_betas_m = np.add(all_sum_betas_m, full_betas_mean_m)
                all_sum_betas2_m = np.add(all_sum_betas2_m, np.power(full_betas_mean_m, 2))
                all_sum_z_scores_m = np.add(all_sum_z_scores_m, full_z_scores_m)
                all_sum_z_scores2_m = np.add(all_sum_z_scores2_m, np.power(full_z_scores_m, 2))
                all_num_sum_m += 1

                all_sum_Ys_m = np.add(all_sum_Ys_m, Y_sample_m)
                all_sum_Ys2_m = np.add(all_sum_Ys2_m, np.power(Y_sample_m, 2))

                R_Y_v = np.zeros(all_sum_Ys_m.shape[1])
                R_beta_v = np.zeros(all_sum_betas_m.shape[1])

                if increase_hyper_if_betas_below is not None:
                    #check to make sure that we satisfy
                    if np.any(all_num_sum_m == 0):
                        gibbs_good = False
                    else:

                        #check both sum of all iterations (to not wait until convergence to detect failures)
                        #and sum of iterations after convergence
                        _, all_cur_avg_betas_v = __outlier_resistant_mean(all_sum_betas_m, all_num_sum_m)

                        fraction_required = 0.001
                        self._record_param("fraction_required_to_not_increase_hyper", fraction_required)

                        #all_low = np.all(all_cur_avg_betas_v / self.scale_factors < increase_hyper_if_betas_below)
                        #all_low = np.mean(all_cur_avg_betas_v / self.scale_factors > increase_hyper_if_betas_below) < fraction_required
                        all_low = False

                        if np.all(num_sum_beta_m > 0):
                            _, cur_avg_betas_v = __outlier_resistant_mean(sum_betas_m, num_sum_beta_m)
                            #all_low = all_low or np.all(cur_avg_betas_v / self.scale_factors < increase_hyper_if_betas_below)
                            #all_low = np.all(cur_avg_betas_v / self.scale_factors < increase_hyper_if_betas_below)
                            all_low = np.mean(cur_avg_betas_v / self.scale_factors > increase_hyper_if_betas_below) < fraction_required
                            
                        #if np.all(all_num_sum_m > 0):
                        #    top_gene_set = np.argmax(np.mean(all_sum_betas_m / all_num_sum_m, axis=0) / self.scale_factors)
                        #    log("Top gene set %s has all value %.3g" % (self.gene_sets[top_gene_set], (np.mean(all_sum_betas_m / all_num_sum_m, axis=0) / self.scale_factors)[top_gene_set]), TRACE)
                        #    top_gene_set2 = np.argmax(all_cur_avg_betas_v / self.scale_factors)
                        #    log("Top gene set %s has all outlier value %.3g" % (self.gene_sets[top_gene_set], (all_cur_avg_betas_v / self.scale_factors)[top_gene_set]), TRACE)

                        if np.all(num_sum_beta_m > 0):
                            top_gene_set = np.argmax(np.mean(sum_betas_m / num_sum_beta_m, axis=0) / self.scale_factors)
                            log("Top gene set %s has value %.3g" % (self.gene_sets[top_gene_set], (np.mean(sum_betas_m / num_sum_beta_m, axis=0) / self.scale_factors)[top_gene_set]), TRACE)
                            top_gene_set2 = np.argmax(cur_avg_betas_v / self.scale_factors)
                            log("Top gene set %s has outlier value %.3g" % (self.gene_sets[top_gene_set2], (cur_avg_betas_v / self.scale_factors)[top_gene_set]), TRACE)

                            #top_gene_sets = np.argsort(np.mean(sum_betas_m / num_sum_beta_m, axis=0) / self.scale_factors)[::-1][:10]
                            #for i in top_gene_sets:
                            #    log("Top %d gene set %s has value %.3g" % (i, self.gene_sets[i], (np.mean(sum_betas_m / num_sum_beta_m, axis=0) / self.scale_factors)[i]), TRACE)
                            #    #print(sum_betas_m[:,i])
                            #    #print(num_sum_beta_m[:,i])

                            #top_gene_sets = np.argsort(cur_avg_betas_v / self.scale_factors)[::-1][:10]
                            #for i in top_gene_sets:
                            #    log("Top %d gene set %s has outlier value %.3g" % (i, self.gene_sets[i], (cur_avg_betas_v[i] / self.scale_factors)[i]), TRACE)
                            #    #print(sum_betas_m[:,i])
                            #    #print(num_sum_beta_m[:,i])


                        if all_low:

                            log("Only %.3g of %d (%.3g) are above %.3g" % (np.sum(cur_avg_betas_v / self.scale_factors > increase_hyper_if_betas_below), len(cur_avg_betas_v), np.mean(cur_avg_betas_v / self.scale_factors > increase_hyper_if_betas_below), increase_hyper_if_betas_below))

                            #at minimum, guarantee that it will restart unless it gets above this
                            gibbs_good = False
                            #only if above num for checking though that we increase and restart
                            if iteration_num > num_before_checking_p_increase:
                                new_p = self.p
                                new_sigma2 = self.sigma2

                                self._record_param("p_scale_factor", p_scale_factor)

                                new_p = self.p * p_scale_factor
                                num_p_increases += 1
                                if new_p > 1:
                                    new_p = 1

                              
                                break_loop = False
                                if new_p != self.p and num_restarts < max_num_restarts:
                                    #update so that new_sigma2 / new_p = self.sigma2 / self.p
                                    new_sigma2 = self.sigma2 * new_p / self.p

                                    self.ps *= new_p / self.p
                                    self.set_p(new_p)
                                    self._record_param("p_adj", new_p)
                                    log("Detected all gene set betas below %.3g; increasing p to %.3g and restarting gibbs" % (increase_hyper_if_betas_below, self.p))

                                    #restart
                                    break_loop = True
                                if new_sigma2 != self.sigma2 and num_restarts < max_num_restarts:
                                    self.sigma2s *= new_sigma2 / self.sigma2
                                    self._record_param("sigma2_adj", new_sigma2)
                                    self.set_sigma(new_sigma2, self.sigma_power)
                                    log("Detected all gene set betas below %.3g; increasing sigma to %.3g and restarting gibbs" % (increase_hyper_if_betas_below, self.sigma2))
                                    break_lopp = True
                                if break_loop:
                                    break
                        else:
                            gibbs_good = True

                if np.any(np.concatenate((burn_in_phase_Y_v, burn_in_phase_beta_v))):
                    if iteration_num + 1 >= max_num_burn_in:
                        burn_in_phase_Y_v[:] = False
                        burn_in_phase_beta_v[:] = False
                        log("Stopping Gibbs burn in after %d iterations" % (iteration_num+1), INFO)
                    elif gauss_seidel:
                        if prev_Ys_m is not None:
                            sum_diff = np.sum(np.abs(prev_Ys_m - Y_sample_m))
                            sum_prev = np.sum(np.abs(prev_Ys_m))
                            max_diff_frac = np.max(np.abs((prev_Ys_m - Y_sample_m)/prev_Ys_m))

                            tot_diff = sum_diff / sum_prev
                            log("Gibbs iteration %d: mean gauss seidel difference = %.4g / %.4g = %.4g; max frac difference = %.4g" % (iteration_num+1, sum_diff, sum_prev, tot_diff, max_diff_frac))
                            if iteration_num > min_num_iter and tot_diff < eps:
                                log("Gibbs gauss converged after %d iterations" % iteration_num, INFO)
                                burn_in_phase_Y_v[:] = False
                                burn_in_phase_beta_v[:] = False

                        prev_Ys_m = Y_sample_m
                    else:
                        def __calculate_R(sum_m, sum2_m, num):
                            #mean of betas across all iterations; psi_dot_j
                            mean_m = sum_m / float(num)
                            #mean of betas across replicates; psi_dot_dot
                            mean_v = np.mean(mean_m, axis=0)
                            #variances of betas across all iterators; s_j
                            var_m = (sum2_m - float(num) * np.power(mean_m, 2)) / (float(num) - 1)
                            B_v = (float(num) / (mean_m.shape[0] - 1)) * np.sum(np.power(mean_m - mean_v, 2), axis=0)
                            W_v = (1.0 / float(mean_m.shape[0])) * np.sum(var_m, axis=0)
                            var_given_y_v = np.add((float(num) - 1) / float(num) * W_v, (1.0 / float(num)) * B_v)
                            var_given_y_v[var_given_y_v < 0] = 0
                            R_v = np.ones(len(W_v))
                            R_non_zero_mask = W_v > 0
                            R_v[R_non_zero_mask] = np.sqrt(var_given_y_v[R_non_zero_mask] / W_v[R_non_zero_mask])
                            return (B_v, W_v, R_v, var_given_y_v)

                        if iteration_num > min_num_burn_in:
                            (B_Y_v, W_Y_v, R_Y_v, var_given_y_Y_v) = __calculate_R(all_sum_Ys_m, all_sum_Ys2_m, iteration_num)
                            (B_beta_v, W_beta_v, R_beta_v, var_given_y_beta_v) = __calculate_R(all_sum_betas_m, all_sum_betas2_m, iteration_num)

                            B_v = np.concatenate((B_Y_v, B_beta_v))
                            W_v = np.concatenate((W_Y_v, W_beta_v))
                            R_v = np.concatenate((R_Y_v, R_beta_v))
                            W_v = np.concatenate((R_Y_v, R_beta_v))

                            mean_thresholded_R_Y = np.mean(R_Y_v[R_Y_v >= 1]) if np.sum(R_Y_v >= 1) > 0 else 1
                            max_index_Y = np.argmax(R_Y_v)
                            mean_thresholded_R_beta = np.mean(R_beta_v[R_beta_v >= 1]) if np.sum(R_beta_v >= 1) > 0 else 1
                            max_index_beta = np.argmax(R_beta_v)
                            mean_thresholded_R = np.mean(R_v[R_v >= 1]) if np.sum(R_v >= 1) > 0 else 1
                            max_index = np.argmax(R_v)
                            log("Gibbs iteration %d: max ind=%d; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;" % (iteration_num+1, max_index, B_v[max_index], W_v[max_index], R_v[max_index], mean_thresholded_R, np.sum(R_v > r_threshold_burn_in)))
                            log("For Y: max ind=%d; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;" % (max_index_Y, B_Y_v[max_index_Y], W_Y_v[max_index_Y], R_Y_v[max_index_Y], mean_thresholded_R_Y, np.sum(R_Y_v > r_threshold_burn_in)), TRACE)
                            log("For beta: max ind=%d; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;" % (max_index_beta, B_beta_v[max_index_beta], W_beta_v[max_index_beta], R_beta_v[max_index_beta], mean_thresholded_R_beta, np.sum(R_beta_v > r_threshold_burn_in)), TRACE)

                            if use_max_r_for_convergence:
                                convergence_statistic = R_v[max_index]
                            else:
                                convergence_statistic = mean_thresholded_R


                            num_Y_converged = np.sum(np.logical_and(burn_in_phase_Y_v, R_Y_v < r_threshold_burn_in))
                            num_beta_converged = np.sum(np.logical_and(burn_in_phase_beta_v, R_beta_v < r_threshold_burn_in))

                            burn_in_phase_Y_v[R_Y_v < r_threshold_burn_in] = False
                            burn_in_phase_beta_v[R_beta_v < r_threshold_burn_in] = False

                            if num_Y_converged > 0:
                                log("Gibbs converged for %d Ys (%d remaining) after %d iterations" % (num_Y_converged, np.sum(burn_in_phase_Y_v), iteration_num+1), INFO)
                            if num_beta_converged > 0:
                                log("Gibbs converged for %d betas (%d remaining) after %d iterations" % (num_beta_converged, np.sum(burn_in_phase_beta_v), iteration_num+1), INFO)

                done = False

                betas_sem2_v = np.zeros(sum_betas_m.shape[1])
                sem2_v = np.zeros(sum_log_pos_m.shape[1])

                converged_Y_v = ~burn_in_phase_Y_v
                converged_beta_v = ~burn_in_phase_beta_v

                if np.sum(converged_Y_v) + np.sum(converged_beta_v) > 0:

                    #sum_Ys_post_m = np.add(sum_Ys_post_m, Y_sample_m)
                    #sum_Ys2_post_m += np.add(sum_Ys2_post_m, np.square(Y_sample_m))
                    #num_sum_post += 1

                    if np.sum(converged_Y_v) > 0:
                        sum_Ys_m[:,converged_Y_v] += Y_sample_m[:,converged_Y_v]
                        sum_Ys2_m[:,converged_Y_v] += np.power(Y_sample_m[:,converged_Y_v], 2)
                        sum_Y_raws_m[:,converged_Y_v] += Y_raw_sample_m[:,converged_Y_v]
                        sum_log_pos_m[:,converged_Y_v] += log_po_sample_m[:,converged_Y_v]
                        sum_log_pos2_m[:,converged_Y_v] += np.power(log_po_sample_m[:,converged_Y_v], 2)
                        sum_log_po_raws_m[:,converged_Y_v] += log_po_raw_sample_m[:,converged_Y_v]
                        sum_priors_m[:,converged_Y_v] += priors_for_Y_m[:,converged_Y_v]
                        sum_Ds_m[:,converged_Y_v] += D_sample_m[:,converged_Y_v]
                        sum_D_raws_m[:,converged_Y_v] += D_raw_sample_m[:,converged_Y_v]
                        sum_bf_orig_m[:,converged_Y_v] += log_bf_m[:,converged_Y_v]
                        sum_bf_orig_raw_m[:,converged_Y_v] += log_bf_raw_m[:,converged_Y_v]
                        num_sum_Y_m[:,converged_Y_v] += 1


                    #temp_genes = ["FTO", "IRS1", "ANKH", "INSR"]
                    #temp_genes = [x for x in temp_genes if x in self.gene_to_ind]

                    if np.sum(converged_beta_v) > 0:

                        sum_betas_m[:,converged_beta_v] += full_betas_mean_m[:,converged_beta_v]
                        sum_betas2_m[:,converged_beta_v] += np.power(full_betas_mean_m[:,converged_beta_v], 2)
                        sum_betas_uncorrected_m[:,converged_beta_v] += uncorrected_betas_mean_m[:,converged_beta_v]
                        sum_postp_m[:,converged_beta_v] += full_postp_sample_m[:,converged_beta_v]
                        sum_beta_tildes_m[:,converged_beta_v] += full_beta_tildes_m[:,converged_beta_v]
                        sum_z_scores_m[:,converged_beta_v] += full_z_scores_m[:,converged_beta_v]
                        num_sum_beta_m[:,converged_beta_v] += 1

                    if self.genes_missing is not None and np.sum(burn_in_phase_beta_v) == 0:
                        burn_in_phase_Y_missing_v = (self.X_orig_missing_genes != 0).multiply(burn_in_phase_beta_v).sum(axis=1).astype(bool).A1

                        converged_Y_missing_v = ~burn_in_phase_Y_missing_v

                        if np.sum(converged_Y_missing_v) > 0:

                            sum_priors_missing_m[:,converged_Y_missing_v] += priors_missing_mean_m[:,converged_Y_missing_v]

                            max_log = 15
                            cur_log_priors_missing_m = priors_missing_mean_m[:,converged_Y_missing_v] + self.background_log_bf
                            cur_log_priors_missing_m[cur_log_priors_missing_m > max_log] = max_log

                            sum_Ds_missing_m[:,converged_Y_missing_v] += np.exp(cur_log_priors_missing_m) / (1 + np.exp(cur_log_priors_missing_m))

                            num_sum_priors_missing_m[:,converged_Y_missing_v] += 1


                    #record these for tracing

                    if np.all(num_sum_Y_m > 1) and np.all(num_sum_beta_m > 1):
                        betas_sem2_m = ((sum_betas2_m / (num_sum_beta_m - 1)) - np.power(sum_betas_m / num_sum_beta_m, 2)) / num_sum_beta_m

                        #calculate effective sample size
                        #see: https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html
                        #we have to keep rho_t for t=1...2m+1
                        #to get rho_t, multiply current Y_sample_m

                        #first the correlation vectors
                        #rho_t = np.zeros(len(var_given_y_v))
                        #for i in range(num_chains_betas):
                        #    np.correlate(
                        #    rho_t += 

                        #1 - ((W_v - (1 / num_chains_betas) * sum_rho_t_m) / var_given_y_v)

                        avg_v = np.mean(sum_log_pos_m / num_sum_Y_m, axis=0)
                        sem2_v = np.var(sum_log_pos_m / num_sum_Y_m, axis=0) / np.mean(num_sum_Y_m, axis=0)

                        max_avg = np.max(avg_v)
                        min_avg = np.min(avg_v)
                        ref_val = max_avg - min_avg
                        if ref_val == 0:
                            ref_val = np.sqrt(np.var(Y_sample_m))
                            if ref_val == 0:
                                ref_val = 1

                        max_sem = np.max(np.sqrt(sem2_v))
                        max_percentage_error = max_sem / ref_val

                        log("Gibbs iteration %d: ref_val=%.3g; max_sem=%.3g; max_ratio=%.3g" % (iteration_num+1, ref_val, max_sem, max_percentage_error))
                        if iteration_num >= min_num_iter and max_percentage_error < max_frac_sem and (increase_hyper_if_betas_below is not None and iteration_num >= num_before_checking_p_increase):
                            log("Desired Gibbs precision achieved; stopping sampling")
                            done = True

                if gene_set_stats_trace_out is not None:
                    log("Writing gene set stats trace", TRACE)
                    for chain_num in range(num_chains):
                        for i in range(len(self.gene_sets)):
                            if only_gene_sets is None or self.gene_sets[i] in only_gene_sets:
                                gene_set_stats_trace_fh.write("%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n" % (iteration_num+1, chain_num+1, self.gene_sets[i], full_beta_tildes_m[chain_num,i] / self.scale_factors[i], full_p_values_m[chain_num,i], full_z_scores_m[chain_num,i], full_ses_m[chain_num,i] / self.scale_factors[i], (uncorrected_betas_mean_m[chain_num,i] if use_mean_betas else uncorrected_betas_sample_m[chain_num,i])  / self.scale_factors[i], (full_betas_mean_m[chain_num,i] if use_mean_betas else full_betas_sample_m[chain_num,i]) / self.scale_factors[i], (full_postp_mean_m[chain_num,i] if use_mean_betas else full_postp_sample_m[chain_num,i]), full_z_cur_beta_tildes_m[chain_num,i], R_beta_v[i], betas_sem2_v[i]))

                    gene_set_stats_trace_fh.flush()

                if done:
                    break

            if gene_set_stats_trace_out is not None:
                gene_set_stats_trace_fh.close()
            if gene_stats_trace_out is not None:
                gene_stats_trace_fh.close()

            #reached the max; go with whatever we have
            if num_restarts >= max_num_restarts:
                gibbs_good = True

            #restart if not good
            if not gibbs_good:
                continue

            assert(np.all(num_sum_Y_m > 0))
            assert(np.all(num_sum_beta_m > 0))

            ##1. calculate mean values for each chain (divide by number -- make sure it is correct; may not be num_avg_Y)
            #beta_chain_means_m = sum_betas_m / num_sum_beta_m
            #Y_chain_means_m = sum_Ys_m / num_sum_Y_m

            ##2. calculate median values across chains (one number per gene set/gene)
            #beta_medians_v = np.median(beta_chain_means_m, axis=0)
            #Y_medians_v = np.median(Y_chain_means_m, axis=0)

            ##3. calculate abs(difference) between each chain and median (one value per chain/geneset)
            #beta_mad_m = np.abs(beta_chain_means_m - beta_medians_v)
            #Y_mad_m = np.abs(Y_chain_means_m - Y_medians_v)

            ##4. calculate median of abs(difference) across chains (one number per gene set/gene)
            #beta_mad_median_v = np.median(beta_mad_m, axis=0)
            #Y_mad_median_v = np.median(Y_mad_m, axis=0)

            ##5. mask any chain that is more than 3 median(abs(difference)) from median
            #beta_outlier_mask_m = beta_chain_means_m > beta_medians_v + 3 * beta_mad_median_v
            #Y_outlier_mask_m = Y_chain_means_m > Y_medians_v + 3 * Y_mad_median_v

            ##6. take average only across chains that are not outliers
            #num_sum_beta_v = np.sum(~beta_outlier_mask_m, axis=0)
            #num_sum_Y_v = np.sum(~Y_outlier_mask_m, axis=0)

            ##should never happen but just in case
            #num_sum_beta_v[num_sum_beta_v == 0] = 1
            #num_sum_Y_v[num_sum_Y_v == 0] = 1

            ##7. to do this, zero out outlier chains, then sum them, then divide by number of outliers
            #sum_Ys_m[Y_outlier_mask_m] = 0
            #avg_Ys_v = np.sum(sum_Ys_m / num_sum_Y_m, axis=0) / num_sum_Y_v

            Y_outlier_mask_m, avg_Ys_v = __outlier_resistant_mean(sum_Ys_m, num_sum_Y_m)
            beta_outlier_mask_m, avg_betas_v = __outlier_resistant_mean(sum_betas_m, num_sum_beta_m)
            
            _, avg_Y_raws_v = __outlier_resistant_mean(sum_Y_raws_m, num_sum_Y_m)

            #sum_log_pos_m[Y_outlier_mask_m] = 0
            #avg_log_pos_v = np.sum(sum_log_pos_m / num_sum_Y_m, axis=0) / num_sum_Y_v
            _, avg_log_pos_v = __outlier_resistant_mean(sum_log_pos_m, num_sum_Y_m, Y_outlier_mask_m)

            _, avg_log_po_raws_v = __outlier_resistant_mean(sum_log_po_raws_m, num_sum_Y_m, Y_outlier_mask_m)

            #sum_log_pos2_m[Y_outlier_mask_m] = 0
            #avg_log_pos2_v = np.sum(sum_log_pos2_m / num_sum_Y_m, axis=0) / num_sum_Y_v
            _, avg_log_pos2_v = __outlier_resistant_mean(sum_log_pos2_m, num_sum_Y_m, Y_outlier_mask_m)

            #sum_Ds_m[Y_outlier_mask_m] = 0
            #avg_Ds_v = np.sum(sum_Ds_m / num_sum_Y_m, axis=0) / num_sum_Y_v
            _, avg_Ds_v = __outlier_resistant_mean(sum_Ds_m, num_sum_Y_m, Y_outlier_mask_m)

            _, avg_D_raws_v = __outlier_resistant_mean(sum_D_raws_m, num_sum_Y_m, Y_outlier_mask_m)

            #sum_priors_m[Y_outlier_mask_m] = 0
            #avg_priors_v = np.sum(sum_priors_m / num_sum_Y_m, axis=0) / num_sum_Y_v
            _, avg_priors_v = __outlier_resistant_mean(sum_priors_m, num_sum_Y_m, Y_outlier_mask_m)

            #sum_bf_orig_m[Y_outlier_mask_m] = 0
            #avg_bf_orig_v = np.sum(sum_bf_orig_m / num_sum_Y_m, axis=0) / num_sum_Y_v
            _, avg_bf_orig_v = __outlier_resistant_mean(sum_bf_orig_m, num_sum_Y_m, Y_outlier_mask_m)

            #sum_bf_orig_raw_m[Y_outlier_mask_m] = 0
            #avg_bf_orig_raw_v = np.sum(sum_bf_orig_raw_m / num_sum_Y_m, axis=0) / num_sum_Y_v
            _, avg_bf_orig_raw_v = __outlier_resistant_mean(sum_bf_orig_raw_m, num_sum_Y_m, Y_outlier_mask_m)

            if self.genes_missing is not None:
                #priors_missing_chain_means_m = sum_priors_missing_m / num_sum_priors_missing_m
                #priors_missing_medians_v = np.median(priors_missing_chain_means_m, axis=0)
                #priors_missing_mad_m = np.abs(priors_missing_chain_means_m - priors_missing_medians_v)
                #priors_missing_mad_median_v = np.median(priors_missing_mad_m, axis=0)
                #priors_missing_outlier_mask_m = priors_missing_chain_means_m > priors_missing_medians_v + 3 * priors_missing_mad_median_v
                #num_sum_priors_missing_v = np.sum(~priors_missing_outlier_mask_m, axis=0)
                #num_sum_priors_missing_v[num_sum_priors_missing_v == 0] = 1

                #assert(np.all(num_sum_priors_missing_m > 0))
                #sum_priors_missing_m[priors_missing_outlier_mask_m] = 0
                #avg_priors_missing_v = np.sum(sum_priors_missing_m / num_sum_priors_missing_m, axis=0) / num_sum_priors_missing_v

                priors_missing_outlier_mask_m, avg_priors_missing_v = __outlier_resistant_mean(sum_priors_missing_m, num_sum_priors_missing_m)

                #sum_Ds_missing_m[priors_missing_outlier_mask_m] = 0
                #avg_Ds_missing_v = np.sum(sum_Ds_missing_m / num_sum_priors_missing_m, axis=0) / num_sum_priors_missing_v
                _, avg_Ds_missing_v = __outlier_resistant_mean(sum_Ds_missing_m, num_sum_priors_missing_m, priors_missing_outlier_mask_m)

            #sum_betas_m[beta_outlier_mask_m] = 0
            #avg_betas_v = np.sum(sum_betas_m / num_sum_beta_m, axis=0) / num_sum_beta_v
            #we did this above

            #sum_betas_uncorrected_m[beta_outlier_mask_m] = 0
            #avg_betas_uncorrected_v = np.sum(sum_betas_uncorrected_m / num_sum_beta_m, axis=0) / num_sum_beta_v
            _, avg_betas_uncorrected_v = __outlier_resistant_mean(sum_betas_uncorrected_m, num_sum_beta_m, beta_outlier_mask_m)

            #sum_postp_m[beta_outlier_mask_m] = 0
            #avg_postp_v = np.sum(sum_postp_m / num_sum_beta_m, axis=0) / num_sum_beta_v
            _, avg_postp_v = __outlier_resistant_mean(sum_postp_m, num_sum_beta_m, beta_outlier_mask_m)

            #sum_beta_tildes_m[beta_outlier_mask_m] = 0
            #avg_beta_tildes_v = np.sum(sum_beta_tildes_m / num_sum_beta_m, axis=0) / num_sum_beta_v
            _, avg_beta_tildes_v = __outlier_resistant_mean(sum_beta_tildes_m, num_sum_beta_m, beta_outlier_mask_m)

            #sum_z_scores_m[beta_outlier_mask_m] = 0
            #avg_z_scores_v = np.sum(sum_z_scores_m / num_sum_beta_m, axis=0) / num_sum_beta_v
            _, avg_z_scores_v = __outlier_resistant_mean(sum_z_scores_m, num_sum_beta_m, beta_outlier_mask_m)

            self.beta_tildes = avg_beta_tildes_v
            self.z_scores = avg_z_scores_v
            self.p_values = 2*scipy.stats.norm.cdf(-np.abs(self.z_scores))
            self.ses = np.full(self.beta_tildes.shape, 100.0)
            self.ses[self.z_scores != 0] = np.abs(self.beta_tildes[self.z_scores != 0] / self.z_scores[self.z_scores != 0])

            self.betas = avg_betas_v
            self.betas_uncorrected = avg_betas_uncorrected_v
            self.inf_betas = None
            self.non_inf_avg_cond_betas = None
            self.non_inf_avg_postps = avg_postp_v

            #priors_missing is at the end
            self.priors = avg_priors_v
            self.priors_missing = avg_priors_missing_v
            self.combined_Ds_missing = avg_Ds_missing_v

            self.Y_for_regression = avg_bf_orig_v
            self.Y = avg_bf_orig_raw_v

            self.combined_Ds_for_regression = avg_Ds_v
            self.combined_Ds = avg_D_raws_v

            self.combined_prior_Ys_for_regression = avg_log_pos_v - self.background_log_bf
            self.combined_prior_Ys = avg_log_po_raws_v - self.background_log_bf

            #self.combined_prior_Y_ses = avg_log_pos_ses

            gene_N = self.get_gene_N()
            gene_N_missing = self.get_gene_N(get_missing=True)

            all_gene_N = gene_N
            if self.genes_missing is not None:
                assert(gene_N_missing is not None)
                all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

            total_priors = np.concatenate((self.priors, self.priors_missing))
            priors_slope = np.cov(total_priors, all_gene_N)[0,1] / np.var(all_gene_N)
            priors_intercept = np.mean(total_priors - all_gene_N * priors_slope)

            log("Adjusting priors with slope %.4g" % priors_slope)
            self.priors_adj = self.priors - priors_slope * gene_N - priors_intercept
            if self.genes_missing is not None:
                self.priors_adj_missing = self.priors_missing - priors_slope * gene_N_missing

            combined_slope = np.cov(self.combined_prior_Ys, gene_N)[0,1] / np.var(gene_N)
            combined_intercept = np.mean(self.combined_prior_Ys - gene_N * combined_slope)

            log("Adjusting combined with slope %.4g" % combined_slope)
            self.combined_prior_Ys_adj = self.combined_prior_Ys - combined_slope * gene_N - combined_intercept

            begin_slice = int(min_num_iter * 0.1)

            if gibbs_good:
                break

        #print("END:")
        #print("Avg Mean Y")
        #for gene in temp_genes:
        #    print("%s: %s" % (gene, avg_Ys[self.gene_to_ind[gene]]))
        #print("Avg PO")
        #for gene in temp_genes:
        #    print("%s: %s" % (gene, avg_log_pos[self.gene_to_ind[gene]]))
        #print("Beta")
        #print(avg_betas)
        #print("Beta Mean")
        #print(np.mean(avg_betas))

    #def run_factor(self, max_num_factors=10, min_gene_set_beta=0.001, min_gene_set_beta_uncorrected=0.001, max_gene_set_p=0.05, multiply_Y=False, run_transpose=True, max_num_iterations=100, rel_tol=0.01):
    def run_factor(self, max_num_factors=15, alpha0=10, beta0=1, gene_set_filter_type=None, gene_set_filter_value=None, gene_filter_type=None, gene_filter_value=None, gene_set_multiply_type=None, gene_multiply_type=None, run_transpose=True, max_num_iterations=100, rel_tol=0.01, lmm_auth_key=None):

        BETA_UNCORRECTED_OPTIONS = ["beta_uncorrected", "betas_uncorrected"]
        BETA_OPTIONS = ["beta", "betas"]
        P_OPTIONS = ["p_value", "p", "p_value"]
        NONE_OPTIONS = ["none"]

        self._record_params({"max_num_factors": max_num_factors, "alpha0": alpha0, "gene_set_filter_type": gene_set_filter_type, "gene_set_filter_value": gene_set_filter_value, "gene_filter_type": gene_filter_type, "gene_filter_value": gene_filter_value, "gene_set_multiply_type": gene_set_multiply_type, "gene_multiply_type": gene_multiply_type, "run_transpose": run_transpose})

        gene_set_mask = np.full(self.X_orig.shape[1], True)
        if gene_set_filter_type is not None and gene_set_filter_value is not None:
            if gene_set_filter_type.lower() in BETA_UNCORRECTED_OPTIONS:
                if self.betas_uncorrected is None:
                    bail("Can't run filtering in factor; betas_uncorrected was not loaded")
                gene_set_mask = self.betas_uncorrected > gene_set_filter_value
            elif gene_set_filter_type.lower() in BETA_OPTIONS:
                if self.betas is None:
                    bail("Can't run filtering in factor; betas was not loaded")
                gene_set_mask = self.betas > gene_set_filter_value
            elif gene_set_filter_type.lower() in P_OPTIONS:
                if self.p_values is None:
                    bail("Can't run filtering in factor; p was not loaded")
                gene_set_mask = self.p_values < gene_set_filter_value
            elif gene_set_filter_type.lower() in NONE_OPTIONS:
                pass
            else:
                bail("Valid values for --gene-set-filter-type are beta_uncorrected|beta|p_value")

        gene_set_vector = np.ones(self.X_orig.shape[1])
        if gene_set_multiply_type is not None:
            if gene_set_multiply_type.lower() in BETA_UNCORRECTED_OPTIONS:
                if self.betas_uncorrected is None:
                    bail("Can't run multiply in factor; betas_uncorrected was not loaded")
                gene_set_vector = self.betas_uncorrected
            elif gene_set_multiply_type.lower() in BETA_OPTIONS:
                if self.betas is None:
                    bail("Can't run multiply in factor; betas was not loaded")
                gene_set_vector = self.betas
            elif gene_set_multiply_type.lower() in P_OPTIONS:
                if self.p_values is None:
                    bail("Can't run multiply in factor; p was not loaded")
                gene_set_vector = -np.log(self.p_values)
            elif gene_set_multiply_type.lower() in NONE_OPTIONS:
                pass
            else:
                bail("Valid values for --gene-set-multiply-type are beta_uncorrected|beta|p_value")


        PRIOR_OPTIONS = ["prior", "priors"]
        COMBINED_OPTIONS = ["combined", "combined_prior_Y", "combined_prior_Ys"]
        Y_OPTIONS = ["y", "log_bf"]

        gene_mask = np.full(self.X_orig.shape[0], True)
        if gene_filter_type is not None and gene_filter_value is not None:
            if gene_filter_type.lower() in PRIOR_OPTIONS:
                if self.priors is None:
                    bail("Can't run filtering in factor; priors were not loaded")
                gene_mask = self.priors > gene_filter_value
            elif gene_filter_type.lower() in COMBINED_OPTIONS:
                if self.combined_prior_Ys is None:
                    bail("Can't run filtering in factor; combined was not loaded")
                gene_mask = self.combined_prior_Ys > gene_filter_value
            elif gene_filter_type.lower() in Y_OPTIONS:
                if self.Y is None:
                    bail("Can't run filtering in factor; log_bf was not loaded")
                gene_mask = self.Y > gene_filter_value
            elif gene_filter_type.lower() in NONE_OPTIONS:
                pass
            else:
                bail("Valid values for --gene-filter-type are prior|combined|log_bf")

        gene_vector = np.ones(self.X_orig.shape[0])
        if gene_multiply_type is not None:
            if gene_multiply_type.lower() in PRIOR_OPTIONS:
                if self.priors is None:
                    bail("Can't run multiply in factor; priors were not loaded")
                gene_vector = self.priors
            elif gene_multiply_type.lower() in COMBINED_OPTIONS:
                if self.combined_prior_Ys is None:
                    bail("Can't run multiply in factor; combined was not loaded")
                gene_vector = self.combined_prior_Ys
            elif gene_multiply_type.lower() in Y_OPTIONS:
                if self.Y is None:
                    bail("Can't run multiply in factor; log_bf was not loaded")
                gene_vector = self.Y
            elif gene_multiply_type.lower() in NONE_OPTIONS:
                pass
            else:
                bail("Valid values for --gene-multiply-type are prior|combined|log_bf")

        #make sure everything is positive
        gene_set_vector[gene_set_vector < 0] = 0
        gene_vector[gene_vector < 0] = 0

        log("Running matrix factorization")

        matrix = self.X_orig[:,gene_set_mask][gene_mask,:].multiply(gene_set_vector[gene_set_mask]).toarray().T * gene_vector[gene_mask]
        log("Matrix shape: (%s, %s)" % (matrix.shape), DEBUG)

        if not run_transpose:
            matrix = matrix.T


        #this code is adapted from https://github.com/gwas-partitioning/bnmf-clustering
        def _bayes_nmf_l2(V0, n_iter=10000, a0=10, tol=1e-7, K=15, K0=15, phi=1.0):

            # V original and compute tildas
            # V0 is scaled down matrix

            # Bayesian NMF with half-normal priors for W and H
            # V0: input z-score matrix (variants x traits)
            # n_iter: Number of iterations for parameter optimization
            # a0: Hyper-parameter for inverse gamma prior on ARD relevance weights
            # tol: Tolerance for convergence of fitting procedure
            # K: Number of clusters to be initialized (algorithm may drive some to zero)
            # K0: Used for setting b0 (lambda prior hyper-parameter) -- should be equal to K
            # phi: Scaling parameter

            eps = 1.e-50
            delambda = 1.0
            #active_nodes = np.sum(V0, axis=0) != 0
            #V0 = V0[:,active_nodes]
            V = V0 - np.min(V0)
            Vmin = np.min(V)
            Vmax = np.max(V)
            N = V.shape[0]
            M = V.shape[1]

            W = np.random.random((N, K)) * Vmax #NxK
            H = np.random.random((K, M)) * Vmax #KxM

            I = np.ones((N, M)) #NxM
            V_ap = W.dot(H) + eps #NxM

            phi = np.power(np.std(V), 2) * phi
            C = (N + M) / 2 + a0 + 1
            b0 = 3.14 * (a0 - 1) * np.mean(V) / (2 * K0)
            lambda_bound = b0 / C
            lambdak = (0.5 * np.sum(np.power(W, 2), axis=0) + 0.5 * np.sum(np.power(H, 2), axis=1) + b0) / C
            lambda_cut = lambda_bound * 1.5

            n_like = [None]
            n_evid = [None]
            n_error = [None]
            n_lambda = [lambdak]
            it = 1
            count = 1
            while delambda >= tol and it < n_iter:
                H = H * (W.T.dot(V)) / (W.T.dot(V_ap) + phi * H * np.repeat(1/lambdak, M).reshape(len(lambdak), M) + eps)
                V_ap = W.dot(H) + eps
                W = W * (V.dot(H.T)) / (V_ap.dot(H.T) + phi * W * np.tile(1/lambdak, N).reshape(N, len(lambdak)) + eps)
                V_ap = W.dot(H) + eps
                lambdak = (0.5 * np.sum(np.power(W, 2), axis=0) + 0.5 * np.sum(np.power(H, 2), axis=1) + b0) / C
                delambda = np.max(np.abs(lambdak - n_lambda[it - 1]) / n_lambda[it - 1])
                like = np.sum(np.power(V - V_ap, 2)) / 2
                n_like.append(like)
                n_evid.append(like + phi * np.sum((0.5 * np.sum(np.power(W, 2), axis=0) + 0.5 * np.sum(np.power(H, 2), axis=1) + b0) / lambdak + C * np.log(lambdak)))
                n_lambda.append(lambdak)
                n_error.append(np.sum(np.power(V - V_ap, 2)))
                if it % 100 == 0:
                    log("Iteration=%d; evid=%.3g; lik=%.3g; err=%.3g; delambda=%.3g; factors=%d; factors_non_zero=%d" % (it, n_evid[it], n_like[it], n_error[it], delambda, np.sum(np.sum(W, axis=0) != 0), np.sum(lambdak >= lambda_cut)), TRACE)
                it += 1

            return W, H, n_like[-1], n_evid[-1], n_lambda[-1], n_error[-1]
                #W # Variant weight matrix (N x K)
                #H # Trait weight matrix (K x M)
                #n_like # List of reconstruction errors (sum of squared errors / 2) per iteration
                #n_evid # List of negative log-likelihoods per iteration
                #n_lambda # List of lambda vectors (shared weights for each of K clusters, some ~0) per iteration
                #n_error # List of reconstruction errors (sum of squared errors) per iteration

        # TODO reference for unpacking results
        result = _bayes_nmf_l2(matrix, a0=alpha0, K=max_num_factors, K0=max_num_factors)
        
        self.exp_lambdak = result[4]
        self.exp_gene_factors = result[1].T
        self.exp_gene_set_factors = result[0]


        #subset_down
        factor_mask = self.exp_lambdak != 0 & (np.sum(self.exp_gene_factors, axis=0) > 0) & (np.sum(self.exp_gene_set_factors, axis=0) > 0)
        factor_mask = factor_mask & (np.max(self.exp_gene_set_factors, axis=0) > 1e-5 * np.max(self.exp_gene_set_factors))

        if np.sum(~factor_mask) > 0:
            self.exp_lambdak = self.exp_lambdak[factor_mask]
            self.exp_gene_factors = self.exp_gene_factors[:,factor_mask]
            self.exp_gene_set_factors = self.exp_gene_set_factors[:,factor_mask]

        self.gene_factor_gene_mask = gene_mask
        self.gene_set_factor_gene_set_mask = gene_set_mask

        gene_set_values = None
        if self.betas is not None:
            gene_set_values = self.betas
        elif self.betas_uncorrected is not None:
            gene_set_values = self.betas_uncorrected

        gene_values = None
        if self.combined_prior_Ys is not None:
            gene_values = self.combined_prior_Ys
        elif self.priors is not None:
            gene_values = self.priors
        elif self.Y is not None:
            gene_values = self.Y

        if gene_set_values is not None:
            self.factor_gene_set_scores = self.exp_gene_set_factors.T.dot(gene_set_values[self.gene_set_factor_gene_set_mask])
        else:
            self.factor_gene_set_scores = self.exp_lambdak

        if gene_values is not None:
            self.factor_gene_scores = self.exp_gene_factors.T.dot(gene_values[self.gene_factor_gene_mask]) / self.exp_gene_factors.shape[0]
        else:
            self.factor_gene_scores = self.exp_lambdak


        reorder_inds = np.argsort(-self.factor_gene_set_scores)
        self.exp_lambdak = self.exp_lambdak[reorder_inds]
        self.factor_gene_set_scores = self.factor_gene_set_scores[reorder_inds]
        self.factor_gene_scores = self.factor_gene_scores[reorder_inds]
        self.exp_gene_factors = self.exp_gene_factors[:,reorder_inds]
        self.exp_gene_set_factors = self.exp_gene_set_factors[:,reorder_inds]

        #now label them
        gene_set_factor_gene_set_inds = np.where(self.gene_set_factor_gene_set_mask)[0]
        gene_factor_gene_inds = np.where(self.gene_factor_gene_mask)[0]

        num_top = 5
        top_gene_inds = np.argsort(-self.exp_gene_factors, axis=0)[:num_top,:]
        top_gene_set_inds = np.argsort(-self.exp_gene_set_factors, axis=0)[:num_top,:]

        self.factor_labels = []
        self.top_gene_sets = []
        self.top_genes = []
        factor_prompts = []
        for i in range(len(self.factor_gene_set_scores)):
            self.top_gene_sets.append([self.gene_sets[i] for i in np.where(self.gene_set_factor_gene_set_mask)[0][top_gene_set_inds[:,i]]])
            self.top_genes.append([self.genes[i] for i in np.where(self.gene_factor_gene_mask)[0][top_gene_inds[:,i]]])
            self.factor_labels.append(self.top_gene_sets[i][0] if len(self.top_gene_sets[i]) > 0 else "")
            factor_prompts.append(",".join(self.top_gene_sets[i]))

        if lmm_auth_key is not None:
            prompt = "Print a label to assign to each group: %s" % (" ".join(["%d. %s" % (j+1, ",".join(self.top_gene_sets[j])) for j in range(len(self.top_gene_sets))]))
            log("Querying LMM with prompt: %s" % prompt, DEBUG)
            response = query_lmm(prompt, lmm_auth_key)
            if response is not None:
                try:
                    responses = response.strip().split("\n")
                    responses = [x for x in responses if len(x) > 0]
                    if len(responses) == len(self.factor_labels):
                        for i in range(len(self.factor_labels)):
                            self.factor_labels[i] = responses[i]
                    else:
                        raise Exception
                except Exception:
                    log("Couldn't decode LMM response %s; using simple label" % response)
                    pass


        log("Found %d factors" % len(self.exp_lambdak), DEBUG)

    def get_col_sums(self, X, num_nonzero=False, axis=0):
        if num_nonzero:
            return X.astype(bool).sum(axis=axis).A1
        else:
            return np.abs(X).sum(axis=axis).A1

    def get_gene_N(self, get_missing=False):
        if get_missing:
            if self.gene_N_missing is None:
                return None
            else:
                return self.gene_N_missing + (self.gene_ignored_N_missing if self.gene_ignored_N_missing is not None else 0)
        else:
            if self.gene_N is None:
                return None
            else:
                return self.gene_N + (self.gene_ignored_N if self.gene_ignored_N is not None else 0)

    def write_gene_set_statistics(self, output_file, basic=False):
        log("Writing gene set stats to %s" % output_file, INFO)
        with open(output_file, 'w') as output_fh:
            if self.gene_sets is None:
                return
            header = "Gene_Set"
            if self.gene_set_labels is not None:
                header = "%s\t%s" % (header, "label")
            if self.X_orig is not None:
                col_sums = self.get_col_sums(self.X_orig)
                header = "%s\t%s" % (header, "N")
                header = "%s\t%s" % (header, "scale")
            if self.beta_tildes is not None:
                header = "%s\t%s\t%s\t%s\t%s\t%s" % (header, "beta_tilde", "beta_tilde_internal", "P", "Z", "SE")
            if self.inf_betas is not None and not basic:
                header = "%s\t%s" % (header, "inf_beta")            
            if self.betas is not None:
                header = "%s\t%s\t%s" % (header, "beta", "beta_internal")
            if self.betas_uncorrected is not None and not basic:
                header = "%s\t%s" % (header, "beta_uncorrected")            
            if not basic:
                if self.non_inf_avg_cond_betas is not None:
                    header = "%s\t%s" % (header, "avg_cond_beta")            
                if self.non_inf_avg_postps is not None:
                    header = "%s\t%s" % (header, "avg_postp")            
                if self.beta_tildes_orig is not None:
                    header = "%s\t%s\t%s\t%s\t%s\t%s" % (header, "beta_tilde_orig", "beta_tilde_internal_orig", "P_orig", "Z_orig", "SE_orig")
                if self.inf_betas_orig is not None:
                    header = "%s\t%s" % (header, "inf_beta_orig")            
                if self.betas_orig is not None:
                    header = "%s\t%s\t%s" % (header, "beta_orig", "beta_internal_orig")
                if self.betas_uncorrected_orig is not None:
                    header = "%s\t%s\t%s" % (header, "beta_uncorrected_orig", "beta_uncorrected_internal_orig")
                if self.non_inf_avg_cond_betas_orig is not None:
                    header = "%s\t%s" % (header, "avg_cond_beta_orig")            
                if self.non_inf_avg_postps_orig is not None:
                    header = "%s\t%s" % (header, "avg_postp_orig")            
                if self.ps is not None or self.p is not None:
                    header = "%s\t%s" % (header, "p_used")
                if self.sigma2s is not None or self.sigma2 is not None:
                    header = "%s\t%s" % (header, "sigma2_used")
                if (self.sigma2s is not None or self.sigma2 is not None) and self.sigma_threshold_k is not None and self.sigma_threshold_xo is not None:
                    header = "%s\t%s" % (header, "sigma2_thresholded")
                if self.X_osc is not None:
                    header = "%s\t%s\t%s\t%s" % (header, "O", "X_O", "weight")
                if self.total_qc_metrics is not None:
                    header = "%s\t%s" % (header, "\t".join(map(lambda x: "avg_%s" % x, [self.huge_gene_covariate_names[i] for i in range(len(self.huge_gene_covariate_names)) if i != self.huge_gene_covariate_intercept_index])))
                if self.mean_qc_metrics is not None:
                    header = "%s\t%s" % (header, "avg_avg_metric")

            output_fh.write("%s\n" % header)

            ordered_i = range(len(self.gene_sets))
            if self.betas is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.betas[k] / self.scale_factors[k])
            elif self.p_values is not None:
                ordered_i = sorted(ordered_i, key=lambda k: self.p_values[k])

            for i in ordered_i:
                line = self.gene_sets[i]
                if self.gene_set_labels is not None:
                    line = "%s\t%s" % (line, self.gene_set_labels[i])
                if self.X_orig is not None:
                    line = "%s\t%d" % (line, col_sums[i])
                    line = "%s\t%.3g" % (line, self.scale_factors[i])

                if self.beta_tildes is not None:
                    line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, self.beta_tildes[i] / self.scale_factors[i], self.beta_tildes[i], self.p_values[i], self.z_scores[i], self.ses[i] / self.scale_factors[i])
                if self.inf_betas is not None and not basic:
                    line = "%s\t%.3g" % (line, self.inf_betas[i] / self.scale_factors[i])            
                if self.betas is not None:
                    line = "%s\t%.3g\t%.3g" % (line, self.betas[i] / self.scale_factors[i], self.betas[i])
                if self.betas_uncorrected is not None and not basic:
                    line = "%s\t%.3g" % (line, self.betas_uncorrected[i] / self.scale_factors[i])            
                if not basic:
                    if self.non_inf_avg_cond_betas is not None:
                        line = "%s\t%.3g" % (line, self.non_inf_avg_cond_betas[i] / self.scale_factors[i])
                    if self.non_inf_avg_postps is not None:
                        line = "%s\t%.3g" % (line, self.non_inf_avg_postps[i])
                    if self.beta_tildes_orig is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, self.beta_tildes_orig[i] / self.scale_factors[i], self.beta_tildes_orig[i], self.p_values_orig[i], self.z_scores_orig[i], self.ses_orig[i] / self.scale_factors[i])
                    if self.inf_betas_orig is not None:
                        line = "%s\t%.3g" % (line, self.inf_betas_orig[i] / self.scale_factors[i])            
                    if self.betas_orig is not None:
                        line = "%s\t%.3g\t%.3g" % (line, self.betas_orig[i] / self.scale_factors[i], self.betas_orig[i])
                    if self.betas_uncorrected_orig is not None:
                        line = "%s\t%.3g\t%.3g" % (line, self.betas_uncorrected_orig[i] / self.scale_factors[i], self.betas_uncorrected_orig[i])
                    if self.non_inf_avg_cond_betas_orig is not None:
                        line = "%s\t%.3g" % (line, self.non_inf_avg_cond_betas_orig[i] / self.scale_factors[i])
                    if self.non_inf_avg_postps_orig is not None:
                        line = "%s\t%.3g" % (line, self.non_inf_avg_postps_orig[i])

                    if self.ps is not None or self.p is not None:
                        line = "%s\t%.3g" % (line, self.ps[i] if self.ps is not None else self.p)
                    if self.sigma2s is not None or self.sigma2 is not None:
                        line = "%s\t%.3g" % (line, self.get_scaled_sigma2(self.scale_factors[i], self.sigma2s[i] if self.sigma2s is not None else self.sigma2, self.sigma_power, None, None))
                    if (self.sigma2s is not None or self.sigma2 is not None) and self.sigma_threshold_k is not None and self.sigma_threshold_xo is not None:
                        line = "%s\t%.3g" % (line, self.get_scaled_sigma2(self.scale_factors[i], self.sigma2s[i] if self.sigma2s is not None else self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo))
                    if self.X_osc is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g" % (line, self.osc[i], self.X_osc[i], self.osc_weights[i])

                    if self.total_qc_metrics is not None:
                        line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, self.total_qc_metrics[i,:])))
                    if self.mean_qc_metrics is not None:
                        line = "%s\t%.3g" % (line, self.mean_qc_metrics[i])


                output_fh.write("%s\n" % line)

            if self.gene_sets_missing is not None:
                ordered_i = range(len(self.gene_sets_missing))
                if self.betas_missing is not None:
                    ordered_i = sorted(ordered_i, key=lambda k: -self.betas_missing[k] / self.scale_factors_missing[k])
                elif self.p_values_missing is not None:
                    ordered_i = sorted(ordered_i, key=lambda k: self.p_values_missing[k])

                col_sums_missing = self.get_col_sums(self.X_orig_missing_gene_sets)
                for i in range(len(self.gene_sets_missing)):
                    line = self.gene_sets_missing[i]
                    if self.gene_set_labels is not None:
                        line = "%s\t%s" % (line, self.gene_set_labels_missing[i])
                    line = "%s\t%d" % (line, col_sums_missing[i])
                    line = "%s\t%.3g" % (line, self.scale_factors_missing[i])

                    if self.beta_tildes is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, self.beta_tildes_missing[i] / self.scale_factors_missing[i], self.beta_tildes_missing[i], self.p_values_missing[i], self.z_scores_missing[i], self.ses_missing[i] / self.scale_factors_missing[i])
                    if self.inf_betas is not None and not basic:
                        line = "%s\t%.3g" % (line, self.inf_betas_missing[i] / self.scale_factors_missing[i])            
                    if self.betas is not None:
                        line = "%s\t%.3g\t%.3g" % (line, self.betas_missing[i] / self.scale_factors_missing[i], self.betas_missing[i])
                    if self.betas_uncorrected is not None and not basic:
                        line = "%s\t%.3g" % (line, self.betas_uncorrected_missing[i] / self.scale_factors_missing[i])            
                    if not basic:
                        if self.non_inf_avg_cond_betas is not None:
                            line = "%s\t%.3g" % (line, self.non_inf_avg_cond_betas_missing[i] / self.scale_factors_missing[i])
                        if self.non_inf_avg_postps is not None:
                            line = "%s\t%.3g" % (line, self.non_inf_avg_postps_missing[i])
                        if self.beta_tildes_orig is not None:
                            line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, self.beta_tildes_missing_orig[i] / self.scale_factors_missing[i], self.beta_tildes_missing_orig[i], self.p_values_missing_orig[i], self.z_scores_missing_orig[i], self.ses_missing_orig[i] / self.scale_factors_missing[i])
                        if self.inf_betas_orig is not None:
                            line = "%s\t%.3g" % (line, self.inf_betas_missing_orig[i] / self.scale_factors_missing[i])            
                        if self.betas_orig is not None:
                            line = "%s\t%.3g\t%.3g" % (line, self.betas_missing_orig[i] / self.scale_factors_missing[i], self.betas_missing_orig[i])
                        if self.betas_uncorrected_orig is not None:
                            line = "%s\t%.3g\t%.3g" % (line, self.betas_uncorrected_missing_orig[i] / self.scale_factors_missing[i], self.betas_uncorrected_missing_orig[i])
                        if self.non_inf_avg_cond_betas_orig is not None:
                            line = "%s\t%.3g" % (line, self.non_inf_avg_cond_betas_missing_orig[i] / self.scale_factors_missing[i])
                        if self.non_inf_avg_postps_orig is not None:
                            line = "%s\t%.3g" % (line, self.non_inf_avg_postps_missing_orig[i])

                        if self.ps is not None or self.p is not None:
                            line = "%s\t%.3g" % (line, self.ps_missing[i] if self.ps_missing is not None else self.p)

                        if self.sigma2s is not None or self.sigma2 is not None:
                            line = "%s\t%.3g" % (line, self.get_scaled_sigma2(self.scale_factors_missing[i], self.sigma2s_missing[i] if self.sigma2s_missing is not None else self.sigma2, self.sigma_power, None, None))
                        if (self.sigma2s is not None or self.sigma2 is not None) and self.sigma_threshold_k is not None and self.sigma_threshold_xo is not None:
                            line = "%s\t%.3g" % (line, self.get_scaled_sigma2(self.scale_factors_missing[i], self.sigma2s_missing[i] if self.sigma2s_missing is not None else self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo))

                        if self.X_osc is not None:
                            line = "%s\t%.3g\t%.3g\t%.3g" % (line, self.osc_missing[i], self.X_osc_missing[i], self.osc_weights_missing[i])

                        if self.total_qc_metrics is not None:
                            line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, self.total_qc_metrics_missing[i,:])))
                        if self.mean_qc_metrics is not None:
                            line = "%s\t%.3g" % (line, self.mean_qc_metrics_missing[i])

                    output_fh.write("%s\n" % line)



            if self.gene_sets_ignored is not None:

                ordered_i = range(len(self.gene_sets_ignored))
                if self.p_values_ignored is not None:
                    ordered_i = sorted(ordered_i, key=lambda k: self.p_values_ignored[k])

                for i in ordered_i:
                    line = "%s" % self.gene_sets_ignored[i]
                    if self.gene_set_labels is not None:
                        line = "%s\t%s" % (line, self.gene_set_labels_ignored[i])

                    line = "%s\t%d" % (line, self.col_sums_ignored[i])
                    line = "%s\t%.3g" % (line, self.scale_factors_ignored[i])

                    if self.beta_tildes is not None:
                        line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, self.beta_tildes_ignored[i] / self.scale_factors_ignored[i], self.beta_tildes_ignored[i], self.p_values_ignored[i], self.z_scores_ignored[i], self.ses_ignored[i] / self.scale_factors_ignored[i])
                    if self.inf_betas is not None and not basic:
                        line = "%s\t%.3g" % (line, 0)            
                    if self.betas is not None:
                        line = "%s\t%.3g\t%.3g" % (line, 0, 0)
                    if self.betas_uncorrected is not None and not basic:
                        line = "%s\t%.3g" % (line, 0)            
                    if not basic:
                        if self.non_inf_avg_cond_betas is not None:
                            line = "%s\t%.3g" % (line, 0)
                        if self.non_inf_avg_postps is not None:
                            line = "%s\t%.3g" % (line, 0)
                        if self.beta_tildes_orig is not None:
                            line = "%s\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g" % (line, self.beta_tildes_ignored[i] / self.scale_factors_ignored[i], self.beta_tildes_ignored[i], self.p_values_ignored[i], self.z_scores_ignored[i], self.ses_ignored[i] / self.scale_factors_ignored[i])
                        if self.inf_betas_orig is not None:
                            line = "%s\t%.3g" % (line, 0)
                        if self.betas_orig is not None:
                            line = "%s\t%.3g\t%.3g" % (line, 0, 0)
                        if self.betas_uncorrected_orig is not None:
                            line = "%s\t%.3g\t%.3g" % (line, 0, 0)
                        if self.non_inf_avg_cond_betas_orig is not None:
                            line = "%s\t%.3g" % (line, 0)
                        if self.non_inf_avg_postps_orig is not None:
                            line = "%s\t%.3g" % (line, 0)

                        if self.ps is not None or self.p is not None:
                            line = "%s\t%s" % (line, "NA")
                        if self.sigma2s is not None or self.sigma2 is not None:
                            line = "%s\t%s" % (line, "NA")
                        if (self.sigma2s is not None or self.sigma2 is not None) and self.sigma_threshold_k is not None and self.sigma_threshold_xo is not None:
                            line = "%s\t%s" % (line, "NA")

                        if self.X_osc is not None:
                            line = "%s\t%s\t%s\t%s" % (line, "NA", "NA", "NA")

                        if self.total_qc_metrics is not None:
                            line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, self.total_qc_metrics_ignored[i,:])))
                        if self.mean_qc_metrics is not None:
                            line = "%s\t%.3g" % (line, self.mean_qc_metrics_ignored[i])

                    output_fh.write("%s\n" % line)

    def write_gene_statistics(self, output_file):
        log("Writing gene stats to %s" % output_file, INFO)

        with open(output_file, 'w') as output_fh:
            if self.genes is not None:
                genes = self.genes
            elif self.gene_to_huge_score is not None:
                genes = list(self.gene_to_huge_score.keys())
            elif self.gene_to_gwas_huge_score is not None:
                genes = list(self.gene_to_huge_score.keys())
            elif self.gene_to_huge_score is not None:
                genes = list(self.gene_to_huge_score.keys())
            else:
                return

            huge_only_genes = set()
            if self.gene_to_huge_score is not None:
                huge_only_genes = set(self.gene_to_huge_score.keys()) - set(genes)
            if self.gene_to_gwas_huge_score is not None:
                huge_only_genes = set(self.gene_to_gwas_huge_score.keys()) - set(genes) - set(huge_only_genes)
            if self.gene_to_exomes_huge_score is not None:
                huge_only_genes = set(self.gene_to_exomes_huge_score.keys()) - set(genes) - set(huge_only_genes)

            if self.genes_missing is not None:
                huge_only_genes = huge_only_genes - set(self.genes_missing)

            huge_only_genes = list(huge_only_genes)

            write_regression = self.Y_for_regression is not None and self.Y is not None and np.any(~np.isclose(self.Y, self.Y_for_regression))

            header = "Gene"

            if self.priors is not None:
                header = "%s\t%s" % (header, "prior")
            if self.priors_adj is not None:
                header = "%s\t%s" % (header, "prior_adj")
            if self.combined_prior_Ys is not None:
                header = "%s\t%s" % (header, "combined")
            if self.combined_prior_Ys_adj is not None:
                header = "%s\t%s" % (header, "combined_adj")
            if self.combined_prior_Y_ses is not None:
                header = "%s\t%s" % (header, "combined_se")
            if self.combined_Ds is not None:
                header = "%s\t%s" % (header, "combined_D")
            if self.gene_to_huge_score is not None:
                header = "%s\t%s" % (header, "huge_score")
            if self.gene_to_gwas_huge_score is not None:
                header = "%s\t%s" % (header, "huge_score_gwas")
            if self.gene_to_gwas_huge_score_uncorrected is not None:
                header = "%s\t%s" % (header, "huge_score_gwas_uncorrected")
            if self.gene_to_exomes_huge_score is not None:
                header = "%s\t%s" % (header, "huge_score_exomes")
            if self.gene_to_positive_controls is not None:
                header = "%s\t%s" % (header, "positive_control")
            if self.Y is not None:
                header = "%s\t%s" % (header, "log_bf")
            if write_regression:
                header = "%s\t%s" % (header, "log_bf_regression")
            if self.Y_w is not None:
                header = "%s\t%s" % (header, "log_bf_w")
            if self.Y_fw is not None:
                header = "%s\t%s" % (header, "log_bf_fw")
            if self.priors_orig is not None:
                header = "%s\t%s" % (header, "prior_orig")
            if self.priors_adj_orig is not None:
                header = "%s\t%s" % (header, "prior_adj_orig")
            if self.batches is not None:
                header = "%s\t%s" % (header, "batch")
            if self.X_orig is not None:
                header = "%s\t%s" % (header, "N")            
            if self.gene_to_chrom is not None:
                header = "%s\t%s" % (header, "Chrom")
            if self.gene_to_pos is not None:
                header = "%s\t%s\t%s" % (header, "Start", "End")

            if self.huge_gene_covariate_zs is not None:
                header = "%s\t%s" % (header, "\t".join(map(lambda x: "avg_%s" % x, [self.huge_gene_covariate_names[i] for i in range(len(self.huge_gene_covariate_names)) if i != self.huge_gene_covariate_intercept_index])))

            output_fh.write("%s\n" % header)

            ordered_i = range(len(self.genes))
            if self.combined_prior_Ys is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.combined_prior_Ys[k])
            elif self.priors is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.priors[k])
            elif self.Y is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.Y[k])
            elif write_regression:
                ordered_i = sorted(ordered_i, key=lambda k: -self.Y_for_regression[k])

            gene_N = self.get_gene_N()
            for i in ordered_i:
                gene = genes[i]
                line = gene
                if self.priors is not None:
                    line = "%s\t%.3g" % (line, self.priors[i])
                if self.priors_adj is not None:
                    line = "%s\t%.3g" % (line, self.priors_adj[i])
                if self.combined_prior_Ys is not None:
                    line = "%s\t%.3g" % (line, self.combined_prior_Ys[i])
                if self.combined_prior_Ys_adj is not None:
                    line = "%s\t%.3g" % (line, self.combined_prior_Ys_adj[i])
                if self.combined_prior_Y_ses is not None:
                    line = "%s\t%.3g" % (line, self.combined_prior_Y_ses[i])
                if self.combined_Ds is not None:
                    line = "%s\t%.3g" % (line, self.combined_Ds[i])
                if self.gene_to_huge_score is not None:
                    if gene in self.gene_to_huge_score:
                        line = "%s\t%.3g" % (line, self.gene_to_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_gwas_huge_score is not None:
                    if gene in self.gene_to_gwas_huge_score:
                        line = "%s\t%.3g" % (line, self.gene_to_gwas_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_gwas_huge_score_uncorrected is not None:
                    if gene in self.gene_to_gwas_huge_score_uncorrected:
                        line = "%s\t%.3g" % (line, self.gene_to_gwas_huge_score_uncorrected[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_exomes_huge_score is not None:
                    if gene in self.gene_to_exomes_huge_score:
                        line = "%s\t%.3g" % (line, self.gene_to_exomes_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_positive_controls is not None:
                    if gene in self.gene_to_positive_controls:
                        line = "%s\t%.3g" % (line, self.gene_to_positive_controls[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.Y is not None:
                    line = "%s\t%.3g" % (line, self.Y[i])
                if write_regression:
                    line = "%s\t%.3g" % (line, self.Y_for_regression[i])
                if self.Y_w is not None:
                    line = "%s\t%.3g" % (line, self.Y_w[i])
                if self.Y_fw is not None:
                    line = "%s\t%.3g" % (line, self.Y_fw[i])
                if self.priors_orig is not None:
                    line = "%s\t%.3g" % (line, self.priors_orig[i])
                if self.priors_adj_orig is not None:
                    line = "%s\t%.3g" % (line, self.priors_adj_orig[i])
                if self.batches is not None:
                    line = "%s\t%s" % (line, self.batches[i])
                if self.X_orig is not None:
                    line = "%s\t%d" % (line, gene_N[i])
                if self.gene_to_chrom is not None:
                    line = "%s\t%s" % (line, self.gene_to_chrom[gene] if gene in self.gene_to_chrom else "NA")
                if self.gene_to_pos is not None:
                    line = "%s\t%s\t%s" % (line, self.gene_to_pos[gene][0] if gene in self.gene_to_pos else "NA", self.gene_to_pos[gene][1] if gene in self.gene_to_pos else "NA")

                if self.huge_gene_covariate_zs is not None:
                    line = "%s\t%s" % (line, "\t".join(map(lambda x: "%.3g" % x, [self.huge_gene_covariate_zs[i,j] for j in range(len(self.huge_gene_covariate_names)) if j != self.huge_gene_covariate_intercept_index])))

                output_fh.write("%s\n" % line)

            if self.genes_missing is not None:
                gene_N_missing = self.get_gene_N(get_missing=True)

                for i in range(len(self.genes_missing)):
                    gene = self.genes_missing[i]
                    line = gene
                    if self.priors is not None:
                        line = ("%s\t%.3g" % (line, self.priors_missing[i])) if self.priors_missing is not None else ("%s\t%s" % (line, "NA"))
                    if self.priors_adj is not None:
                        line = ("%s\t%.3g" % (line, self.priors_adj_missing[i])) if self.priors_adj_missing is not None else ("%s\t%s" % (line, "NA"))
                    if self.combined_prior_Ys is not None:
                        #has no Y of itself so its combined is just the prior
                        line = "%s\t%.3g" % (line, self.priors_missing[i])
                    if self.combined_prior_Ys_adj is not None:
                        #has no Y of itself so its combined is just the prior
                        line = "%s\t%.3g" % (line, self.priors_adj_missing[i])
                    if self.combined_prior_Y_ses is not None:
                        line = "%s\t%s" % (line, "NA")
                    if self.combined_Ds_missing is not None:
                        line = "%s\t%.3g" % (line, self.combined_Ds_missing[i])
                    if self.gene_to_huge_score is not None:
                        if gene in self.gene_to_huge_score:
                            line = "%s\t%.3g" % (line, self.gene_to_huge_score[gene])
                        else:
                            line = "%s\t%s" % (line, "NA")
                    if self.gene_to_gwas_huge_score is not None:
                        if gene in self.gene_to_gwas_huge_score:
                            line = "%s\t%.3g" % (line, self.gene_to_gwas_huge_score[gene])
                        else:
                            line = "%s\t%s" % (line, "NA")
                    if self.gene_to_gwas_huge_score_uncorrected is not None:
                        if gene in self.gene_to_gwas_huge_score_uncorrected:
                            line = "%s\t%.3g" % (line, self.gene_to_gwas_huge_score_uncorrected[gene])
                        else:
                            line = "%s\t%s" % (line, "NA")
                    if self.gene_to_exomes_huge_score is not None:
                        if gene in self.gene_to_exomes_huge_score:
                            line = "%s\t%.3g" % (line, self.gene_to_exomes_huge_score[gene])
                        else:
                            line = "%s\t%s" % (line, "NA")
                    if self.gene_to_positive_controls is not None:
                        if gene in self.gene_to_positive_controls:
                            line = "%s\t%.3g" % (line, self.gene_to_positive_controls[gene])
                        else:
                            line = "%s\t%s" % (line, "NA")
                    if self.Y is not None:
                        line = "%s\t%s" % (line, "NA")
                    if write_regression:
                        line = "%s\t%s" % (line, "NA")
                    if self.Y_w is not None:
                        line = "%s\t%s" % (line, "NA")
                    if self.Y_fw is not None:
                        line = "%s\t%s" % (line, "NA")
                    if self.priors_orig is not None:
                        line = ("%s\t%.3g" % (line, self.priors_missing_orig[i])) if self.priors_missing_orig is not None else ("%s\t%s" % (line, "NA"))

                    if self.priors_adj_orig is not None:
                        line = ("%s\t%.3g" % (line, self.priors_adj_missing_orig[i])) if self.priors_adj_missing_orig is not None else ("%s\t%s" % (line, "NA"))
                    if self.batches is not None:
                        line = "%s\t%s" % (line, "NA")
                    if self.X_orig is not None:
                        line = "%s\t%d" % (line, gene_N_missing[i])
                    if self.gene_to_chrom is not None:
                        line = "%s\t%s" % (line, self.gene_to_chrom[gene] if gene in self.gene_to_chrom else "NA")
                    if self.gene_to_pos is not None:
                        line = "%s\t%s\t%s" % (line, self.gene_to_pos[gene][0] if gene in self.gene_to_pos else "NA", self.gene_to_pos[gene][1] if gene in self.gene_to_pos else "NA")

                    if self.huge_gene_covariate_zs is not None:
                        line = "%s\t%s" % (line, "\t".join(["NA" for j in range(len(self.huge_gene_covariate_names)) if j != self.huge_gene_covariate_intercept_index]))

                    output_fh.write("%s\n" % line)

            for i in range(len(huge_only_genes)):
                gene = huge_only_genes[i]
                line = gene
                if self.priors is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.priors_adj is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.combined_prior_Ys is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.combined_prior_Ys_adj is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.combined_prior_Y_ses is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.combined_Ds_missing is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.gene_to_huge_score is not None:
                    if gene in self.gene_to_huge_score:
                        line = "%s\t%.3g" % (line, self.gene_to_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_gwas_huge_score is not None:
                    if gene in self.gene_to_gwas_huge_score:
                        line = "%s\t%.3g" % (line, self.gene_to_gwas_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_gwas_huge_score_uncorrected is not None:
                    if gene in self.gene_to_gwas_huge_score_uncorrected:
                        line = "%s\t%.3g" % (line, self.gene_to_gwas_huge_score_uncorrected[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_exomes_huge_score is not None:
                    if gene in self.gene_to_exomes_huge_score:
                        line = "%s\t%.3g" % (line, self.gene_to_exomes_huge_score[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.gene_to_positive_controls is not None:
                    if gene in self.gene_to_positive_controls:
                        line = "%s\t%.3g" % (line, self.gene_to_positive_controls[gene])
                    else:
                        line = "%s\t%s" % (line, "NA")
                if self.Y is not None:
                    line = "%s\t%s" % (line, "NA")
                if write_regression:
                    line = "%s\t%s" % (line, "NA")
                if self.Y_w is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.Y_fw is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.priors_orig is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.priors_adj_orig is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.batches is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.X_orig is not None:
                    line = "%s\t%s" % (line, "NA")
                if self.gene_to_chrom is not None:
                    line = "%s\t%s" % (line, self.gene_to_chrom[gene] if gene in self.gene_to_chrom else "NA")
                if self.gene_to_pos is not None:
                    line = "%s\t%s\t%s" % (line, self.gene_to_pos[gene][0] if gene in self.gene_to_pos else "NA", self.gene_to_pos[gene][1] if gene in self.gene_to_pos else "NA")
                    
                if self.huge_gene_covariate_zs is not None:
                    line = "%s\t%s" % (line, "\t".join(["NA" for j in range(len(self.huge_gene_covariate_names)) if j != self.huge_gene_covariate_intercept_index]))

                output_fh.write("%s\n" % line)

    def write_gene_gene_set_statistics(self, output_file):
        log("Writing gene gene set stats to %s" % output_file, INFO)

        if self.genes is None or self.X_orig is None or self.betas is None:
            return

        if self.gene_to_gwas_huge_score is not None and self.gene_to_exomes_huge_score is not None:
            gene_to_huge_score = self.gene_to_gwas_huge_score
            huge_score_label = "huge_score_gwas"
            gene_to_huge_score2 = self.gene_to_exomes_huge_score
            huge_score2_label = "huge_score_exomes"
        else:
            gene_to_huge_score = self.gene_to_huge_score
            huge_score_label = "huge_score"
            gene_to_huge_score2 = None
            huge_score2_label = None
            if gene_to_huge_score is None:
                gene_to_huge_score = self.gene_to_gwas_huge_score
                huge_score_label = "huge_score_gwas"
            if gene_to_huge_score is None:
                gene_to_huge_score = self.gene_to_exomes_huge_score
                huge_score_label = "huge_score_exomes"

        write_regression = self.Y_for_regression is not None and self.Y is not None and np.any(~np.isclose(self.Y, self.Y_for_regression))

        with open(output_file, 'w') as output_fh:

            header = "Gene"

            if self.priors is not None:
                header = "%s\t%s" % (header, "prior")
            if self.combined_prior_Ys is not None:
                header = "%s\t%s" % (header, "combined")
            if self.Y is not None:
                header = "%s\t%s" % (header, "log_bf")
            if write_regression:
                header = "%s\t%s" % (header, "log_bf_for_regression")
            if gene_to_huge_score is not None:
                header = "%s\t%s" % (header, huge_score_label)
            if gene_to_huge_score2 is not None:
                header = "%s\t%s" % (header, huge_score2_label)

            header = "%s\t%s\t%s\t%s" % (header, "gene_set", "beta", "weight")

            output_fh.write("%s\n" % header)

            ordered_i = range(len(self.genes))
            if self.combined_prior_Ys is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.combined_prior_Ys[k])
            elif self.priors is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.priors[k])
            elif self.Y is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.Y[k])
            elif write_regression is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.Y_for_regression[k])

            for i in ordered_i:
                gene = self.genes[i]

                if np.abs(self.X_orig[i,:]).sum() == 0:
                    continue

                ordered_j = sorted(self.X_orig[i,:].nonzero()[1], key=lambda k: -self.betas[k] / self.scale_factors[k])

                for j in ordered_j:
                    if self.betas[j] == 0:
                        continue
                    line = gene
                    if self.priors is not None:
                        line = "%s\t%.3g" % (line, self.priors[i])
                    if self.combined_prior_Ys is not None:
                        line = "%s\t%.3g" % (line, self.combined_prior_Ys[i])
                    if self.Y is not None:
                        line = "%s\t%.3g" % (line, self.Y[i])
                    if write_regression:
                        line = "%s\t%.3g" % (line, self.Y_for_regression[i])
                    if gene_to_huge_score is not None:
                        huge_score = gene_to_huge_score[gene] if gene in gene_to_huge_score else 0
                        line = "%s\t%.3g" % (line, huge_score)
                    if gene_to_huge_score2 is not None:
                        huge_score2 = gene_to_huge_score2[gene] if gene in gene_to_huge_score2 else 0
                        line = "%s\t%.3g" % (line, huge_score2)


                    line = "%s\t%s\t%.3g\t%.3g" % (line, self.gene_sets[j], self.betas[j] / self.scale_factors[j], self.X_orig[i,j])
                    output_fh.write("%s\n" % line)

            ordered_i = range(len(self.genes_missing))
            if self.priors is not None:
                ordered_i = sorted(ordered_i, key=lambda k: -self.priors_missing[k])

            for i in ordered_i:
                gene = self.genes_missing[i]

                if np.abs(self.X_orig_missing_genes[i,:]).sum() == 0:
                    continue

                ordered_j = sorted(self.X_orig_missing_genes[i,:].nonzero()[1], key=lambda k: -self.betas[k] / self.scale_factors[k])

                for j in ordered_j:
                    if self.betas[j] == 0:
                        continue
                    line = gene
                    if self.priors is not None:
                        line = "%s\t%.3g" % (line, self.priors_missing[i])
                    if self.combined_prior_Ys is not None:
                        line = "%s\t%.3g" % (line, self.priors_missing[i])
                    if self.Y is not None:
                        line = "%s\t%s" % (line, "NA")
                    if write_regression:
                        line = "%s\t%s" % (line, "NA")
                    if gene_to_huge_score is not None:
                        line = "%s\t%s" % (line, "NA")
                    if gene_to_huge_score2 is not None:
                        line = "%s\t%s" % (line, "NA")

                    line = "%s\t%s\t%.3g\t%.3g" % (line, self.gene_sets[j], self.betas[j] / self.scale_factors[j], self.X_orig_missing_genes[i,j])
                    output_fh.write("%s\n" % line)


    def write_gene_set_overlap_statistics(self, output_file):
        log("Writing gene set overlap stats to %s" % output_file, INFO)
        with open(output_file, 'w') as output_fh:
            if self.gene_sets is None:
                return
            if self.X_orig is None or self.betas is None or self.betas_uncorrected is None or self.mean_shifts is None or self.scale_factors is None:
                return
            header = "Gene_Set\tbeta\tbeta_uncorrected\tGene_Set_overlap\tV_beta\tV\tbeta_overlap\tbeta_uncorrected_overlap"
            output_fh.write("%s\n" % header)

            print_mask = self.betas_uncorrected != 0
            gene_sets = [self.gene_sets[i] for i in np.where(print_mask)[0]]
            X_to_print = self.X_orig[:,print_mask]
            mean_shifts = self.mean_shifts[print_mask]
            scale_factors = self.scale_factors[print_mask]
            betas_uncorrected = self.betas_uncorrected[print_mask]
            betas = self.betas[print_mask]

            num_batches = self._get_num_X_blocks(X_to_print)

            ordered_i = sorted(range(len(gene_sets)), key=lambda k: -betas[k] / scale_factors[k])

            gene_sets = [gene_sets[i] for i in ordered_i]
            X_to_print = X_to_print[:,ordered_i]
            mean_shifts = mean_shifts[ordered_i]
            scale_factors = scale_factors[ordered_i]
            betas_uncorrected = betas_uncorrected[ordered_i]
            betas = betas[ordered_i]

            for batch in range(num_batches):
                begin = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                if end > X_to_print.shape[1]:
                    end = X_to_print.shape[1]

                X_to_print[:,begin:end]
                mean_shifts[begin:end]
                scale_factors[begin:end]

                cur_V = self._compute_V(X_to_print[:,begin:end], mean_shifts[begin:end], scale_factors[begin:end], X_orig2=X_to_print, mean_shifts2=mean_shifts, scale_factors2=scale_factors)
                cur_V_beta = cur_V * betas

                for i in range(end - begin):
                    outer_ind = int(i + batch * self.batch_size)
                    ordered_j = sorted(np.where(cur_V_beta[i,:] != 0)[0], key=lambda k: -cur_V_beta[i,k] / scale_factors[k])
                    for j in ordered_j:
                        if outer_ind == j:
                            continue
                        output_fh.write("%s\t%.3g\t%.3g\t%s\t%.3g\t%.3g\t%.3g\t%.3g\n" % (gene_sets[outer_ind], betas[outer_ind] / scale_factors[outer_ind], betas_uncorrected[outer_ind] / scale_factors[outer_ind], gene_sets[j], cur_V_beta[i, j] / scale_factors[i], cur_V[i,j], betas[j] / scale_factors[j], betas_uncorrected[j] / scale_factors[j]))


    def write_huge_gene_covariates(self, output_file):
        if self.genes is None or self.huge_gene_covariates is None:
            return

        assert(self.huge_gene_covariates.shape[1] == len(self.huge_gene_covariate_names))
        log("Writing covs to %s" % output_file, INFO)

        with open(output_file, 'w') as output_fh:

            if self.huge_gene_covariate_betas is not None:
                value_out = "#betas\tbetas"
                for j in range(self.huge_gene_covariates.shape[1]):
                    value_out += ("\t%.4g" % self.huge_gene_covariate_betas[j])
                output_fh.write("%s\n" % value_out)

            header = "%s\t%s" % ("Gene\tin_regression", "\t".join(self.huge_gene_covariate_names))
            output_fh.write("%s\n" % header)

            for i in range(len(self.genes)):
                value_out = "%s\t%s" % (self.genes[i], self.huge_gene_covariates_mask[i])
                for j in range(self.huge_gene_covariates.shape[1]):
                    value_out += ("\t%.4g" % self.huge_gene_covariates[i,j])
                output_fh.write("%s\n" % value_out)


    def write_gene_effectors(self, output_file):
        if self.genes is None or self.huge_signal_bfs is None:
            return

        assert(self.huge_signal_bfs.shape[1] == len(self.huge_signals))
        
        log("Writing gene effectors to %s" % output_file, INFO)

        if self.gene_to_gwas_huge_score is not None and self.gene_to_exomes_huge_score is not None:
            gene_to_huge_score = self.gene_to_gwas_huge_score
        else:
            gene_to_huge_score = self.gene_to_huge_score
            if gene_to_huge_score is None:
                gene_to_huge_score = self.gene_to_gwas_huge_score
            if gene_to_huge_score is None:
                gene_to_huge_score = self.gene_to_exomes_huge_score

        with open(output_file, 'w') as output_fh:

            header = "Lead_locus\tP\tGene"

            if self.combined_prior_Ys is not None:
                header = "%s\t%s" % (header, "cond_prob_total") #probability of each gene under assumption that only one is causal
            if self.Y is not None and self.priors is not None:
                header = "%s\t%s" % (header, "cond_prob_signal") #probability of each gene under assumption that only one is an effector (more than one could be causal if there are multiple SNPs each with different effectors)
            if self.priors is not None:
                header = "%s\t%s" % (header, "cond_prob_prior") #probability of each gene using only priors (assumption one causal gene)
            if gene_to_huge_score is not None:
                header = "%s\t%s" % (header, "conf_prob_huge") #probability of each gene using only distance/s2g (assumption one causal effector)
            if self.combined_Ds is not None:
                header = "%s\t%s" % (header, "combined_D")

            output_fh.write("%s\n" % header)

            for signal_ind in range(len(self.huge_signals)):

                gene_inds = self.huge_signal_bfs[:, signal_ind].nonzero()[0]

                max_log_bf = 10

                cond_prob_total = None
                if self.combined_prior_Ys is not None:
                    combined_prior_Y_bfs = self.combined_prior_Ys[gene_inds]
                    combined_prior_Y_bfs[combined_prior_Y_bfs > max_log_bf] = max_log_bf                    
                    combined_prior_Y_bfs = np.exp(combined_prior_Y_bfs)
                    cond_prob_total = combined_prior_Y_bfs / np.sum(combined_prior_Y_bfs)

                cond_prob_prior = None
                if self.priors is not None:
                    prior_bfs = self.priors[gene_inds]
                    prior_bfs[prior_bfs > max_log_bf] = max_log_bf                    
                    prior_bfs = np.exp(prior_bfs)
                    cond_prob_prior = prior_bfs / np.sum(prior_bfs)
                    
                    if self.Y is not None:
                        log_bf_bfs = self.huge_signal_bfs[:,signal_ind].todense().A1[gene_inds] * prior_bfs
                        cond_prob_log_bf = log_bf_bfs / np.sum(log_bf_bfs)

                cond_prob_huge = None
                if gene_to_huge_score is not None:
                    cond_prob_huge = self.huge_signal_bfs[:,signal_ind].todense().A1[gene_inds]
                    cond_prob_huge /= np.sum(cond_prob_huge)

                for i in range(len(gene_inds)):
                    gene_ind = gene_inds[i]
                    line = "%s:%d\t%.3g\t%s" % (self.huge_signals[signal_ind][0], self.huge_signals[signal_ind][1], self.huge_signals[signal_ind][2], self.genes[gene_ind])

                    if self.combined_prior_Ys is not None:
                        line = "%s\t%.3g" % (line, cond_prob_total[i])
                    if self.Y is not None and self.priors is not None:
                        line = "%s\t%.3g" % (line, cond_prob_log_bf[i])
                    if self.priors is not None:
                        line = "%s\t%.3g" % (line, cond_prob_prior[i])
                    if gene_to_huge_score is not None:
                        line = "%s\t%.3g" % (line, cond_prob_huge[i])
                    if self.combined_Ds is not None:
                        line = "%s\t%.3g" % (line, self.combined_Ds[gene_ind])

                    output_fh.write("%s\n" % line)


    def write_matrix_factors(self, factors_output_file=None, gene_set_factors_output_file=None, gene_factors_output_file=None, marker_factors_output_file=None):

        if self.factor_gene_set_scores is None:
            return

        ordered_inds = range(len(self.factor_gene_set_scores))

        if factors_output_file is not None:
            assert(self.exp_gene_set_factors.shape[0] == np.sum(self.gene_set_factor_gene_set_mask))
            assert(self.exp_gene_factors.shape[0] == np.sum(self.gene_factor_gene_mask))
            log("Writing factors to %s" % factors_output_file, INFO)
            with open(factors_output_file, 'w') as output_fh:
                header = "Factor"
                header = "%s\t%s" % (header, "label")
                header = "%s\t%s" % (header, "gene_set_score")
                header = "%s\t%s" % (header, "gene_score")
                header = "%s\t%s" % (header, "top_genes")
                header = "%s\t%s" % (header, "top_gene_sets")
                output_fh.write("%s\n" % (header))
                    
                for i in ordered_inds:
                    line = "Factor%d" % (i+1)
                    line = "%s\t%s" % (line, self.factor_labels[i])
                    line = "%s\t%.3g" % (line, self.factor_gene_set_scores[i])
                    line = "%s\t%.3g" % (line, self.factor_gene_scores[i])
                    line = "%s\t%s" % (line, ",".join(self.top_genes[i]))
                    line = "%s\t%s" % (line, ",".join(self.top_gene_sets[i]))
                    output_fh.write("%s\n" % (line))

        if gene_set_factors_output_file is not None and self.exp_gene_set_factors is not None and self.gene_set_factor_gene_set_mask is not None:
            assert(self.exp_gene_set_factors.shape[0] == np.sum(self.gene_set_factor_gene_set_mask))
            log("Writing gene set factors to %s" % gene_set_factors_output_file, INFO)
            with open(gene_set_factors_output_file, 'w') as output_fh:
                header = "Gene_Set"
                if self.betas is not None:
                    header = "%s\t%s" % (header, "beta")
                if self.betas_uncorrected is not None:
                    header = "%s\t%s" % (header, "beta_uncorrected")
                output_fh.write("%s\t%s\n" % (header, "\t".join(["Factor%d" % (i+1) for i in ordered_inds])))

                gene_set_factor_gene_set_inds = np.where(self.gene_set_factor_gene_set_mask)[0]
                for i in range(self.exp_gene_set_factors.shape[0]):
                    orig_i = gene_set_factor_gene_set_inds[i]
                    line = self.gene_sets[orig_i]
                    if self.betas is not None:
                        line = "%s\t%.3g" % (line, self.betas[orig_i])
                    if self.betas_uncorrected is not None:
                        line = "%s\t%.3g" % (line, self.betas_uncorrected[orig_i])

                    output_fh.write("%s\t%s\n" % (line, "\t".join(["%.4g" % (self.exp_gene_set_factors[i,j]) for j in ordered_inds])))

        if gene_factors_output_file is not None and self.exp_gene_factors is not None and self.gene_factor_gene_mask is not None:
            assert(self.exp_gene_factors.shape[0] == np.sum(self.gene_factor_gene_mask))
            log("Writing gene factors to %s" % gene_factors_output_file, INFO)
            with open(gene_factors_output_file, 'w') as output_fh:
                header = "Gene"
                if self.combined_prior_Ys is not None:
                    header = "%s\t%s" % (header, "combined")
                if self.Y is not None:
                    header = "%s\t%s" % (header, "log_bf")
                if self.priors is not None:
                    header = "%s\t%s" % (header, "prior")

                output_fh.write("%s\t%s\n" % (header, "\t".join(["Factor%d" % (i+1) for i in ordered_inds])))

                gene_factor_gene_inds = np.where(self.gene_factor_gene_mask)[0]
                for i in range(self.exp_gene_factors.shape[0]):
                    orig_i = gene_factor_gene_inds[i]
                    line = self.genes[orig_i]
                    if self.combined_prior_Ys is not None:
                        line = "%s\t%.3g" % (line, self.combined_prior_Ys[orig_i])
                    if self.Y is not None:
                        line = "%s\t%.3g" % (line, self.Y[orig_i])
                    if self.priors is not None:
                        line = "%s\t%.3g" % (line, self.priors[orig_i])

                    output_fh.write("%s\t%s\n" % (line, "\t".join(["%.4g" % (self.exp_gene_factors[i,j]) for j in ordered_inds])))

        if marker_factors_output_file is not None and self.exp_gene_factors is not None and self.gene_factor_gene_mask is not None and self.exp_gene_set_factors is not None and self.gene_set_factor_gene_set_mask is not None:
            assert(self.exp_gene_factors.shape[0] == np.sum(self.gene_factor_gene_mask))            
            assert(self.exp_gene_set_factors.shape[0] == np.sum(self.gene_set_factor_gene_set_mask))
            log("Writing marker factors to %s" % marker_factors_output_file, INFO)
            #order all of the gene and gene set factors
            top_gene_set_factors = np.argsort(-self.exp_gene_set_factors, axis=0)
            max_gene_set_factors = np.max(self.exp_gene_set_factors, axis=0)
            max_gene_set_factors[max_gene_set_factors == 0] = 1
            top_gene_factors = np.argsort(-self.exp_gene_factors, axis=0)
            max_gene_factors = np.max(self.exp_gene_factors, axis=0)
            max_gene_factors[max_gene_factors == 0] = 1
            
            with open(marker_factors_output_file, 'w') as output_fh:
                output_fh.write("%s\n" % ("\t".join(["Factor%d_top_gene_sets\tFactor%d_gene_set_value\tFactor%d_top_genes\tF%d_gene_val" % (i+1, i+1, i+1, i+1) for i in ordered_inds])))

                gene_factor_gene_inds = np.where(self.gene_factor_gene_mask)[0]
                gene_set_factor_gene_set_inds = np.where(self.gene_set_factor_gene_set_mask)[0]

                for i in range(max(self.exp_gene_set_factors.shape[0], self.exp_gene_factors.shape[0])):
                    output_fh.write("%s\n" % ("\t".join(["%s\t%s\t%s\t%s" % (self.gene_sets[gene_set_factor_gene_set_inds[top_gene_set_factors[i,j]]] if i < len(top_gene_set_factors[:,j]) else "", "%.4g" % (self.exp_gene_set_factors[top_gene_set_factors[i,j], j] / max_gene_set_factors[j]) if i < len(top_gene_set_factors[:,j]) else "", self.genes[gene_factor_gene_inds[top_gene_factors[i,j]]] if i < len(top_gene_factors[:,j]) else "", "%.4g" % (self.exp_gene_factors[top_gene_factors[i,j], j] / max_gene_factors[j]) if i < len(top_gene_factors[:,j]) else "") for j in ordered_inds])))

    def write_clusters(self, gene_set_clusters_output_file=None, gene_clusters_output_file=None):

        if self.exp_lambdak is None:
            return

        ordered_inds = range(len(self.factor_gene_set_scores))

        if gene_set_clusters_output_file is not None and self.exp_gene_set_factors is not None and self.gene_set_factor_gene_set_mask is not None:
            assert(self.exp_gene_set_factors.shape[0] == np.sum(self.gene_set_factor_gene_set_mask))
            
            #this uses value relative to others in the cluster
            #values_for_cluster = self.exp_gene_set_factors / np.sum(self.exp_gene_set_factors, axis=0)
            #this uses strongest absolute value
            values_for_cluster = self.exp_gene_set_factors

            log("Writing gene set clusters to %s" % gene_set_clusters_output_file, INFO)
            with open(gene_set_clusters_output_file, 'w') as output_fh:
                gene_set_factor_gene_set_inds = np.where(self.gene_set_factor_gene_set_mask)[0]
                header = "Gene_Set"
                key_fn = None
                if self.betas is not None:
                    header = "%s\t%s" % (header, "beta")
                    key_fn = lambda k: -self.betas[gene_set_factor_gene_set_inds[k]]
                if self.betas_uncorrected is not None:
                    header = "%s\t%s" % (header, "beta_uncorrected")
                    if key_fn is None:
                        key_fn = lambda k: -self.betas_uncorrected_[gene_set_factor_gene_set_inds[k]]

                if key_fn is None:
                    key_fn = lambda k: k

                output_fh.write("%s\t%s\t%s\t%s\n" % (header, "cluster", "label", "\t".join(["Factor%d" % (i+1) for i in ordered_inds])))

                for i in sorted(range(values_for_cluster.shape[0]), key=key_fn):
                    if np.sum(self.exp_gene_set_factors[i,:]) == 0:
                        continue
                    orig_i = gene_set_factor_gene_set_inds[i]
                    line = self.gene_sets[orig_i]
                    if self.betas is not None:
                        line = "%s\t%.3g" % (line, self.betas[orig_i])
                    if self.betas_uncorrected is not None:
                        line = "%s\t%.3g" % (line, self.betas_uncorrected[orig_i])
                    cluster = np.argmax(values_for_cluster[i,:])
                    output_fh.write("%s\tFactor%d\t%s\t%s\n" % (line, cluster + 1, self.factor_labels[cluster], "\t".join(["%.4g" % (self.exp_gene_set_factors[i,j]) for j in ordered_inds])))

        if gene_clusters_output_file is not None and self.exp_gene_factors is not None and self.gene_factor_gene_mask is not None:
            assert(self.exp_gene_factors.shape[0] == np.sum(self.gene_factor_gene_mask))

            #this uses value relative to others in the cluster
            values_for_cluster = self.exp_gene_factors / np.sum(self.exp_gene_factors, axis=0)
            #this uses strongest absolute value
            values_for_cluster = self.exp_gene_factors

            log("Writing gene clusters  to %s" % gene_clusters_output_file, INFO)
            with open(gene_clusters_output_file, 'w') as output_fh:
                gene_factor_gene_inds = np.where(self.gene_factor_gene_mask)[0]
                header = "Gene"
                key_fn = None
                if self.combined_prior_Ys is not None:
                    header = "%s\t%s" % (header, "combined")
                    key_fn = lambda k: -self.combined_prior_Ys[gene_factor_gene_inds[k]]
                if self.Y is not None:
                    header = "%s\t%s" % (header, "log_bf")
                    if key_fn is None:
                        key_fn = lambda k: -self.Y[gene_factor_gene_inds[k]]
                if self.priors is not None:
                    header = "%s\t%s" % (header, "prior")
                    if key_fn is None:
                        key_fn = lambda k: -self.prior[gene_factor_gene_inds[k]]
                if key_fn is None:
                    key_fn = lambda k: k

                output_fh.write("%s\t%s\t%s\t%s\n" % (header, "cluster", "label", "\t".join(["Factor%d" % (i+1) for i in ordered_inds])))

                gene_set_factor_gene_inds = np.where(self.gene_factor_gene_mask)[0]
                for i in sorted(range(values_for_cluster.shape[0]), key=key_fn):
                    if np.sum(self.exp_gene_factors[i,:]) == 0:
                        continue

                    orig_i = gene_set_factor_gene_inds[i]
                    line = self.genes[orig_i]
                    if self.combined_prior_Ys is not None:
                        line = "%s\t%.3g" % (line, self.combined_prior_Ys[orig_i])
                    if self.Y is not None:
                        line = "%s\t%.3g" % (line, self.Y[orig_i])
                    if self.priors is not None:
                        line = "%s\t%.3g" % (line, self.priors[orig_i])
                    cluster = np.argmax(values_for_cluster[i,:])
                    output_fh.write("%s\tFactor%d\t%s\t%s\n" % (line, cluster + 1, self.factor_labels[cluster], "\t".join(["%.4g" % (self.exp_gene_factors[i,j]) for j in ordered_inds])))

    #HELPER FUNCTIONS

    '''
    Read in gene bfs for LOGISTIC or EMPIRICAL mapping
    '''
    def _record_param(self, param, value, overwrite=False, record_only_first_time=False):
        if param not in self.params:
            self.param_keys.append(param)
            self.params[param] = value
        elif record_only_first_time:
            return
        elif type(self.params[param]) == list:
            if self.params[param][-1] != value:
                self.params[param].append(value)
        elif self.params[param] != value:
            if overwrite:
                self.params[param] = value
            else:
                self.params[param] = [self.params[param], value]

    def _record_params(self, params, overwrite=False, record_only_first_time=False):
        for param in params:
            if params[param] is not None:
                self._record_param(param, params[param], overwrite=overwrite, record_only_first_time=record_only_first_time)

    def _read_gene_bfs(self, gene_bfs_in, gene_bfs_id_col=None, gene_bfs_log_bf_col=None, gene_bfs_combined_col=None, gene_bfs_prior_col=None, gene_bfs_sd_col=None, **kwargs):

        #require X matrix

        if gene_bfs_in is None:
            bail("Require --gene-bfs-in for this operation")

        log("Reading --gene-bfs-in file %s" % gene_bfs_in, INFO)
        gene_in_bfs = {}
        gene_in_combined = None
        gene_in_priors = None
        with open(gene_bfs_in) as gene_bfs_fh:
            header_cols = gene_bfs_fh.readline().strip().split()
            if gene_bfs_id_col is None:
                gene_bfs_id_col = "Gene"

            id_col = self._get_col(gene_bfs_id_col, header_cols)

            if gene_bfs_log_bf_col is not None:
                bf_col = self._get_col(gene_bfs_log_bf_col, header_cols)
            else:
                bf_col = self._get_col("log_bf", header_cols)

            if bf_col is None:
                bail("--gene-bfs-bf-col required for this operation")

            combined_col = None
            if gene_bfs_combined_col is not None:
                combined_col = self._get_col(gene_bfs_combined_col, header_cols, True)
            else:
                combined_col = self._get_col("combined", header_cols, False)

            prior_col = None
            if gene_bfs_prior_col is not None:
                prior_col = self._get_col(gene_bfs_prior_col, header_cols, True)
            else:
                prior_col = self._get_col("prior", header_cols, False)

            if combined_col is not None:
                gene_in_combined = {}
            if prior_col is not None:
                gene_in_priors = {}

            for line in gene_bfs_fh:
                cols = line.strip().split()
                if id_col >= len(cols) or bf_col >= len(cols) or (combined_col is not None and combined_col >= len(cols)) or (prior_col is not None and prior_col >= len(cols)):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[id_col]
                
                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]

                try:
                    bf = float(cols[bf_col])
                except ValueError:
                    if not cols[bf_col] == "NA":
                        warn("Skipping unconvertible value %s for gene_set %s" % (cols[bf_col], gene))
                    continue
                gene_in_bfs[gene] = bf

                if combined_col is not None:
                    try:
                        combined = float(cols[combined_col])
                    except ValueError:
                        if not cols[combined_col] == "NA":
                            warn("Skipping unconvertible value %s for gene_set %s" % (cols[combined_col], gene))
                        continue
                    gene_in_combined[gene] = combined

                if prior_col is not None:
                    try:
                        prior = float(cols[prior_col])
                    except ValueError:
                        if not cols[prior_col] == "NA":
                            warn("Skipping unconvertible value %s for gene_set %s" % (cols[prior_col], gene))
                        continue
                    gene_in_priors[gene] = prior


        if self.genes is not None:
            genes = self.genes
            gene_to_ind = self.gene_to_ind
        else:
            genes = []
            gene_to_ind = {}

        gene_bfs = np.array([np.nan] * len(genes))
        
        extra_gene_bfs = []
        extra_genes = []
        for gene in gene_in_bfs:
            bf = gene_in_bfs[gene]
            if gene in gene_to_ind:
                gene_bfs[gene_to_ind[gene]] = bf
            else:
                extra_gene_bfs.append(bf)
                extra_genes.append(gene)

        return (gene_bfs, extra_genes, np.array(extra_gene_bfs), gene_in_combined, gene_in_priors)

    '''
    Read in gene Z scores for linear mapping
    '''
    def _read_gene_zs(self, gene_zs_in, gene_zs_id_col=None, gene_zs_value_col=None, background_95_prior=None, gws_threshold=2.5e-6, gws_prob_true=0.95, max_mean_posterior=0.2, max_95_posterior=None, **kwargs):

        if gene_zs_in is None:
            bail("Require --gene-zs-in for this operation")

        log("Reading --gene-zs-in file %s" % gene_zs_in, INFO)
        with open(gene_zs_in) as gene_zs_fh:
            header_cols = gene_zs_fh.readline().strip().split()
            if gene_zs_id_col is None:
                bail("--gene-zs-id-col required for this operation")
            if gene_zs_value_col is None:
                bail("--gene-zs-value-col required for this operation")
            id_col = self._get_col(gene_zs_id_col, header_cols)
            value_col = self._get_col(gene_zs_value_col, header_cols)
            gene_zs = {}
            line_num_to_gene = []
            line_num = 0
            for line in gene_zs_fh:
                cols = line.strip().split()
                if id_col > len(cols) or value_col > len(cols):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[id_col]
                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]

                line_num_to_gene.append(gene)
                try:
                    value = float(cols[value_col])
                except ValueError:
                    if not cols[value_col] == "NA":
                        warn("Skipping unconvertible beta_tilde value %s for gene_set %s" % (cols[value_col], gene))
                    continue
                gene_zs[gene] = np.abs(value)
                line_num += 1

        #top posterior specifies what we want the posterior of the top ranked gene to actually be
        gws_z_threshold = -scipy.stats.norm.ppf(gws_threshold/2)
        median_z = np.median(np.array(list(gene_zs.values())))

        #logistic = L/(1 + exp(-k*(x-xo)))
        #Need:
        #L = asymptote (max allowed is 20%?)
        #then need two points to constrain it
        #average over all genes equals 0.05
        #gws threshold (how is this different from max?)
        #k = (log(1/y2 - L) - log(1/y1 - 1))/(x2-x1)
        #xo = (x1 * log(1/y2 - L) - x2 * log(1/y1 - L)) / (log(1/y2 - L) - log(1/y1 - L))

        x1 = median_z
        y1 = self.background_prior
        y1_bf = np.log(y1 / (1 - y1))

        x2 = gws_z_threshold
        y2 = gws_prob_true * max_mean_posterior
        y2_bf = np.log(y2 / (1 - y2))

        log("Fitting prop true logistic model with max bf=%.3g, points (%.3g,%.3g) and (%.3g,%.3g)" % (max_mean_posterior, x1, y1, x2, y2))

        L_param = max_mean_posterior
        k_param = (np.log(y2 / (L_param - y2)) - np.log(y1 / (L_param - y1))) / (x2 - x1)
        x_o_param = np.log((L_param - y2) / y2) / k_param + x2

        log("Using L=%.3g, k=%.3g, x_o=%.3g for logistic model of BF(Z)" % (L_param, k_param, x_o_param))

        if self.genes is not None:
            genes = self.genes
            gene_to_ind = self.gene_to_ind
        else:
            genes = []
            gene_to_ind = {}

        gene_bfs = np.array([np.nan] * len(genes))
        extra_gene_bfs = []
        extra_genes = []
        #determine scale factor
        for gene in gene_zs:
            posterior = L_param / (1 + np.exp(-k_param * (gene_zs[gene] - x_o_param)))
            bf = np.log(posterior / (1.0 - posterior)) - self.background_log_bf

            #print(gene_zs[gene],posterior,bf)

            if gene in gene_to_ind:
                gene_bfs[gene_to_ind[gene]] = bf
            else:
                extra_gene_bfs.append(bf)
                extra_genes.append(gene)


        extra_gene_bfs = np.array(extra_gene_bfs)

        #if len(gene_bfs) == 0:
        #    bail("No genes in gene sets had percentiles!")

        return (gene_bfs, extra_genes, extra_gene_bfs)

    '''
    Read in gene percentiles for INVERSE_NORMALIZE mapping
    '''
    def _read_gene_percentiles(self, gene_percentiles_in, gene_percentiles_id_col=None, gene_percentiles_value_col=None, gene_percentiles_higher_is_better=False, top_posterior=0.99, min_prob=1e-4, max_prob=1-1e-4, **kwargs):

        if gene_percentiles_in is None:
            bail("Require --gene-percentiles-in for this operation")

        #top posterior specifies what we want the posterior of the top ranked gene to actually be
        top_log_pos_odd=np.log(top_posterior / (1-top_posterior))

        log("Reading --gene-percentiles-in file %s" % gene_percentiles_in, INFO)
        with open(gene_percentiles_in) as gene_percentiles_fh:
            header_cols = gene_percentiles_fh.readline().strip().split()
            if gene_percentiles_id_col is None:
                bail("--gene-percentiles-id-col required for this operation")
            if gene_percentiles_value_col is None:
                bail("--gene-percentiles-value-col required for this operation")
            id_col = self._get_col(gene_percentiles_id_col, header_cols)
            value_col = self._get_col(gene_percentiles_value_col, header_cols)
            gene_percentiles = {}
            line_num_to_gene = []
            line_num = 0
            for line in gene_percentiles_fh:
                cols = line.strip().split()
                if id_col > len(cols) or value_col > len(cols):
                    warn("Skipping due to too few columns in line: %s" % line)
                    continue

                gene = cols[id_col]
                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]

                line_num_to_gene.append(gene)
                try:
                    value = float(cols[value_col])
                except ValueError:
                    if not cols[value_col] == "NA":
                        warn("Skipping unconvertible beta_tilde value %s for gene_set %s" % (cols[value_col], gene))
                    continue
                gene_percentiles[gene] = value
                line_num += 1

            sorted_gene_percentiles = sorted(gene_percentiles.keys(), key=lambda x: gene_percentiles[x], reverse=gene_percentiles_higher_is_better)
            scale_factor=(top_log_pos_odd - self.background_log_bf)/scipy.stats.norm.ppf(float(len(sorted_gene_percentiles))/(len(sorted_gene_percentiles)+1))
            log("Using mean=%.3g, scale=%.3g for inverse normalized model of prob true" % (self.background_log_bf, scale_factor))

            #first gene is best
            for i in range(len(sorted_gene_percentiles)):
                gene_percentiles[sorted_gene_percentiles[i]] = 1 - float(i+1) / (len(sorted_gene_percentiles)+1)

            if self.genes is not None:
                genes = self.genes
                gene_to_ind = self.gene_to_ind
            else:
                genes = []
                gene_to_ind = {}

            gene_bf = np.array([np.nan] * len(genes))
            extra_gene_bf = []
            extra_genes = []
            #determine scale factor
            for gene in gene_percentiles:
                bf = scipy.stats.norm.ppf(gene_percentiles[gene], loc=self.background_log_bf, scale=scale_factor)
                if gene in gene_to_ind:
                    gene_bf[gene_to_ind[gene]] = bf
                else:
                    extra_gene_bf.append(bf)
                    extra_genes.append(gene)
            extra_gene_bf = np.array(extra_gene_bf)

        if len(gene_bf) == 0:
            bail("No genes in gene sets had percentiles!")

        return (gene_bf, extra_genes, extra_gene_bf)

    def convert_prior_to_var(self, top_prior, num):
        top_bf = np.log((top_prior) / (1 - top_prior)) - self.background_log_bf 
        var = np.square(top_bf / (-scipy.stats.norm.ppf(1.0 / num)))

        return (var)

    def _determine_columns(self, filename):
        #try to determine columns for gene_id, var_id, chrom, pos, p, beta, se, freq, n

        log("Trying to determine columns from headers and data for %s..." % filename)
        header = None
        with open_gz(filename) as fh:
            header = fh.readline().strip()
            orig_header_cols = header.split()

            header_cols = [x.strip('"').strip("'").strip() for x in orig_header_cols]

            def __get_possible_from_headers(header_cols, possible_headers1, possible_headers2=None):
                possible = np.full(len(header_cols), False)
                possible_inds = [i for i in range(len(header_cols)) if header_cols[i].lower() in possible_headers1]
                if len(possible_inds) == 0 and possible_headers2 is not None:
                    possible_inds = [i for i in range(len(header_cols)) if header_cols[i].lower() in possible_headers2]
                possible[possible_inds] = True
                return possible

            possible_gene_id_headers = set(["gene","id"])
            possible_var_id_headers = set(["var","id","rs", "varid"])
            possible_chrom_headers = set(["chr", "chrom", "chromosome"])
            possible_pos_headers = set(["pos", "bp", "position", "base_pair_location"])
            possible_locus_headers = set(["variant"])
            possible_p_headers = set(["p-val", "p_val", "pval", "p.value", "p-value", "p_value"])
            possible_p_headers2 = set(["p"])
            possible_beta_headers = set(["beta","effect"])
            possible_se_headers = set(["se","std", "stderr", "standard_error"])
            possible_freq_headers = set(["maf","freq"])
            possible_freq_headers2 = set(["af", "effect_allele_frequency"])
            possible_n_headers = set(["sample", "neff", "TotalSampleSize"])
            possible_n_headers2 = set(["n"])

            possible_gene_id_cols = __get_possible_from_headers(header_cols, possible_gene_id_headers)
            possible_var_id_cols = __get_possible_from_headers(header_cols, possible_var_id_headers)
            possible_chrom_cols = __get_possible_from_headers(header_cols, possible_chrom_headers)
            possible_locus_cols = __get_possible_from_headers(header_cols, possible_locus_headers)
            possible_pos_cols = __get_possible_from_headers(header_cols, possible_pos_headers)
            possible_p_cols = __get_possible_from_headers(header_cols, possible_p_headers, possible_p_headers2)
            possible_beta_cols = __get_possible_from_headers(header_cols, possible_beta_headers)
            possible_se_cols = __get_possible_from_headers(header_cols, possible_se_headers)
            possible_freq_cols = __get_possible_from_headers(header_cols, possible_freq_headers, possible_freq_headers2)
            possible_n_cols = __get_possible_from_headers(header_cols, possible_n_headers, possible_n_headers2)

            missing_vals = set(["", ".", "-", "na"])
            num_read = 0
            max_to_read = 1000

            for line in fh:
                cols = line.strip().split()
                seen_non_missing = False
                for i in range(len(cols)):
                    token = cols[i].lower()

                    if token.lower() in missing_vals:
                        continue

                    seen_non_missing = True

                    if possible_gene_id_cols[i]:
                        try:
                            val = float(cols[i])
                            if not int(val) == val:
                                possible_gene_id_cols[i] = False
                        except ValueError:
                            pass
                    if possible_var_id_cols[i]:
                        if len(token) < 4:
                            possible_var_id_cols[i] = False

                        if "chr" in token or ":" in token or "rs" in token or "_" in token or "-" in token or "var" in token:
                            pass
                        else:
                            possible_var_id_cols[i] = False
                    if possible_chrom_cols[i]:
                        if "chr" in token or "x" in token or "y" in token or "m" in token:
                            pass
                        else:
                            try:
                                val = int(cols[i])
                                if val < 1 or val > 26:
                                    possible_chrom_cols[i] = False
                            except ValueError:
                                possible_chrom_cols[i] = False
                    if possible_locus_cols[i]:
                        if "chr" in token or "x" in token or "y" in token or "m" in token:
                            pass
                        else:
                            try:
                                locus = None
                                for delim in [":", "_"]:
                                    if delim in cols[i]:
                                        locus = cols[i].split(delim)
                                if locus is not None and len(locus) >= 2:
                                    chrom = int(locus[0])
                                    pos = int(locus[1])
                                    if chrom < 1 or chrom > 26:
                                        possible_locus_cols[i] = False
                            except ValueError:
                                possible_locus_cols[i] = False
                    if possible_pos_cols[i]:
                        try:
                            if len(token) < 4:
                                possible_pos_cols[i] = False
                            val = float(cols[i])
                            if not int(val) == val:
                                possible_pos_cols[i] = False
                        except ValueError:
                            possible_pos_cols[i] = False

                    if possible_p_cols[i]:
                        try:
                            val = float(cols[i])
                            if val > 1 or val < 0:
                                possible_p_cols[i] = False
                        except ValueError:
                            
                            possible_p_cols[i] = False
                    if possible_beta_cols[i]:
                        try:
                            val = float(cols[i])
                        except ValueError:
                            possible_beta_cols[i] = False
                    if possible_se_cols[i]:
                        try:
                            val = float(cols[i])
                            if val < 0:
                                possible_se_cols[i] = False
                        except ValueError:
                            possible_se_cols[i] = False
                    if possible_freq_cols[i]:
                        try:
                            val = float(cols[i])
                            if val > 1 or val < 0:
                                possible_freq_cols[i] = False
                        except ValueError:
                            possible_freq_cols[i] = False
                    if possible_n_cols[i]:
                        if len(token) < 3:
                            possible_n_cols[i] = False
                        else:
                            try:
                                val = float(cols[i])
                                if val < 0:
                                    possible_n_cols[i] = False
                            except ValueError:
                                possible_n_cols[i] = False
                if seen_non_missing:
                    num_read += 1
                    if num_read >= max_to_read:
                        break
                    
        possible_beta_cols[possible_p_cols] = False
        possible_beta_cols[possible_se_cols] = False
        possible_beta_cols[possible_pos_cols] = False

        total_possible = possible_gene_id_cols.astype(int) + possible_var_id_cols.astype(int) + possible_chrom_cols.astype(int) + possible_pos_cols.astype(int) + possible_p_cols.astype(int) + possible_beta_cols.astype(int) + possible_se_cols.astype(int) + possible_freq_cols.astype(int) + possible_n_cols.astype(int)
        for possible_cols in [possible_gene_id_cols, possible_var_id_cols, possible_chrom_cols, possible_pos_cols, possible_p_cols, possible_beta_cols, possible_se_cols, possible_freq_cols, possible_n_cols]:
            possible_cols[total_possible > 1] = False

        orig_header_cols = np.array(orig_header_cols)

        return (orig_header_cols[possible_gene_id_cols], orig_header_cols[possible_var_id_cols], orig_header_cols[possible_chrom_cols], orig_header_cols[possible_pos_cols], orig_header_cols[possible_locus_cols], orig_header_cols[possible_p_cols], orig_header_cols[possible_beta_cols], orig_header_cols[possible_se_cols], orig_header_cols[possible_freq_cols], orig_header_cols[possible_n_cols], header)

    def _adjust_bf(self, Y, min_mean_bf, max_mean_bf):
        Y_to_use = np.exp(Y)
        Y_mean = np.mean(Y_to_use)
        if min_mean_bf is not None and Y_mean < min_mean_bf:
            scale_factor = min_mean_bf / Y_mean
            log("Scaling up BFs by %.4g" % scale_factor)
            Y_to_use = Y_to_use * scale_factor
        elif max_mean_bf is not None and Y_mean > max_mean_bf:
            scale_factor = max_mean_bf / Y_mean
            log("Scaling down BFs by %.4g" % scale_factor)
            Y_to_use = Y_to_use * max_mean_bf / Y_mean
        return np.log(Y_to_use)

    def _complete_p_beta_se(self, p, beta, se):
        p_none_mask = np.logical_or(p == None, np.isnan(p))
        beta_none_mask = np.logical_or(beta == None, np.isnan(beta))
        se_none_mask = np.logical_or(se == None, np.isnan(se))

        se_zero_mask = np.logical_and(~se_none_mask, se == 0)
        se_zero_beta_non_zero_mask = np.logical_and(se_zero_mask, np.logical_and(~beta_none_mask, beta != 0))

        if np.sum(se_zero_beta_non_zero_mask) != 0:
            warn("%d variants had zero SEs; setting these to beta zero and se 1" % (np.sum(se_zero_beta_non_zero_mask)))
            beta[se_zero_beta_non_zero_mask] = 0
        se[se_zero_mask] = 1

        bad_mask = np.logical_and(np.logical_and(p_none_mask, beta_none_mask), se_none_mask)
        if np.sum(bad_mask) > 0:
            warn("Couldn't infer p/beta/se at %d positions; setting these to beta zero and se 1" % (np.sum(bad_mask)))
            p[bad_mask] = 1
            beta[bad_mask] = 0
            se[bad_mask] = 1
            p_none_mask[bad_mask] = False
            beta_none_mask[bad_mask] = False
            se_none_mask[bad_mask] = False

        if np.sum(p_none_mask) > 0:
            p[p_none_mask] = 2 * scipy.stats.norm.pdf(-np.abs(beta[p_none_mask] / se[p_none_mask]))
        if np.sum(beta_none_mask) > 0:
            z = np.abs(scipy.stats.norm.ppf(np.array(p[beta_none_mask]/2)))
            beta[beta_none_mask] = z * se[beta_none_mask]
        if np.sum(se_none_mask) > 0:
            z = np.abs(scipy.stats.norm.ppf(np.array(p[se_none_mask]/2)))
            z[z == 0] = 1
            se[se_none_mask] = np.abs(beta[se_none_mask] / z)
        return (p, beta, se)
        
    def _distill_huge_signal_bfs(self, huge_signal_bfs, huge_signal_posteriors, huge_signal_sum_gene_cond_probabilities, huge_signal_mean_gene_pos, huge_signal_max_closest_gene_prob, cap_region_posterior, scale_region_posterior, phantom_region_posterior, allow_evidence_of_absence, correct_huge, huge_gene_covariates, huge_gene_covariates_mask, huge_gene_covariates_mat_inv, gene_prob_genes, total_genes=None, rel_prior_log_bf=None):

        if huge_signal_bfs is None:
            return

        if total_genes is not None:
            total_genes = self.genes

        #print("DELETE THE IND MAP!!!")
        #gene_to_ind = self._construct_map_to_ind(gene_prob_genes)


        if rel_prior_log_bf is None:
            prior_log_bf = np.full((1,huge_signal_bfs.shape[0]), self.background_log_bf)
        else:
            prior_log_bf = rel_prior_log_bf + self.background_log_bf
            if len(prior_log_bf.shape) == 1:
                prior_log_bf = prior_log_bf[np.newaxis,:]

        if prior_log_bf.shape[1] != huge_signal_bfs.shape[0]:
            bail("Error: priors shape did not match huge results shape (%s vs. %s)" % (prior_log_bf.shape, huge_signal_bfs.T.shape))

        #print("ORIG BFs")
        #print(huge_signal_bfs)

        if phantom_region_posterior:
            #first add an entry at the end to prior that is background_prior
            prior_log_bf = np.hstack((prior_log_bf, np.full((prior_log_bf.shape[0], 1), self.background_log_bf)))

            #then add a row to the bottom of signal_bfs
            phantom_probs = np.zeros(huge_signal_sum_gene_cond_probabilities.shape)
            phantom_mask = np.logical_and(huge_signal_sum_gene_cond_probabilities > 0, huge_signal_sum_gene_cond_probabilities < 1)

            phantom_probs[phantom_mask] = 1.0 - huge_signal_sum_gene_cond_probabilities[phantom_mask]

            #we need to set the BFs such that, when we add BFs below (with uniform prior) and then divide by total, we will get huge_signal_sum_gene_cond_probabilities for the non phantom
            #we *cannot* just convert phantom prob to phantom bf (like we do for signals; e.g. phantom_bfs = (phantom_probs / (1 - phantom_probs)) / self.background_bf) because the signals are defined as marginal probabilities
            #the BF needed to take gene in isolation from 0.05 to posterior
            #for phantom, we don't know the marginal -- it is inherently a joint estimate
            phantom_bfs = np.zeros(phantom_probs.shape)
            phantom_bfs[phantom_mask] = huge_signal_bfs.sum(axis=0).A1[phantom_mask] * (1.0 / huge_signal_sum_gene_cond_probabilities[phantom_mask] - 1.0)

            huge_signal_bfs = sparse.csc_matrix(sparse.vstack((huge_signal_bfs, phantom_bfs)))

            #print("NOW BFs")
            #print(huge_signal_bfs)

            #print(np.where(huge_signal_bfs[gene_to_ind["DOCK4"],:].todense() != 0))

            #print(huge_signal_bfs[gene_to_ind["UBE2E2"],:])
            #z_ind = 0
            #print(huge_signal_bfs[:,z_ind])

            huge_signal_sum_gene_cond_probabilities = huge_signal_sum_gene_cond_probabilities + phantom_probs

        prior_bf = np.exp(prior_log_bf)

        prior = prior_bf / (1 + prior_bf)

        if np.sum(prior == 0) > 0:
            print("ZERO:",prior_bf[prior == 0])

        prior[prior == 1] = 1 - 1e-4
        prior[prior == 0] = 1e-4


        #utility sparse matrices to use within loop
        #huge results matrix has posteriors for the region
        signal_log_priors = sparse.csr_matrix(copy.copy(huge_signal_bfs).T)
        sparse_aux = copy.copy(signal_log_priors)

        huge_results = np.zeros(prior_log_bf.shape)
        for i in range(prior_log_bf.shape[0]):

            #need prior * (1 - other_prior)^N in each entry
            #due to sparse matrices, and limiting memory usage, have to overwrite 
            #also, the reason for the complication below is that we have to work in log space, which
            #requires addition rather than subtraction, which we can't do directly on sparse matrices
            #we also need to switch between operating on data (when we do pointwise operations)
            #and operating on matrices (when we sum across axes)

            #priors specific to the signal
            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.multiply(np.log(1 - prior[i,:])))
            other_log_priors = sparse_aux.sum(axis=1).A1

            signal_log_priors.data = np.ones(len(signal_log_priors.data))
            signal_log_priors = sparse.csr_matrix(signal_log_priors.multiply(np.log(prior[i,:])))

            #now this has log(prior/(1-prior))
            signal_log_priors.data = signal_log_priors.data - sparse_aux.data

            #now need to add in (1-prior)^N
            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.T.multiply(other_log_priors).T)

            #now this has log(prior * (1-other_prior)^N)
            signal_log_priors.data = signal_log_priors.data + sparse_aux.data

            #now normalize
            #log_sum_bf = c + np.log(np.sum(np.exp(var_log_bf[region_vars] - c)))
            #where c is max value

            #to get max, have to add minimum (to get it over zero) and then subtract it
            c = signal_log_priors.min(axis=1)

            #ensure all c are positive (otherwise this will be removed from the sparse matrix and break the subsequent operations on data)
            c = c.toarray()
            c[c == 0] = np.min(c) * 1e-4

            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.multiply(c))

            signal_log_priors.data = signal_log_priors.data - sparse_aux.data

            c = signal_log_priors.max(axis=1) + c

            signal_log_priors.data = signal_log_priors.data + sparse_aux.data

            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.multiply(c))
            #store the max
            c_data = copy.copy(sparse_aux.data)

            #subtract c
            sparse_aux.data = np.exp(signal_log_priors.data - c_data)

            norms = sparse_aux.sum(axis=1).A1
            norms[norms != 0] = np.log(norms[norms != 0])

            sparse_aux.data = np.ones(len(sparse_aux.data))
            sparse_aux = sparse.csr_matrix(sparse_aux.T.multiply(norms).T)

            sparse_aux.data = c_data + sparse_aux.data

            signal_log_priors.data = signal_log_priors.data - sparse_aux.data

            #finally, we can obtain the priors matrix
            signal_log_priors.data = np.exp(signal_log_priors.data)            
            
            signal_priors = signal_log_priors.T

            #first have to adjust for priors
            #convert to BFs
            #we are overwriting data but drawing from the original (copied) huge_signal_bfs
            #multiply by priors. The final probabilities are proportional to the BFs * prior probabilities

            cur_huge_signal_bfs = huge_signal_bfs.multiply(signal_priors)

            #print("FINAL POSTS0")
            #print(cur_huge_signal_bfs[gene_to_ind["UBE2E2"],:])
            #z_ind = 0
            #print(signal_priors[:,z_ind])
            #print(cur_huge_signal_bfs[:,z_ind])


            #print("ORIG BFs")
            #print(huge_signal_bfs[gene_to_ind["UBE2E2"],:])
            #z_ind = 0
            #print(huge_signal_bfs[:,z_ind])

            #print("J3")
            #print(cur_huge_signal_bfs[gene_to_ind["ZAN"],:])
            #z_ind = 3260
            #print(huge_signal_bfs[:,z_ind])


            #print("PRIORS")
            #print(signal_priors)

            #rescale; these are now posteriors for the signal
            #either:
            #1. sum to 1 (scale_region_posterior)
            #2. reduce (but don't increase) to 1 (cap_region_posterior)
            #3. leave them as is (but scale to be bayes factors before normalizing)

            new_norms = cur_huge_signal_bfs.sum(axis=0).A1

            if not scale_region_posterior and not cap_region_posterior:
                #treat them as bayes factors
                new_norms /= (huge_signal_mean_gene_pos * (np.mean(prior_bf[i:])/self.background_bf))

            #print("NORMS")
            #print(new_norms)

            #this scales everything to sum to 1

            #in case any have zero
            new_norms[new_norms == 0] = 1

            cur_huge_signal_bfs = sparse.csr_matrix(cur_huge_signal_bfs.multiply(1.0 / new_norms))

            if not scale_region_posterior and not cap_region_posterior:
                #convert them back to probabilities
                cur_huge_signal_bfs.data = cur_huge_signal_bfs.data / (1 + cur_huge_signal_bfs.data)

            #cur_huge_signal_bfs are actually now probabilities that sum to 1 (incorporating priors)
            #signal_cap_norm_factor incorporates both scaling to the signal prob, as well as any capping to reduce the probabilities to their original sum
            if cap_region_posterior:
                signal_cap_norm_factor = huge_signal_posteriors * huge_signal_sum_gene_cond_probabilities
            else:
                signal_cap_norm_factor = copy.copy(huge_signal_posteriors)

            #this is the "fudge factor" that accounts for the fact that the causal gene could be outside of this window
            #we don't need to do it under the phantom gene model because we already added a phantom gene to absorb 1 - max_closest_gene_prob

            max_per_signal = cur_huge_signal_bfs.max(axis=0).todense().A1 * signal_cap_norm_factor
            overflow_mask = max_per_signal > huge_signal_max_closest_gene_prob
            signal_cap_norm_factor[overflow_mask] *= (huge_signal_max_closest_gene_prob / max_per_signal[overflow_mask])

            #rescale to the signal probability (cur_huge_signal_posteriors is the probability that the signal is true)
            cur_huge_signal_bfs = sparse.csr_matrix(cur_huge_signal_bfs.multiply(signal_cap_norm_factor))


            if not allow_evidence_of_absence:
                #this part now ensures that nothing with absence of evidence has evidence of absence
                #consider two coin flips: first, it is causal due to the GWAS signal here
                #second, it is causal for some other reason
                #to not be causal, both need to come up negatuve
                #first has probability equal to huge_signal_posteriors
                #second has probability equal to prior

                #cur_huge_signal_bfs.data = 1 - (1 - np.array(cur_huge_signal_bfs.data)) * (1 - prior[i,:])
                #but, we cannot subtract from sparse matrices so have to do it this way

                cur_huge_signal_bfs.data = 1 - cur_huge_signal_bfs.data
                
                #cur_huge_signal_bfs = sparse.csc_matrix(cur_huge_signal_bfs.T.multiply(1 - prior[i,:]).T)
                cur_huge_signal_bfs = sparse.csc_matrix(cur_huge_signal_bfs.T.multiply(1 - self.background_prior).T)

                cur_huge_signal_bfs.data = 1 - cur_huge_signal_bfs.data            

            #print("FINAL POSTS")
            #print(cur_huge_signal_bfs)
            #print(np.sum(cur_huge_signal_bfs))

            #take care of case where they sum to above 1 (and in which case we have to treat them as BFs)
            #bf_mask = cur_huge_signal_posteriors > 1
            #if np.sum(bf_mask) > 0:
            #    cur_huge_signal_bfs = sparse.csr_matrix(cur_huge_signal_bfs)
            #    cur_huge_signal_bfs_to_change = cur_huge_signal_bfs[:,bf_mask]
            #    cur_huge_signal_bfs_to_change.data = cur_huge_signal_bfs_to_change.data / (1 + cur_huge_signal_bfs_to_change.data)
            #    cur_huge_signal_bfs[:,bf_mask] = cur_huge_signal_bfs_to_change

            #print("FINAL POSTS1")
            #print(cur_huge_signal_bfs)
            #print(np.sum(cur_huge_signal_bfs))

            #print("FINAL POSTS0")
            #print(cur_huge_signal_bfs[gene_to_ind["UBE2E2"],:])
            #print(np.sum(cur_huge_signal_bfs))

            #now, add in extra probability for cur_huge_signal_bfs not equal to 1
            #sparse_aux.data = np.ones(len(sparse_aux.data))
            #sparse_aux = sparse.csr_matrix(sparse_aux.T.multiply(1 - cur_huge_signal_posteriors).T)
            #sparse_aux = sparse.csr_matrix(sparse_aux.multiply(prior[i,:]))
            #cur_huge_signal_bfs = cur_huge_signal_bfs + sparse_aux.T

            #print("FINAL POSTS2")
            #print(cur_huge_signal_bfs)
            #print(np.sum(cur_huge_signal_bfs))

            if cur_huge_signal_bfs.shape[1] > 0:
                #disable option to sum huge
                #if sum_huge:
                #    cur_huge_signal_bfs.data = np.log(1 - cur_huge_signal_bfs.data)
                #    huge_results[i,:] = 1 - np.exp(cur_huge_signal_bfs.sum(axis=1).A1)

                #This now has strongest signal posterior across all signals
                huge_results[i,:] = cur_huge_signal_bfs.max(axis=1).todense().A1


        #anything that was zero tacitly has probability equal to prior

        huge_results[huge_results == 0] = self.background_prior

        absent_genes = set()
        if total_genes is not None:
            #have to account for these
            absent_genes = set(total_genes) - set(gene_prob_genes)

        total_prob_causal = np.sum(huge_results)
        mean_prob_causal = (total_prob_causal + self.background_prior * len(absent_genes)) / (len(gene_prob_genes) + len(absent_genes)) 
        norm_constant = self.background_prior / mean_prob_causal

        #only normalize if enough genes
        max_prob = 1
        if len(gene_prob_genes) < 1000:
            norm_constant = max_prob
        elif norm_constant >= 1:
            norm_constant = max_prob

        #fix the maximum background prior across all genes
        #max_background_prior = None
        i#f max_background_prior is not None and mean_prob_causal > max_background_prior:
        #    norm_constant = max_background_prior / mean_prob_causal
        #else:

        norm_constant = max_prob

        if norm_constant != 1:
            log("Scaling output probabilities by %.4g" % norm_constant)

        huge_results *= norm_constant
        #now have to subtract out the prior

        okay_mask = huge_results < 1

        #print("Priors",prior[:,self.gene_to_ind['MIGA2']])
        #print("Initially (total prob)",huge_results[:,self.gene_to_ind['MIGA2']])

        #we will add this to the prior to get the final posterior, so just subtract it
        huge_results[okay_mask] = np.log(huge_results[okay_mask] / (1 - huge_results[okay_mask])) - self.background_log_bf

        #print("Now ",huge_results[:,gene_to_ind['ZAN']])

        huge_results[~okay_mask] = np.max(huge_results[okay_mask])

        #print("Now2 ",huge_results[:,self.gene_to_ind['MIGA2']])
        absent_prob = self.background_prior * norm_constant
        absent_log_bf = np.log(absent_prob / (1 - absent_prob)) - self.background_log_bf

        if phantom_region_posterior:
            huge_results = huge_results[:,:-1]

        huge_results_uncorrected = copy.copy(huge_results)
        if correct_huge and huge_gene_covariates is not None:
            assert(huge_gene_covariates_mat_inv is not None)
            assert(huge_gene_covariates_mask is not None)
            huge_results_mask = np.all(huge_results < np.mean(huge_results) + 5 * np.std(huge_results), axis=0)
            cur_huge_gene_covariates_mask = np.logical_and(huge_gene_covariates_mask, huge_results_mask)
            #dimensions are num_covariates x chains
            
            pred_slopes = huge_gene_covariates_mat_inv.dot(huge_gene_covariates[cur_huge_gene_covariates_mask,:].T).dot(huge_results[:,cur_huge_gene_covariates_mask].T)
            self.huge_gene_covariate_betas = np.mean(pred_slopes, axis=1)
            log("Mean slopes are %s" % self.huge_gene_covariate_betas, TRACE)

            param_names = ["%s_beta" % self.huge_gene_covariate_names[i] for i in range(len(self.huge_gene_covariate_names)) if i != self.huge_gene_covariate_intercept_index]
            param_values = self.huge_gene_covariate_betas
            self._record_params(dict(zip(param_names, param_values)), record_only_first_time=True)

            pred_huge_values = huge_gene_covariates.dot(pred_slopes).T
            pred_huge_residuals = huge_results - pred_huge_values

            #flag those that are very high
            max_huge_change = 1.0
            bad_mask = pred_huge_residuals - huge_results > max_huge_change
            if np.sum(bad_mask) > 0:
                warn("Not correcting %d genes for covariates due to large swings; there may be a problem with the covariates or input" % np.sum(bad_mask))
            huge_results[~bad_mask] = pred_huge_residuals[~bad_mask]

        #print("Now ",huge_results[:,gene_to_ind['ZAN']])

        huge_results = np.squeeze(huge_results)
        if huge_results_uncorrected is not huge_results:
            huge_results_uncorrected = np.squeeze(huge_results_uncorrected)

        #if self.gene_to_ind is not None:
        #    print(huge_results[:,self.gene_to_ind["KLF14"]])

        return (huge_results, huge_results_uncorrected, absent_genes, absent_log_bf)


    def _read_loc_file(self, loc_file, return_intervals=False, hold_out_chrom=None):

        gene_to_chrom = {}
        gene_to_pos = {}
        gene_chrom_name_pos = {}

        chrom_interval_to_gene = {}

        with open(loc_file) as loc_fh:
            for line in loc_fh:
                cols = line.strip().split()
                if len(cols) != 6:
                    bail("Format for --gene-loc-file is:\n\tgene_id\tchrom\tstart\tstop\tstrand\tgene_name\nOffending line:\n\t%s" % line)
                gene = cols[5]
                if self.gene_label_map is not None and gene in self.gene_label_map:
                    gene = self.gene_label_map[gene]
                chrom = cols[1]
                if hold_out_chrom is not None and chrom == hold_out_chrom:
                    continue
                pos1 = int(cols[2])
                pos2 = int(cols[3])

                if gene in gene_to_chrom and gene_to_chrom[gene] != chrom:
                    warn("Gene %s appears multiple times with different chromosomes; keeping only first" % gene)
                    continue

                if gene in gene_to_pos and np.abs(np.mean(gene_to_pos[gene]) - np.mean((pos1,pos2))) > 1e7:
                    warn("Gene %s appears multiple times with far away positions; keeping only first" % gene)
                    continue

                gene_to_chrom[gene] = chrom
                gene_to_pos[gene] = (pos1, pos2)

                if chrom not in gene_chrom_name_pos:
                    gene_chrom_name_pos[chrom] = {}
                if gene not in gene_chrom_name_pos[chrom]:
                    gene_chrom_name_pos[chrom][gene] = set()
                if pos1 not in gene_chrom_name_pos[chrom][gene]:
                    gene_chrom_name_pos[chrom][gene].add(pos1)
                if pos2 not in gene_chrom_name_pos[chrom][gene]:
                    gene_chrom_name_pos[chrom][gene].add(pos2)

                if pos2 < pos1:
                    t = pos1
                    pos1 = pos2
                    pos2 = t

                if return_intervals:
                    if chrom not in chrom_interval_to_gene:
                        chrom_interval_to_gene[chrom] = {}

                    if (pos1, pos2) not in chrom_interval_to_gene[chrom]:
                        chrom_interval_to_gene[chrom][(pos1, pos2)] = []

                    chrom_interval_to_gene[chrom][(pos1, pos2)].append(gene) 

                #we consider distance to a gene to be both its start, its end, and also intermediate points within it
                #split_gene_length determines how many intermediate points there are
                split_gene_length = 1000000
                if pos2 > pos1:
                    for posm in range(pos1, pos2, split_gene_length)[1:]:
                        gene_chrom_name_pos[chrom][gene].add(posm)

        if return_intervals:
            return chrom_interval_to_gene
        else:
            return (gene_chrom_name_pos, gene_to_chrom, gene_to_pos)



    def _read_correlations(self, gene_cor_file=None, gene_loc_file=None, gene_cor_file_gene_col=1, gene_cor_file_cor_start_col=10, compute_correlation_distance_function=True):
        if gene_cor_file is not None:
            log("Reading in correlations from %s" % gene_cor_file)
            unique_genes = np.array([True] * len(self.genes))
            correlation_m = [np.ones(len(self.genes))]
            with open(gene_cor_file) as gene_cor_fh:
                gene_cor_file_gene_col = gene_cor_file_gene_col - 1
                gene_cor_file_cor_start_col = gene_cor_file_cor_start_col - 1
                #store the genes in order, which we will need in order to map from each line in the file to correlation
                gene_cor_file_gene_names = []
                new_gene_index = {}
                cor_file_index = 0
                j = 0
                for line in gene_cor_fh:
                    if line[0] == "#":
                        continue
                    cols = line.strip().split()
                    if len(cols) < gene_cor_file_cor_start_col:
                        bail("Not enough columns in --gene-cor-file. Offending line:\n\t%s" % line)
                    gene_name = cols[gene_cor_file_gene_col]
                    if self.gene_label_map is not None and gene_name in self.gene_label_map:
                        gene_name = self.gene_label_map[gene_name]

                    gene_cor_file_gene_names.append(gene_name)
                    i = j - 1
                    if gene_name in self.gene_to_ind:
                        new_gene_index[gene_name] = cor_file_index
                        gene_correlations = [float(x) for x in cols[gene_cor_file_cor_start_col:]]
                        for gc_i in range(1,len(gene_correlations)+1):
                            cur_cor = gene_correlations[-gc_i]
                            if gc_i > cor_file_index:
                                bail("Error in --gene-cor-file: number of correlations is more than the number of genes seen to this point")
                            gene_i = gene_cor_file_gene_names[cor_file_index - gc_i]
                            if gene_i not in self.gene_to_ind:
                                continue
                            if cur_cor >= 1:
                                unique_genes[self.gene_to_ind[gene_i]] = False
                                #log("Excluding %s (correlation=%.4g with %s)" % (gene_i, cur_cor, gene_name), TRACE)

                            #store the values for the regression(s)
                            correlation_m_ind = j - i
                            while correlation_m_ind >= len(correlation_m):
                                correlation_m.append(np.zeros(len(self.genes)))
                            correlation_m[correlation_m_ind][i] = cur_cor
                            i -= 1
                        j += 1
                    cor_file_index += 1
            correlation_m = np.array(correlation_m)

            #now subset down the duplicate locations
            #self._subset_genes(unique_genes, skip_V=True, skip_scale_factors=True)
            #correlation_m = correlation_m[:,unique_genes]
            #log("Excluded %d duplicate genes" % sum(~unique_genes))

            sorted_gene_indices = sorted(range(len(self.genes)), key=lambda k: new_gene_index[self.genes[k]] if self.genes[k] in new_gene_index else 0)
            #sort the X and y values
            self._sort_genes(sorted_gene_indices, skip_V=True, skip_scale_factors=True)

        else:
            if gene_loc_file is None:
                bail("Need --gene-loc-file if don't specify --gene-cor-file")

            self.gene_locations = {}
            log("Reading gene locations")

            if self.gene_to_chrom is None:
                self.gene_to_chrom = {}
            if self.gene_to_pos is None:
                self.gene_to_pos = {}

            unique_genes = np.array([True] * len(self.genes))
            location_genes = {}
            with open(gene_loc_file) as gene_loc_fh:
                for line in gene_loc_fh:
                    cols = line.strip().split()
                    if len(cols) != 6:
                        bail("Format for --gene-loc-file is:\n\tgene_id\tchrom\tstart\tstop\tstrand\tgene_name\nOffending line:\n\t%s" % line)
                    gene_name = cols[5]
                    if gene_name not in self.gene_to_ind:
                        continue

                    chrom = cols[1]
                    start = int(cols[2])
                    end = int(cols[3])

                    self.gene_to_chrom[gene_name] = chrom
                    self.gene_to_pos[gene_name] = (start, end)

                    location = (chrom, start, end)
                    self.gene_locations[gene_name] = location
                    if location in location_genes:
                        #keep the one with highest Y
                        old_ind = self.gene_to_ind[location_genes[location]]
                        new_ind = self.gene_to_ind[gene_name]
                        if self.Y[old_ind] >= self.Y[new_ind]:
                            unique_genes[new_ind] = False
                            log("Excluding %s (duplicate of %s)" % (self.genes[new_ind], self.genes[old_ind]), TRACE)
                        else:
                            unique_genes[new_ind] = True
                            unique_genes[old_ind] = False
                            log("Excluding %s (duplicate of %s)" % (self.genes[old_ind], self.genes[new_ind]), TRACE)
                            location_genes[location] = gene_name
                    else:
                        location_genes[location] = gene_name

            #now subset down the duplicate locations
            self._subset_genes(unique_genes, skip_V=True, skip_scale_factors=True)

            sorted_gene_indices = sorted(range(len(self.genes)), key=lambda k: self.gene_locations[self.genes[k]] if self.genes[k] in self.gene_locations else ("NA", 0))

            #sort the X and y values
            self._sort_genes(sorted_gene_indices, skip_V=True, skip_scale_factors=True)

            #now we have to determine the relationship between distance and correlation
            correlation_m = self._compute_correlations_from_distance(compute_correlation_distance_function=compute_correlation_distance_function)

        #set the diagonal to 1
        correlation_m[0,:] = 1.0

        log("Banded correlation matrix: shape %s, %s" % (correlation_m.shape[0], correlation_m.shape[1]), DEBUG)
        log("Non-zero entries: %s" % sum(sum(correlation_m > 0)), DEBUG)

        return correlation_m

    def _compute_correlations_from_distance(self, Y=None, compute_correlation_distance_function=True):
        if self.genes is None:
            return None

        if Y is None:
            Y = self.Y

        if Y is None or self.gene_locations is None:
            return None

        correlation_m = [np.zeros(len(self.genes))]

        max_distance_to_model = 1000000.0
        num_bins = 1000
        distance_num = np.zeros(num_bins)
        distance_denom = np.zeros(num_bins)
        log("Calculating distance/correlation function")
        #this loop does two things
        #first, it stores the distances in a banded matrix -- this will be used later to compute the correlations
        #second, it stores the various distances / empirical covariances in two arrays for doing the regression
        for i in range(len(self.genes)):
            if self.genes[i] in self.gene_locations:
                loc = self.gene_locations[self.genes[i]]
                #traverse in each direction to find pairs within a range
                for j in range(i+1, len(self.genes)):
                    if self.genes[j] in self.gene_locations:
                        loc2 = self.gene_locations[self.genes[j]]
                        if not loc[0] == loc2[0]:
                            continue
                        distance = np.abs(loc2[1] - loc[1])
                        if distance > max_distance_to_model:
                            break
                        #store the values for the regression(s)
                        bin_number = int((distance / max_distance_to_model) * (num_bins - 1))
                        if Y[i] != 0:
                            distance_num[bin_number] += Y[i] * Y[j]
                            distance_denom[bin_number] += Y[i]**2
                        #store the distances for later
                        correlation_m_ind = j - i
                        while correlation_m_ind >= len(correlation_m):
                            correlation_m.append(np.array([np.inf] * len(self.genes)))
                        #print("Adding %s, %s, %s, %s" % (i, correlation_m_ind, j, correlation))
                        correlation_m[correlation_m_ind][i] = distance

        correlation_m = np.array(correlation_m)

        # fit function
        slope = -5.229e-07
        intercept = 0.54

        if compute_correlation_distance_function:
            bin_Y = distance_num[distance_denom != 0] / distance_denom[distance_denom != 0]
            bin_X = (np.array(range(len(distance_num))) * (max_distance_to_model / num_bins))[distance_denom != 0]
            sd_outlier_threshold = 3
            bin_outlier_max = np.mean(bin_Y) + sd_outlier_threshold * np.std(bin_Y)
            bin_mask = np.logical_and(bin_Y > -bin_outlier_max, bin_Y < bin_outlier_max)
            bin_Y = bin_Y[bin_mask]
            bin_X = bin_X[bin_mask]

            slope = np.cov(bin_X, bin_Y)[0,1] / np.var(bin_X)
            intercept = np.mean(bin_Y - bin_X * slope)
            max_distance = -intercept / slope
            if slope > 0:
                log("Slope was positive; setting all correlations to zero")
                intercept = 0
                slope = 0
            elif intercept < 0:
                log("Incercept was negative; setting all correlations to zero")                
                intercept = 0
                slope = 0
            else:
                log("Fit function from bins: r^2 = %.2g%.4gx; max distance=%d" % (intercept, slope, max_distance))
        else:
            log("Using precomputed function: r^2 = %.2g%.4gx; max distance=%d" % (intercept, slope, max_distance))

        if slope < 0:
            max_distance = -intercept / slope
            log("Using function: r^2 = %.2g + %.4g * x; max distance=%d" % (intercept, slope, max_distance))                
            self._record_params({"correlation_slope": slope, "correlation_intercept": intercept, "correlation_max_dist": max_distance})

            #map the values over from raw values to correlations/covariances
            correlation_m = intercept + slope * correlation_m
            correlation_m[correlation_m <= 0] = 0
            correlation_m[0,:] = 1.0
        else:
            correlation_m[0,:] = 1.0
            correlation_m[1:,:] = 0.0

        return correlation_m
            

    def _compute_beta_tildes(self, X, Y, y_var, scale_factors, mean_shifts, resid_correlation_matrix=None, log_fun=log):

        log_fun("Calculating beta tildes")

        #Y can be a matrix with dimensions:
        #number of parallel runs x number of gene sets
        if len(Y.shape) == 2:
            len_Y = Y.shape[1]
            Y = (Y.T - np.mean(Y, axis=1)).T
        else:
            len_Y = Y.shape[0]
            Y = Y - np.mean(Y)

        dot_product = np.array(X.T.dot(Y.T) / len_Y).T

        variances = np.power(scale_factors, 2)

        #avoid divide by 0 only
        variances[variances == 0] = 1

        #multiply by scale factors because we store beta_tilde in units of scaled X
        beta_tildes = scale_factors * dot_product / variances

        if len(Y.shape) == 2:
            ses = np.outer(np.sqrt(y_var), scale_factors)
        else:
            ses = np.sqrt(y_var) * scale_factors

        ses /= (np.sqrt(variances * (len_Y - 1)))

        #FIXME: implement exact SEs
        #rather than just using y_var as a constant, calculate X.multiply(beta_tildes)
        #then, subtract out Y for non-zero entries, sum square, sum total
        #then, add in square of Y for zero entries, add in total
        #use these to calculate the variance term

        se_inflation_factors = None
        if resid_correlation_matrix is not None:
            log_fun("Adjusting standard errors for correlations", DEBUG)
            #need to multiply by inflation factors: (X * sigma * X) / variances

            #SEs and betas are stored in units of centered and scaled X
            #we do not need to scale X here, however, because cor_variances will then be in units of unscaled X
            #since variances are also in units of unscaled X, these will cancel out

            r_X = resid_correlation_matrix.dot(X)
            r_X_col_means = r_X.multiply(X).sum(axis=0).A1 / X.shape[0]
            cor_variances = r_X_col_means - np.square(r_X_col_means)
            
            #never increase significance
            cor_variances[cor_variances < variances] = variances[cor_variances < variances]

            #both cor_variances and variances are in units of unscaled X
            se_inflation_factors = np.sqrt(cor_variances / variances)

        return self._finalize_regression(beta_tildes, ses, se_inflation_factors)

    def _compute_logistic_beta_tildes(self, X, Y, scale_factors, mean_shifts, resid_correlation_matrix=None, convert_to_dichotomous=True, rel_tol=0.01, X_stacked=None, append_pseudo=True, log_fun=log):

        log_fun("Calculating beta tildes")

        if Y is self.Y or Y is self.Y_for_regression:
            Y = copy.copy(Y)

        #Y can be a matrix with dimensions:
        #number of parallel runs x number of gene sets
        if len(Y.shape) == 1:
            orig_vector = True
            Y = Y[np.newaxis,:]
        else:
            orig_vector = False
        
        if convert_to_dichotomous:
            if np.sum(np.logical_and(Y != 0, Y != 1)) > 0:
                Y[np.isnan(Y)] = 0
                mult_sum = 1
                #log_fun("Multiplying Y sums by %.3g" % mult_sum)
                Y_sums = np.sum(Y, axis=1).astype(int) * mult_sum
                Y_sorted = np.sort(Y, axis=1)[:,::-1]
                Y_cumsum = np.cumsum(Y_sorted, axis=1)
                threshold_val = np.diag(Y_sorted[:,Y_sums])

                true_mask = (Y.T > threshold_val).T
                Y[true_mask] = 1
                Y[~true_mask] = 0
                log_fun("Converting values to dichotomous outcomes; y=1 for input y > %s" % threshold_val, DEBUG)

        #if len(self.genes) == Y.shape[1]:
        #    for i in range(len(self.genes)):
        #        print("%s\t%.3g" % (self.genes[i], Y[0,i]))

        log_fun("Outcomes: %d=1, %d=0; mean=%.3g" % (np.sum(Y==1), np.sum(Y==0), np.mean(Y)), TRACE)

        len_Y = Y.shape[1]
        num_chains = Y.shape[0]

        if append_pseudo:
            log_fun("Appending pseudo counts", TRACE)
            Y_means = np.mean(Y, axis=1)[:,np.newaxis]

            Y = np.hstack((Y, Y_means))

            X = sparse.csc_matrix(sparse.vstack((X, np.ones(X.shape[1]))))
            if X_stacked is not None:
                X_stacked = sparse.csc_matrix(sparse.vstack((X_stacked, np.ones(X_stacked.shape[1]))))

        #treat multiple chains as just additional gene set coefficients
        if X_stacked is None:
            if num_chains > 1:
                X_stacked = sparse.hstack([X] * num_chains)
            else:
                X_stacked = X

        num_non_zero = np.tile((X != 0).sum(axis=0).A1, num_chains)

        #old, memory more intensive
        #num_non_zero = (X_stacked != 0).sum(axis=0).A1

        num_zero = X_stacked.shape[0] - num_non_zero


        #WHEN YOU LOAD A NON-0/1 GENE SET, USE THIS CODE TO TEST
        #if X.shape[1] > 1:
        #    import statsmodels.api as sm
        #    for i in range(X.shape[1]):
        #        logit_mod = sm.Logit(Y.ravel(), sm.add_constant(X[:,i].todense().A1))
        #        logit_res = logit_mod.fit()
        #        print(logit_res.summary())

        #initialize
        #one per gene set
        beta_tildes = np.zeros(X.shape[1] * num_chains)
        #one per gene set
        alpha_tildes = np.zeros(X.shape[1] * num_chains)
        it = 0

        compute_mask = np.full(len(beta_tildes), True)
        diverged_mask = np.full(len(beta_tildes), False)

        def __compute_Y_R(X, beta_tildes, alpha_tildes, max_cap=0.999):
            exp_X_stacked_beta_alpha = X.multiply(beta_tildes)
            exp_X_stacked_beta_alpha.data += (X != 0).multiply(alpha_tildes).data
            max_val = 100
            overflow_mask = exp_X_stacked_beta_alpha.data > max_val
            exp_X_stacked_beta_alpha.data[overflow_mask] = max_val
            np.exp(exp_X_stacked_beta_alpha.data, out=exp_X_stacked_beta_alpha.data)
            
            #each gene set corresponds to a 2 feature regression
            #Y/R_pred have dim (num_genes, num_chains * num_gene_sets)
            Y_pred = copy.copy(exp_X_stacked_beta_alpha)
            #add in intercepts
            Y_pred.data = Y_pred.data / (1 + Y_pred.data)
            Y_pred.data[Y_pred.data > max_cap] = max_cap
            R = copy.copy(Y_pred)
            R.data = Y_pred.data * (1 - Y_pred.data)
            return (Y_pred, R)

        max_it = 100

        log_fun("Performing IRLS...")
        while True:
            it += 1
            prev_beta_tildes = copy.copy(beta_tildes)
            prev_alpha_tildes = copy.copy(alpha_tildes)

            #we are doing num_chains x X.shape[1] IRLS iterations in parallel.
            #Each parallel is a univariate regression of one gene set + intercept
            #first dimension is parallel chains
            #second dimension is each gene set as a univariate regression
            #calculate R
            #X is genesets*chains x genes

            #we are going to do this only for non-zero entries
            #the other entries are technically incorrect, but okay since we are only ever multiplying these by X (which have 0 at these entries)
            (Y_pred, R) = __compute_Y_R(X_stacked[:,compute_mask], beta_tildes[compute_mask], alpha_tildes[compute_mask])

            #values for the genes with zero for the gene set
            #these are constant across all genes (within a gene set/chain)
            max_val = 100
            overflow_mask = alpha_tildes > max_val
            alpha_tildes[overflow_mask] = max_val

            Y_pred_zero = np.exp(alpha_tildes[compute_mask])
            Y_pred_zero = Y_pred_zero / (1 + Y_pred_zero)
            R_zero = Y_pred_zero * (1 - Y_pred_zero)
 
            Y_sum_per_chain = np.sum(Y, axis=1)
            Y_sum = np.tile(Y_sum_per_chain, X.shape[1])

            #first term: phi*w in Bishop
            #This has length (num_chains * num_gene_sets)

            X_r_X_beta = X_stacked[:,compute_mask].power(2).multiply(R).sum(axis=0).A1.ravel()
            X_r_X_alpha = R.sum(axis=0).A1.ravel() + R_zero * num_zero[compute_mask]
            X_r_X_beta_alpha = X_stacked[:,compute_mask].multiply(R).sum(axis=0).A1.ravel()
            #inverse of [[a b] [c d]] is (1 / (ad - bc)) * [[d -b] [-c a]]
            #a = X_r_X_beta
            #b = c = X_r_X_beta_alpha
            #d = X_r_X_alpha
            denom = X_r_X_beta * X_r_X_alpha - np.square(X_r_X_beta_alpha)

            diverged = np.logical_or(np.logical_or(X_r_X_beta == 0, X_r_X_beta_alpha == 0), denom == 0)

            if np.sum(diverged) > 0:
                log_fun("%d beta_tildes diverged" % np.sum(diverged), TRACE)
                not_diverged = ~diverged

                cur_indices = np.where(compute_mask)[0]

                compute_mask[cur_indices[diverged]] = False
                diverged_mask[cur_indices[diverged]] = True

                #need to convert format in order to support indexing
                Y_pred = sparse.csc_matrix(Y_pred)
                R = sparse.csc_matrix(R)

                Y_pred = Y_pred[:,not_diverged]
                R = R[:,not_diverged]
                Y_pred_zero = Y_pred_zero[not_diverged]
                R_zero = R_zero[not_diverged]
                X_r_X_beta = X_r_X_beta[not_diverged]
                X_r_X_alpha = X_r_X_alpha[not_diverged]
                X_r_X_beta_alpha = X_r_X_beta_alpha[not_diverged]
                denom = denom[not_diverged]

            if np.sum(np.isnan(X_r_X_beta) | np.isnan(X_r_X_alpha) | np.isnan(X_r_X_beta_alpha)) > 0:
                bail("Error: something went wrong")

            #second term: r_inv * (y-t) in Bishop
            #for us, X.T.dot(Y_pred - Y)

            R_inv_Y_T_beta = X_stacked[:,compute_mask].multiply(Y_pred).sum(axis=0).A1.ravel() - X.T.dot(Y.T).T.ravel()[compute_mask]
            R_inv_Y_T_alpha = (Y_pred.sum(axis=0).A1.ravel() + Y_pred_zero * num_zero[compute_mask]) - Y_sum[compute_mask]

            beta_tilde_row = (X_r_X_beta * prev_beta_tildes[compute_mask] + X_r_X_beta_alpha * prev_alpha_tildes[compute_mask] - R_inv_Y_T_beta)
            alpha_tilde_row = (X_r_X_alpha * prev_alpha_tildes[compute_mask] + X_r_X_beta_alpha * prev_beta_tildes[compute_mask] - R_inv_Y_T_alpha)


            beta_tildes[compute_mask] = (X_r_X_alpha * beta_tilde_row - X_r_X_beta_alpha * alpha_tilde_row) / denom
            alpha_tildes[compute_mask] = (X_r_X_beta * alpha_tilde_row - X_r_X_beta_alpha * beta_tilde_row) / denom

            diff = np.abs(beta_tildes - prev_beta_tildes)
            diff_denom = np.abs(beta_tildes + prev_beta_tildes)
            diff_denom[diff_denom == 0] = 1
            rel_diff = diff / diff_denom

            #log_fun("%d left to compute; max diff=%.4g" % (np.sum(compute_mask), np.max(rel_diff)))
               
            compute_mask[np.logical_or(rel_diff < rel_tol, beta_tildes == 0)] = False
            if np.sum(compute_mask) == 0:
                log_fun("Converged after %d iterations" % it, TRACE)
                break
            if it == max_it:
                log_fun("Stopping with %d still not converged" % np.sum(compute_mask), TRACE)
                diverged_mask[compute_mask] = True
                break

        
        while True:
            #handle diverged
            if np.sum(diverged_mask) > 0:
                beta_tildes[diverged_mask] = 0
                alpha_tildes[diverged_mask] = Y_sum[diverged_mask] / len_Y

            max_coeff = 100            

            #genes x num_coeffs
            (Y_pred, V) = __compute_Y_R(X_stacked, beta_tildes, alpha_tildes)

            #this is supposed to calculate (X^t * V * X)-1
            #where X is the n x 2 matrix of genes x (1/0, 1)
            #d / (ad - bc) is inverse formula
            #a = X.multiply(V).multiply(X)
            #b = c = sum(X.multiply(V)
            #d = V.sum() + constant values for all zero X (since those aren't in V)
            #also need to add in enough p*(1-p) values for all of the X=0 entries; this is where the p_const * number of zero X comes in

            params_too_large_mask = np.logical_or(np.abs(alpha_tildes) > max_coeff, np.abs(beta_tildes) > max_coeff)
            #to prevent overflow
            alpha_tildes[np.abs(alpha_tildes) > max_coeff] = max_coeff

            p_const = np.exp(alpha_tildes) / (1 + np.exp(alpha_tildes))

            #jason_mask = V.sum(axis=0).A1 + p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1) == 0
            #jason_mask[0] = True
            #if np.sum(jason_mask) > 0:
            #    print("JASON ALSO HAVE",np.sum(jason_mask))
            #    print(V.sum(axis=0).A1[jason_mask])
            #    print((p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1))[jason_mask])
            #    print(Y_pred.sum(axis=0).A1[jason_mask])
            #    print(beta_tildes[jason_mask])
            #    print(alpha_tildes[jason_mask])
            #    print(p_const[jason_mask])
            #    print((X_stacked != 0).sum(axis=0).A1[jason_mask])
            #    self.write_X("x2.gz")
            #    for c in range(Y.shape[0]):
            #        y = ""
            #        for j in range(len(Y[c,:])):
            #            y = "%s\t%s" % (y, Y[c,j])
            #        print(y)


                #for i in np.where(jason_mask)[0]:
                #    x = X_stacked[:,i].todense().A1
                #    c = int(i / X.shape[1])
                #    print("INDEX",i,"CHAIN",c+1)
                #    for j in range(len(Y[c,:])):
                #        print(x[j], Y[c,j])

            variance_denom = (V.sum(axis=0).A1 + p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1))
            denom_zero = variance_denom == 0
            variance_denom[denom_zero] = 1

            variances = X_stacked.power(2).multiply(V).sum(axis=0).A1 - np.power(X_stacked.multiply(V).sum(axis=0).A1, 2) / variance_denom
            variances[denom_zero] = 100

            #set them to diverged also if variances are negative or variance denom is 0 or if params are too large
            additional_diverged_mask = np.logical_and(~diverged_mask, np.logical_or(np.logical_or(variances < 0, denom_zero), params_too_large_mask))

            if np.sum(additional_diverged_mask) > 0:
                #additional divergences
                diverged_mask = np.logical_or(diverged_mask, additional_diverged_mask)
            else:
                break

        se_inflation_factors = None
        if resid_correlation_matrix is not None:
            log("Adjusting standard errors for correlations", DEBUG)
            #need to multiply by inflation factors: (X * sigma * X) / variances

            if append_pseudo:
                resid_correlation_matrix = sparse.hstack((resid_correlation_matrix, np.zeros(resid_correlation_matrix.shape[0])[:,np.newaxis]))
                new_bottom_row = np.zeros(resid_correlation_matrix.shape[1])

                new_bottom_row[-1] = 1
                resid_correlation_matrix = sparse.vstack((resid_correlation_matrix, new_bottom_row)).tocsc()

            cor_variances = copy.copy(variances)

            #Old, memory intensive version
            #r_X = resid_correlation_matrix.dot(X_stacked)
            #cor_variances = r_X.multiply(X_stacked).multiply(V).sum(axis=0).A1 - r_X.multiply(V).sum(axis=0).A1 / len_Y

            r_X = resid_correlation_matrix.dot(X)
            #we will only be using this to multiply matrices that are non-zero only when X is
            r_X = (X != 0).multiply(r_X)

            #OLD ONE -- wrong denominator
            #cor_variances = sparse.hstack([r_X.multiply(X)] * num_chains).multiply(V).sum(axis=0).A1 - sparse.hstack([r_X] * num_chains).multiply(V).sum(axis=0).A1 / len_Y

            #NEW ONE

            cor_variances = sparse.hstack([r_X.multiply(X)] * num_chains).multiply(V).sum(axis=0).A1 - sparse.hstack([r_X] * num_chains).multiply(V).sum(axis=0).A1 / (V.sum(axis=0).A1 + p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1))

            #both cor_variances and variances are in units of unscaled X
            variances[variances == 0] = 1
            se_inflation_factors = np.sqrt(cor_variances / variances)

        
        #now unpack the chains

        if num_chains > 1:
            beta_tildes = beta_tildes.reshape(num_chains, X.shape[1])
            alpha_tildes = alpha_tildes.reshape(num_chains, X.shape[1])
            variances = variances.reshape(num_chains, X.shape[1])
            diverged_mask = diverged_mask.reshape(num_chains, X.shape[1])
            if se_inflation_factors is not None:
                se_inflation_factors = se_inflation_factors.reshape(num_chains, X.shape[1])
        else:
            beta_tildes = beta_tildes[np.newaxis,:]
            alpha_tildes = alpha_tildes[np.newaxis,:]
            variances = variances[np.newaxis,:]
            diverged_mask = diverged_mask[np.newaxis,:]
            if se_inflation_factors is not None:
                se_inflation_factors = se_inflation_factors[np.newaxis,:]

        variances[:,scale_factors == 0] = 1

        #not necessary
        #if inflate_se:
        #    inflate_mask = scale_factors > np.mean(scale_factors)
        #    variances[:,inflate_mask] *= np.mean(np.power(scale_factors, 2)) / np.power(scale_factors[inflate_mask], 2)  

        #multiply by scale factors because we store beta_tilde in units of scaled X
        beta_tildes = scale_factors * beta_tildes

        ses = scale_factors / np.sqrt(variances)

        if orig_vector:
            beta_tildes = np.squeeze(beta_tildes, axis=0)
            alpha_tildes = np.squeeze(alpha_tildes, axis=0)
            variances = np.squeeze(variances, axis=0)
            ses = np.squeeze(ses, axis=0)
            diverged_mask = np.squeeze(diverged_mask, axis=0)

            if se_inflation_factors is not None:
                se_inflation_factors = np.squeeze(se_inflation_factors, axis=0)

        return self._finalize_regression(beta_tildes, ses, se_inflation_factors) + (alpha_tildes, diverged_mask)


    def _finalize_regression(self, beta_tildes, ses, se_inflation_factors):

        if se_inflation_factors is not None:
            ses *= se_inflation_factors

        if np.prod(ses.shape) > 0:
            #empty mask
            empty_mask = np.logical_and(beta_tildes == 0, ses <= 0)
            max_se = np.max(ses)

            ses[empty_mask] = max_se * 100 if max_se > 0 else 100

            #if no y var, set beta tilde to 0

            beta_tildes[ses <= 0] = 0

        z_scores = np.zeros(beta_tildes.shape)
        ses_positive_mask = ses > 0
        z_scores[ses_positive_mask] = beta_tildes[ses_positive_mask] / ses[ses_positive_mask]
        if np.any(~ses_positive_mask):
            warn("There were %d gene sets with negative ses; setting z-scores to 0" % (np.sum(~ses_positive_mask)))
        p_values = 2*scipy.stats.norm.cdf(-np.abs(z_scores))
        return (beta_tildes, ses, z_scores, p_values, se_inflation_factors)

    def _correct_beta_tildes(self, beta_tildes, ses, se_inflation_factors, total_qc_metrics, mean_qc_metrics, add_missing=True, fit=True):

        orig_total_qc_metrics = total_qc_metrics

        #inputs have first axis equal to number of chains
        if len(beta_tildes.shape) == 1:
            beta_tildes = beta_tildes[np.newaxis,:]
        if len(ses.shape) == 1:
            ses = ses[np.newaxis,:]
        if se_inflation_factors is not None and len(se_inflation_factors.shape) == 1:
            se_inflation_factors = se_inflation_factors[np.newaxis,:]

        if total_qc_metrics is None or mean_qc_metrics is None:
            if not self.huge_correct_huge:
                warn("--correct-huge was not use, so skipping correction")
            else:
                warn("Huge scores were not used, so skipping correction")
        else:

            remove_mask = np.full(beta_tildes.shape[1], False)
            if add_missing:
                if self.beta_tildes_missing is not None:
                    beta_tildes = np.hstack((beta_tildes, np.tile(self.beta_tildes_missing, beta_tildes.shape[0]).reshape(beta_tildes.shape[0], len(self.beta_tildes_missing))))
                    ses = np.hstack((ses, np.tile(self.ses_missing, ses.shape[0]).reshape(ses.shape[0], len(self.ses_missing))))
                    if se_inflation_factors is not None:
                        se_inflation_factors = np.hstack((se_inflation_factors, np.tile(self.se_inflation_factors_missing, se_inflation_factors.shape[0]).reshape(se_inflation_factors.shape[0], len(self.se_inflation_factors_missing))))

                    total_qc_metrics = np.vstack((total_qc_metrics, self.total_qc_metrics_missing))
                    mean_qc_metrics = np.append(mean_qc_metrics, self.mean_qc_metrics_missing)
                    remove_mask = np.append(remove_mask, np.full(len(self.beta_tildes_missing), True))

                if self.beta_tildes_ignored is not None:
                    beta_tildes = np.hstack((beta_tildes, np.tile(self.beta_tildes_ignored, beta_tildes.shape[0]).reshape(beta_tildes.shape[0], len(self.beta_tildes_ignored))))
                    ses = np.hstack((ses, np.tile(self.ses_ignored, ses.shape[0]).reshape(ses.shape[0], len(self.ses_ignored))))
                    if se_inflation_factors is not None:
                        se_inflation_factors = np.hstack((se_inflation_factors, np.tile(self.se_inflation_factors_ignored, se_inflation_factors.shape[0]).reshape(se_inflation_factors.shape[0], len(self.se_inflation_factors_ignored))))

                    total_qc_metrics = np.vstack((total_qc_metrics, self.total_qc_metrics_ignored))
                    mean_qc_metrics = np.append(mean_qc_metrics, self.mean_qc_metrics_ignored)
                    remove_mask = np.append(remove_mask, np.full(len(self.beta_tildes_ignored), True))

            #this causes matrix to become very unstable (since it is linear combination of the others)
            #total_qc_metrics = np.hstack((total_qc_metrics, mean_qc_metrics[:,np.newaxis]))

            #this uses only avg metrics; don't do it
            #total_qc_metrics = mean_qc_metrics[:,np.newaxis]

            z_scores = np.zeros(beta_tildes.shape)
            z_scores[ses != 0] = beta_tildes[ses != 0] / ses[ses != 0]
            z_scores_mask = np.all(np.logical_and(z_scores <= np.mean(z_scores) + 5 * np.std(z_scores), ses != 0), axis=0)
            metrics_mask = np.all(total_qc_metrics <= np.mean(total_qc_metrics, axis=0) + 5 * np.std(total_qc_metrics, axis=0), axis=1)
            pred_mask = np.logical_and(z_scores_mask, metrics_mask)


            #get univariate regression coefficients
            (metric_beta_tildes_m, metric_ses_m, metric_z_scores_m, metric_p_values_m, metric_se_inflation_factors_m) = self._compute_beta_tildes(total_qc_metrics[pred_mask,:], z_scores[:,pred_mask], np.var(z_scores[:,pred_mask], axis=1), np.std(total_qc_metrics[pred_mask,:], axis=0), np.mean(total_qc_metrics[pred_mask,:], axis=0), resid_correlation_matrix=None, log_fun=lambda x, y=0: 1)
                        
            #filter out metrics as needed
            keep_metrics = np.where(np.any(metric_p_values_m < 0.05, axis=0))[0]
            if len(keep_metrics) < total_qc_metrics.shape[1]:
                log("Not using %d non-significant metrics" % (total_qc_metrics.shape[1] - len(keep_metrics)))
                total_qc_metrics = total_qc_metrics[:,keep_metrics]

            #find the intercept index
            intercept_mask = (np.std(total_qc_metrics, axis=0) == 0)
            if np.sum(intercept_mask) == 0:
                total_qc_metrics = np.hstack((total_qc_metrics, np.ones((total_qc_metrics.shape[0],1))))
                intercept_mask = np.append(intercept_mask, True)

            #make this smaller for numerical stability
            assert(total_qc_metrics is not orig_total_qc_metrics)
            total_qc_metrics /= np.sqrt(total_qc_metrics.shape[0])

            total_qc_metrics_mat_inv = np.linalg.inv(total_qc_metrics.T.dot(total_qc_metrics))

            pred_slopes = total_qc_metrics_mat_inv.dot(total_qc_metrics[pred_mask,:].T).dot(z_scores[:,pred_mask].T)
            total_qc_metrics_betas = np.mean(pred_slopes, axis=1)
            log("Mean slopes are %s" % total_qc_metrics_betas, TRACE)

            if self.huge_gene_covariate_names is not None:
                param_names = ["%s_beta" % self.huge_gene_covariate_names[i] for i in range(len(self.huge_gene_covariate_names)) if i != self.huge_gene_covariate_intercept_index]
                param_values = total_qc_metrics_betas
                self._record_params(dict(zip(param_names, param_values)), record_only_first_time=True)


            if np.sum(remove_mask) > 0:
                beta_tildes = beta_tildes[:,~remove_mask]
                ses = ses[:,~remove_mask]
                z_scores = z_scores[:,~remove_mask]
                if se_inflation_factors is not None:
                    se_inflation_factors = se_inflation_factors[:,~remove_mask]
                total_qc_metrics = total_qc_metrics[~remove_mask,:]
                mean_qc_metrics = mean_qc_metrics[~remove_mask]

            #print("SLOPES",pred_slopes[~intercept_mask])
            #print("INT",pred_slopes[intercept_mask])

            #don't add intercept, because we may have input gene sets that just have an average value very high
            #don't want to correct for that, only the trend
            pred_values = total_qc_metrics[:,~intercept_mask].dot(pred_slopes[~intercept_mask]).T

            pred_residuals = z_scores - pred_values

            #ds = "mp_abnormal_Z_line_morphology"
            #if ds in self.gene_set_to_ind:
            #    ind = self.gene_set_to_ind[ds]
            #    print(ds,total_qc_metrics[ind,:])
            #    print(pred_values[:,ind])
            #    print(pred_residuals[:,ind])
            #    print(z_scores[:,ind])


            #only adjust those that are predicted to decrease AND do not have zero beta tildes
            inflate_mask = np.logical_and(np.abs(pred_residuals) < np.abs(z_scores), beta_tildes != 0)
            new_ses = copy.copy(ses)
            if np.sum(inflate_mask) > 0:
                log("Inflating %d standard errors" % np.sum(inflate_mask))

            new_ses[inflate_mask] = beta_tildes[inflate_mask] / pred_residuals[inflate_mask]

            if se_inflation_factors is not None:
                se_inflation_factors[inflate_mask] *= new_ses[inflate_mask] / ses[inflate_mask]

            ses = new_ses
     
        if beta_tildes.shape[0] == 1:
            beta_tildes = np.squeeze(beta_tildes, axis=0)
            ses = np.squeeze(ses, axis=0)
            if se_inflation_factors is not None:
                se_inflation_factors = np.squeeze(se_inflation_factors, axis=0)

        #in case original ses are zero
        zero_se_mask = ses == 0
        assert(np.sum(np.logical_and(zero_se_mask, beta_tildes != 0)) == 0)
        z_scores = np.zeros(beta_tildes.shape)
        z_scores[~zero_se_mask] = beta_tildes[~zero_se_mask] / ses[~zero_se_mask]
        p_values = 2*scipy.stats.norm.cdf(-np.abs(z_scores))


        return (beta_tildes, ses, z_scores, p_values, se_inflation_factors)

    def _calculate_inf_betas(self, beta_tildes=None, ses=None, V=None, V_cor=None, se_inflation_factors=None, V_inv=None, scale_factors=None, is_dense_gene_set=None):
        if V is None:
            bail("Require V")
        if beta_tildes is None:
            beta_tildes = self.beta_tildes
        if ses is None:
            ses = self.ses
        if scale_factors is None:
            scale_factors = self.scale_factors
        if is_dense_gene_set is None:
            is_dense_gene_set = self.is_dense_gene_set

        if V is None:
            bail("V is required for this operation")
        if beta_tildes is None:
            bail("Cannot calculate sigma with no stats loaded!")
        if self.sigma2 is None:
            bail("Need sigma to calculate betas!")

        log("Calculating infinitesimal betas")
        sigma2 = self.sigma2
        if self.sigma_power is not None:
            #sigma2 = self.sigma2 * np.power(scale_factors, self.sigma_power)
            sigma2 = self.get_scaled_sigma2(scale_factors, self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

            #for dense gene sets, scaling by size doesn't make sense. So use mean size across sparse gene sets
            if np.sum(is_dense_gene_set) > 0:
                if np.sum(~is_dense_gene_set) > 0:
                    #sigma2[is_dense_gene_set] = self.sigma2 * np.power(np.mean(scale_factors[~is_dense_gene_set]), self.sigma_power)
                    sigma2[is_dense_gene_set] = self.get_scaled_sigma2(np.mean(scale_factors[~is_dense_gene_set]), self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

                else:
                    #sigma2[is_dense_gene_set] = self.sigma2 * np.power(np.mean(scale_factors), self.sigma_power)
                    sigma2[is_dense_gene_set] = self.get_scaled_sigma2(np.mean(scale_factors), self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

        orig_shrinkage_fac=np.diag(np.square(ses)/sigma2)
        shrinkage_fac = orig_shrinkage_fac

        #handle corrected OLS case
        if V_cor is not None and se_inflation_factors is not None:
            if V_inv is None:
                V_inv = self._invert_sym_matrix(V)
            shrinkage_fac = V_cor.dot(V_inv).dot(shrinkage_fac / np.square(se_inflation_factors))
            shrinkage_inv = self._invert_matrix(V + shrinkage_fac)
            return shrinkage_inv.dot(beta_tildes)
        else:
            cho_factor = scipy.linalg.cho_factor(V + shrinkage_fac)
            return scipy.linalg.cho_solve(cho_factor, beta_tildes)

    #there are two levels of parallelization here:
    #1. num_chains: sample multiple independent chains with the same beta/se/V
    #2. multiple parallel runs with different beta/se (and potentially V). To do this, pass in lists of beta and se (must be the same length) and an optional list of V (V must have same length as beta OR must be not a list, in which case the same V will be used for all betas and ses

    #to run this in parallel, pass in two-dimensional matrix for beta_tildes (rows are parallel runs, columns are beta_tildes)
    #you can pass in multiple V as well with rows/columns mapping to gene sets and a first dimension mapping to parallel runs
    def _calculate_non_inf_betas(self, initial_p, return_sample=False, max_num_burn_in=None, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, eps=0.01, max_frac_sem=0.01, max_allowed_batch_correlation=None, beta_outlier_iqr_threshold=5, gauss_seidel=False, update_hyper_sigma=True, update_hyper_p=True, adjust_hyper_sigma_p=False, sigma_num_devs_to_top=2.0, p_noninf_inflate=1.0, num_p_pseudo=1, sparse_solution=False, sparse_frac_betas=None, betas_trace_out=None, betas_trace_gene_sets=None, beta_tildes=None, ses=None, V=None, X_orig=None, scale_factors=None, mean_shifts=None, is_dense_gene_set=None, ps=None, sigma2s=None, assume_independent=False, num_missing_gene_sets=None, debug_genes=None, debug_gene_sets=None):

        debug_gene_sets = None

        if max_num_burn_in is None:
            max_num_burn_in = int(max_num_iter * .25)
        if max_num_burn_in >= max_num_iter:
            max_num_burn_in = int(max_num_iter * .25)

        #if (update_hyper_p or update_hyper_sigma) and gauss_seidel:
        #    log("Using Gibbs sampling for betas since update hyper was requested")
        #    gauss_seidel = False

        if ses is None:
            ses = self.ses
        if beta_tildes is None:
            beta_tildes = self.beta_tildes
            
        if X_orig is None and not assume_independent:
            X_orig = self.X_orig
        if scale_factors is None:
            scale_factors = self.scale_factors
        if mean_shifts is None:
            mean_shifts = self.mean_shifts

        use_X = False
        if V is None and not assume_independent:
            if X_orig is None or scale_factors is None or mean_shifts is None:
                bail("Require X, scale, and mean if V is None")
            else:
                use_X = True
                log("Using low memory X instead of V", TRACE)

        if is_dense_gene_set is None:
            is_dense_gene_set = self.is_dense_gene_set
        if ps is None:
            ps = self.ps
        if sigma2s is None:
            sigma2s = self.sigma2s

        if self.sigma2 is None:
            bail("Need sigma to calculate betas!")

        if initial_p is not None:
            self.set_p(initial_p)

        if self.p is None and ps is None:
            bail("Need p to calculate non-inf betas")

        if not len(beta_tildes.shape) == len(ses.shape):
            bail("If running parallel beta inference, beta_tildes and ses must have same shape")

        if len(beta_tildes.shape) == 0 or beta_tildes.shape[0] == 0:
            bail("No gene sets are left!")

        #convert the beta_tildes and ses to matrices -- columns are num_parallel
        #they are always stored as matrices, with 1 column as needed
        #V on the other hand will be a 2-D matrix if it is constant across all parallel (or if there is only 1)
        #checking len(V.shape) can therefore distinguish a constant from variable V

        multiple_V = False
        sparse_V = False

        if len(beta_tildes.shape) > 1:
            num_gene_sets = beta_tildes.shape[1]

            if not beta_tildes.shape[0] == ses.shape[0]:
                bail("beta_tildes and ses must have same number of parallel runs")

            #dimensions should be num_gene_sets, num_parallel
            num_parallel = beta_tildes.shape[0]
            beta_tildes_m = copy.copy(beta_tildes)
            ses_m = copy.copy(ses)

            if V is not None and type(V) is sparse.csc_matrix:
                sparse_V = True
                multiple_V = False
            elif V is not None and len(V.shape) == 3:
                if not V.shape[0] == beta_tildes.shape[0]:
                    bail("V must have same number of parallel runs as beta_tildes")
                multiple_V = True
                sparse_V = False
            else:
                multiple_V = False
                sparse_V = False

        else:
            num_gene_sets = len(beta_tildes)
            if V is not None and type(V) is sparse.csc_matrix:
                num_parallel = 1
                multiple_V = False
                sparse_V = True
                beta_tildes_m = beta_tildes[np.newaxis,:]
                ses_m = ses[np.newaxis,:]
            elif V is not None and len(V.shape) == 3:
                num_parallel = V.shape[0]
                multiple_V = True
                sparse_V = False
                beta_tildes_m = np.tile(beta_tildes, num_parallel).reshape((num_parallel, len(beta_tildes)))
                ses_m = np.tile(ses, num_parallel).reshape((num_parallel, len(ses)))
            else:
                num_parallel = 1
                multiple_V = False
                sparse_V = False
                beta_tildes_m = beta_tildes[np.newaxis,:]
                ses_m = ses[np.newaxis,:]

        if num_parallel == 1 and multiple_V:
            multiple_V = False
            V = V[0,:,:]

        if multiple_V:
            assert(not use_X)

        if scale_factors.shape != mean_shifts.shape:
            bail("scale_factors must have same dimension as mean_shifts")

        if len(scale_factors.shape) == 2 and not scale_factors.shape[0] == num_parallel:
            bail("scale_factors must have same number of parallel runs as beta_tildes")
        elif len(scale_factors.shape) == 1 and num_parallel == 1:
            scale_factors_m = scale_factors[np.newaxis,:]
            mean_shifts_m = mean_shifts[np.newaxis,:]
        elif len(scale_factors.shape) == 1 and num_parallel > 1:
            scale_factors_m = np.tile(scale_factors, num_parallel).reshape((num_parallel, len(scale_factors)))
            mean_shifts_m = np.tile(mean_shifts, num_parallel).reshape((num_parallel, len(mean_shifts)))
        else:
            scale_factors_m = copy.copy(scale_factors)
            mean_shifts_m = copy.copy(mean_shifts)

        if len(is_dense_gene_set.shape) == 2 and not is_dense_gene_set.shape[0] == num_parallel:
            bail("is_dense_gene_set must have same number of parallel runs as beta_tildes")
        elif len(is_dense_gene_set.shape) == 1 and num_parallel == 1:
            is_dense_gene_set_m = is_dense_gene_set[np.newaxis,:]
        elif len(is_dense_gene_set.shape) == 1 and num_parallel > 1:
            is_dense_gene_set_m = np.tile(is_dense_gene_set, num_parallel).reshape((num_parallel, len(is_dense_gene_set)))
        else:
            is_dense_gene_set_m = copy.copy(is_dense_gene_set)

        if ps is not None:
            if len(ps.shape) == 2 and not ps.shape[0] == num_parallel:
                bail("ps must have same number of parallel runs as beta_tildes")
            elif len(ps.shape) == 1 and num_parallel == 1:
                ps_m = ps[np.newaxis,:]
            elif len(ps.shape) == 1 and num_parallel > 1:
                ps_m = np.tile(ps, num_parallel).reshape((num_parallel, len(ps)))
            else:
                ps_m = copy.copy(ps)
        else:
            ps_m = self.p

        if sigma2s is not None:
            if len(sigma2s.shape) == 2 and not sigma2s.shape[0] == num_parallel:
                bail("sigma2s must have same number of parallel runs as beta_tildes")
            elif len(sigma2s.shape) == 1 and num_parallel == 1:
                orig_sigma2_m = sigma2s[np.newaxis,:]
            elif len(sigma2s.shape) == 1 and num_parallel > 1:
                orig_sigma2_m = np.tile(sigma2s, num_parallel).reshape((num_parallel, len(sigma2s)))
            else:
                orig_sigma2_m = copy.copy(sigma2s)
        else:
            orig_sigma2_m = self.sigma2

        #for efficiency, batch genes to be updated each cycle
        if assume_independent:
            gene_set_masks = [np.full(beta_tildes_m.shape[1], True)]
        else:
            gene_set_masks = self._compute_gene_set_batches(V, X_orig=X_orig, mean_shifts=mean_shifts, scale_factors=scale_factors, use_sum=True, max_allowed_batch_correlation=max_allowed_batch_correlation)
            
        sizes = [float(np.sum(x)) / (num_parallel if multiple_V else 1) for x in gene_set_masks]
        log("Analyzing %d gene sets in %d batches of gene sets; size range %d - %d" % (num_gene_sets, len(gene_set_masks), min(sizes) if len(sizes) > 0 else 0, max(sizes)  if len(sizes) > 0 else 0), DEBUG)

        #get the dimensions of the gene_set_masks to match those of the betas
        if num_parallel == 1:
            assert(not multiple_V)
            #convert the vectors into matrices with one dimension
            gene_set_masks = [x[np.newaxis,:] for x in gene_set_masks]
        elif not multiple_V:
            #we have multiple parallel but only one V
            gene_set_masks = [np.tile(x, num_parallel).reshape((num_parallel, len(x))) for x in gene_set_masks]

        #variables are denoted
        #v: vectors of dimension equal to the number of gene sets
        #m: data that varies by parallel runs and gene sets
        #t: data that varies by chains, parallel runs, and gene sets

        #rules:
        #1. adding a lower dimensional tensor to higher dimenional ones means final dimensions must match. These operations are usually across replicates
        #2. lower dimensional masks on the other hand index from the beginning dimensions (can use :,:,mask to index from end)
        
        tensor_shape = (num_chains, num_parallel, num_gene_sets)
        matrix_shape = (num_parallel, num_gene_sets)

        #these are current posterior means (including p and the conditional beta). They are used to calculate avg_betas
        #using these as the actual betas would yield the Gauss-seidel algorithm
        curr_post_means_t = np.zeros(tensor_shape)
        curr_postp_t = np.ones(tensor_shape)

        #these are the current betas to be used in each iteration
        initial_sd = np.std(beta_tildes_m)
        if initial_sd == 0:
            initial_sd = 1

        curr_betas_t = scipy.stats.norm.rvs(0, initial_sd, tensor_shape)

        res_beta_hat_t = np.zeros(tensor_shape)

        avg_betas_m = np.zeros(matrix_shape)
        avg_betas2_m = np.zeros(matrix_shape)
        avg_postp_m = np.zeros(matrix_shape)
        num_avg = 0

        #these are the posterior betas averaged across iterations
        sum_betas_t = np.zeros(tensor_shape)
        sum_betas2_t = np.zeros(tensor_shape)

        # Setting up constants
        #hyperparameters
        #shrinkage prior
        if self.sigma_power is not None:
            #sigma2_m = orig_sigma2_m * np.power(scale_factors_m, self.sigma_power)
            sigma2_m = self.get_scaled_sigma2(scale_factors_m, orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

            #for dense gene sets, scaling by size doesn't make sense. So use mean size across sparse gene sets
            if np.sum(is_dense_gene_set_m) > 0:
                if np.sum(~is_dense_gene_set_m) > 0:
                    #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m[~is_dense_gene_set_m]), self.sigma_power)
                    sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m[~is_dense_gene_set_m]), self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
                else:
                    #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m), self.sigma_power)
                    sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m), self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

        else:
            sigma2_m = orig_sigma2_m

        if ps_m is not None and np.min(ps_m) != np.max(ps_m):
            p_text = "mean p=%.3g (%.3g-%.3g)" % (self.p, np.min(ps_m), np.max(ps_m))
        else:
            p_text = "p=%.3g" % (self.p)
        if np.min(orig_sigma2_m) != np.max(orig_sigma2_m):
            sigma2_text = "mean sigma=%.3g (%.3g-%.3g)" % (self.sigma2, np.min(orig_sigma2_m), np.max(orig_sigma2_m))
        else:
            sigma2_text = "sigma=%.3g" % (self.sigma2)

        if np.min(orig_sigma2_m) != np.max(orig_sigma2_m):
            sigma2_p_text = "mean sigma2/p=%.3g (%.3g-%.3g)" % (self.sigma2/self.p, np.min(orig_sigma2_m/ps_m), np.max(orig_sigma2_m/ps_m))
        else:
            sigma2_p_text = "sigma2/p=%.3g" % (self.sigma2/self.p)


        tag = ""
        if assume_independent:
            tag = "independent "
        elif sparse_V:
            tag = "partially independent "
            
        log("Calculating %snon-infinitesimal betas with %s, %s; %s" % (tag, p_text, sigma2_text, sigma2_p_text))

        #generate the diagonals to use per replicate
        if assume_independent:
            V_diag_m = None
            account_for_V_diag_m = False
        else:
            if V is not None:
                if num_parallel > 1:
                    #dimensions are num_parallel, num_gene_sets, num_gene_sets
                    if multiple_V:
                        V_diag_m = np.diagonal(V, axis1=1, axis2=2)
                    else:
                        if sparse_V:
                            V_diag = V.diagonal()
                        else:
                            V_diag = np.diag(V)
                        V_diag_m = np.tile(V_diag, num_parallel).reshape((num_parallel, len(V_diag)))
                else:
                    if sparse_V:
                        V_diag_m = V.diagonal()[np.newaxis,:]                        
                    else:
                        V_diag_m = np.diag(V)[np.newaxis,:]

                account_for_V_diag_m = not np.isclose(V_diag_m, np.ones(matrix_shape)).all()
            else:
                #we compute it from X, so we know it is always 1
                V_diag_m = None
                account_for_V_diag_m = False

        se2s_m = np.power(ses_m,2)

        #the below code is based off of the LD-pred code for SNP PRS
        iteration_num = 0
        burn_in_phase_v = np.array([True for i in range(num_parallel)])


        if betas_trace_out is not None:
            betas_trace_fh = open_gz(betas_trace_out, 'w')
            betas_trace_fh.write("It\tParallel\tChain\tGene_Set\tbeta_post\tbeta\tpostp\tres_beta_hat\tbeta_tilde\tbeta_internal\tres_beta_hat_internal\tbeta_tilde_internal\tse_internal\tsigma2\tp\tR\tR_weighted\tSEM\n")

        #print("TEST V_DIAG_M!")
        prev_betas_m = None
        sigma_underflow = False
        printed_warning_swing = False
        printed_warning_increase = False
        while iteration_num < max_num_iter:  #Big iteration

            #if some have not converged, only sample for those that have not converged (for efficiency)
            compute_mask_v = copy.copy(burn_in_phase_v)
            if np.sum(compute_mask_v) == 0:
                compute_mask_v[:] = True

            hdmp_m = (sigma2_m / ps_m)
            hdmpn_m = hdmp_m + se2s_m
            hdmp_hdmpn_m = (hdmp_m / hdmpn_m)

            #if iteration_num == 0:
            #    print(hdmp_hdmpn_m[0,:],(sigma2_m / ps_m)[0,:],se2s_m[0,:])

            norm_scale_m = np.sqrt(np.multiply(hdmp_hdmpn_m, se2s_m))
            c_const_m = (ps_m / np.sqrt(hdmpn_m))

            d_const_m = (1 - ps_m) / ses_m

            iteration_num += 1

            #default to 1
            curr_postp_t[:,compute_mask_v,:] = np.ones(tensor_shape)[:,compute_mask_v,:]

            #sample whether each gene set has non-zero effect
            rand_ps_t = np.random.random(tensor_shape)
            #generate normal random variable sampling
            rand_norms_t = scipy.stats.norm.rvs(0, 1, tensor_shape)

            for gene_set_mask_ind in range(len(gene_set_masks)):

                #the challenge here is that gene_set_mask_m produces a ragged (non-square) tensor
                #so we are going to "flatten" the last two dimensions
                #this requires some care, in particular when running einsum, which requires a square tensor

                gene_set_mask_m = gene_set_masks[gene_set_mask_ind]
                
                if debug_gene_sets is not None:
                    cur_debug_gene_sets = [debug_gene_sets[i] for i in range(len(debug_gene_sets)) if gene_set_mask_m[0,i]]

                #intersect compute_max_v with the rows of gene_set_mask (which are the parallel runs)
                compute_mask_m = np.logical_and(compute_mask_v, gene_set_mask_m.T).T

                current_num_parallel = sum(compute_mask_v)

                #Value to use when determining if we should force an alpha shrink if estimates are way off compared to heritability estimates.  (Improves MCMC convergence.)
                #zero_jump_prob=0.05
                #frac_betas_explained = max(0.00001,np.sum(np.apply_along_axis(np.mean, 0, np.power(curr_betas_m,2)))) / self.y_var
                #frac_sigma_explained = self.sigma2_total_var / self.y_var
                #alpha_shrink = min(1 - zero_jump_prob, 1.0 / frac_betas_explained, (frac_sigma_explained + np.mean(np.power(ses[i], 2))) / frac_betas_explained)
                alpha_shrink = 1

                #subtract out the predicted effects of the other betas
                #we need to zero out diagonal of V to do this, but rather than do this we will add it back in

                #1. First take the union of the current_gene_set_mask
                #this is to allow us to run einsum
                #we are going to do it across more gene sets than are needed, and then throw away the computations that are extra for each batch
                compute_mask_union = np.any(compute_mask_m, axis=0)

                #2. Retain how to filter from the union down to each mask
                compute_mask_union_filter_m = compute_mask_m[:,compute_mask_union]

                if assume_independent:
                    res_beta_hat_t_flat = beta_tildes_m[compute_mask_m]
                else:
                    if multiple_V:

                        #3. Do einsum across the union
                        #This does pointwise matrix multiplication of curr_betas_t (sliced on axis 1) with V (sliced on axis 0), maintaining axis 0 for curr_betas_t
                        res_beta_hat_union_t = np.einsum('hij,ijk->hik', curr_betas_t[:,compute_mask_v,:], V[compute_mask_v,:,:][:,:,compute_mask_union]).reshape((num_chains, current_num_parallel, np.sum(compute_mask_union)))

                    elif sparse_V:
                        res_beta_hat_union_t = V[compute_mask_union,:].dot(curr_betas_t[:,compute_mask_v,:].T.reshape((curr_betas_t.shape[2], np.sum(compute_mask_v) * curr_betas_t.shape[0]))).reshape((np.sum(compute_mask_union), np.sum(compute_mask_v), curr_betas_t.shape[0])).T
                    elif use_X:
                        if len(compute_mask_union.shape) == 2:
                            assert(compute_mask_union.shape[0] == 1)
                            compute_mask_union = np.squeeze(compute_mask_union)
                        #curr_betas_t: (num_chains, num_parallel, num_gene_sets)
                        #X_orig: (num_genes, num_gene_sets)
                        #X_orig_t: (num_gene_sets, num_genes)
                        #mean_shifts_m: (num_parallel, num_gene_sets)

                        curr_betas_filtered_t = curr_betas_t[:,compute_mask_v,:] / scale_factors_m[compute_mask_v,:]

                        #have to reshape latter two dimensions before multiplying because sparse matrix can only handle 2-D

                        interm = X_orig.dot(curr_betas_filtered_t.T.reshape((curr_betas_filtered_t.shape[2],curr_betas_filtered_t.shape[0] * curr_betas_filtered_t.shape[1]))).reshape((X_orig.shape[0],curr_betas_t.shape[1],curr_betas_t.shape[0])) - np.sum(mean_shifts_m * curr_betas_filtered_t, axis=2).T
                        #interm: (num_genes, num_parallel, num_chains)

                        #num_gene sets, num_parallel, num_chains
                        res_beta_hat_union_t = (X_orig[:,compute_mask_union].T.dot(interm.reshape((interm.shape[0],interm.shape[1]*interm.shape[2]))).reshape((np.sum(compute_mask_union),interm.shape[1],interm.shape[2])) - mean_shifts_m.T[compute_mask_union,:][:,compute_mask_v,np.newaxis] * np.sum(interm, axis=0)).T

                        res_beta_hat_union_t /= (X_orig.shape[0] * scale_factors_m[compute_mask_v,:][:,compute_mask_union])
                    else:
                        res_beta_hat_union_t = curr_betas_t[:,compute_mask_v,:].dot(V[:,compute_mask_union])

                    if betas_trace_out is not None and betas_trace_gene_sets is not None:
                        all_map = self._construct_map_to_ind(betas_trace_gene_sets)
                        cur_sets = [betas_trace_gene_sets[x] for x in range(len(betas_trace_gene_sets)) if compute_mask_union[x]]
                        cur_map = self._construct_map_to_ind(cur_sets)

                    #4. Now restrict to only the actual masks (which flattens things because the compute_mask_m is not square)

                    res_beta_hat_t_flat = res_beta_hat_union_t[:,compute_mask_union_filter_m[compute_mask_v,:]]
                    assert(res_beta_hat_t_flat.shape[1] == np.sum(compute_mask_m))

                    #dimensions of res_beta_hat_t_flat are (num_chains, np.sum(compute_mask_m))
                    #dimensions of beta_tildes_m are (num_parallel, num_gene_sets))
                    #subtraction will subtract matrix from each of the matrices in the tensor

                    res_beta_hat_t_flat = beta_tildes_m[compute_mask_m] - res_beta_hat_t_flat

                    if account_for_V_diag_m:
                        #dimensions of V_diag_m are (num_parallel, num_gene_sets)
                        #curr_betas_t is (num_chains, num_parallel, num_gene_sets)
                        res_beta_hat_t_flat = res_beta_hat_t_flat + V_diag_m[compute_mask_m] * curr_betas_t[:,compute_mask_m]
                    else:
                        res_beta_hat_t_flat = res_beta_hat_t_flat + curr_betas_t[:,compute_mask_m]
                
                b2_t_flat = np.power(res_beta_hat_t_flat, 2)
                d_const_b2_exp_t_flat = d_const_m[compute_mask_m] * np.exp(-b2_t_flat / (se2s_m[compute_mask_m] * 2.0))
                numerator_t_flat = c_const_m[compute_mask_m] * np.exp(-b2_t_flat / (2.0 * hdmpn_m[compute_mask_m]))
                numerator_zero_mask_t_flat = (numerator_t_flat == 0)
                denominator_t_flat = numerator_t_flat + d_const_b2_exp_t_flat
                denominator_t_flat[numerator_zero_mask_t_flat] = 1


                d_imaginary_mask_t_flat = ~np.isreal(d_const_b2_exp_t_flat)
                numerator_imaginary_mask_t_flat = ~np.isreal(numerator_t_flat)

                if np.any(np.logical_or(d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)):

                    warn("Detected imaginary numbers!")
                    #if d is imaginary, we set it to 1
                    denominator_t_flat[d_imaginary_mask_t_flat] = numerator_t_flat[d_imaginary_mask_t_flat]
                    #if d is real and numerator is imaginary, we set to 0 (both numerator and denominator will be imaginary)
                    numerator_t_flat[np.logical_and(~d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)] = 0

                    #Original code for handling edge cases; adapted above
                    #Commenting these out for now, but they are here in case we ever detect non real numbers
                    #if need them, masked_array is too inefficient -- change to real mask
                    #d_real_mask_t = np.isreal(d_const_b2_exp_t)
                    #numerator_real_mask_t = np.isreal(numerator_t)
                    #curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_not(d_real_mask_t), fill_value = 1).filled()
                    #curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_and(d_real_mask_t, np.logical_not(numerator_real_mask_t)), fill_value=0).filled()
                    #curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_and(np.logical_and(d_real_mask_t, numerator_real_mask_t), numerator_zero_mask_t), fill_value=0).filled()



                curr_postp_t[:,compute_mask_m] = (numerator_t_flat / denominator_t_flat)


                #calculate current posterior means
                #the left hand side, because it is masked, flattens the latter two dimensions into one
                #so we flatten the result of the right hand size to a 1-D array to match up for the assignment
                curr_post_means_t[:,compute_mask_m] = hdmp_hdmpn_m[compute_mask_m] * (curr_postp_t[:,compute_mask_m] * res_beta_hat_t_flat)

                   
                if gauss_seidel:
                    proposed_beta_t_flat = curr_post_means_t[:,compute_mask_m]
                else:
                    norm_mean_t_flat = hdmp_hdmpn_m[compute_mask_m] * res_beta_hat_t_flat

                    #draw from the conditional distribution
                    proposed_beta_t_flat = norm_mean_t_flat + norm_scale_m[compute_mask_m] * rand_norms_t[:,compute_mask_m]

                    #set things to zero that sampled below p
                    zero_mask_t_flat = rand_ps_t[:,compute_mask_m] >= curr_postp_t[:,compute_mask_m] * alpha_shrink
                    proposed_beta_t_flat[zero_mask_t_flat] = 0

                #update betas
                #do this inside loop since this determines the res_beta
                #same idea as above for collapsing
                curr_betas_t[:,compute_mask_m] = proposed_beta_t_flat
                res_beta_hat_t[:,compute_mask_m] = res_beta_hat_t_flat

                #if debug_gene_sets is not None:
                #    my_cur_tensor_shape = (1 if assume_independent else num_chains, current_num_parallel, np.sum(gene_set_mask_m[0,]))
                #    my_cur_tensor_shape2 = (num_chains, current_num_parallel, np.sum(gene_set_mask_m[0,]))
                #    my_res_beta_hat_t = res_beta_hat_t_flat.reshape(my_cur_tensor_shape)
                #    my_proposed_beta_t = proposed_beta_t_flat.reshape(my_cur_tensor_shape2)
                #    my_norm_mean_t = norm_mean_t_flat.reshape(my_cur_tensor_shape)
                #    top_set = [cur_debug_gene_sets[i] for i in range(len(cur_debug_gene_sets)) if np.abs(my_res_beta_hat_t[0,0,i]) == np.max(np.abs(my_res_beta_hat_t[0,0,:]))][0]
                #    print("TOP IS",top_set)
                #    gs = set([ "mp_absent_T_cells", top_set])
                #    ind = [i for i in range(len(cur_debug_gene_sets)) if cur_debug_gene_sets[i] in gs]
                #    for i in ind:
                #        print("BETA_TILDE",cur_debug_gene_sets[i],beta_tildes_m[0,i]/scale_factors_m[0,i])
                #        print("Z",cur_debug_gene_sets[i],beta_tildes_m[0,i]/ses_m[0,i])
                #        print("RES",cur_debug_gene_sets[i],my_res_beta_hat_t[0,0,i]/scale_factors_m[0,i])
                #        #print("RESF",cur_debug_gene_sets[i],res_beta_hat_t_flat[i]/scale_factors_m[0,i])
                #        print("NORM_MEAN",cur_debug_gene_sets[i],my_norm_mean_t[0,0,i])
                #        print("NORM_SCALE_M",cur_debug_gene_sets[i],norm_scale_m[0,i])
                #        print("RAND_NORMS",cur_debug_gene_sets[i],rand_norms_t[0,0,i])
                #        print("PROP",cur_debug_gene_sets[i],my_proposed_beta_t[0,0,i]/scale_factors_m[0,i])
                #        ind2 = [j for j in range(len(debug_gene_sets)) if debug_gene_sets[j] == cur_debug_gene_sets[i]]
                #        for j in ind2:
                #            print("POST",cur_debug_gene_sets[i],curr_post_means_t[0,0,j]/scale_factors_m[0,i])
                #            print("SIGMA",sigma2_m if type(sigma2_m) is float or type(sigma2_m) is np.float64 else sigma2_m[0,i])
                #            print("P",cur_debug_gene_sets[i],curr_postp_t[0,0,j],self.p)
                #            print("HDMP",hdmp_m/np.square(scale_factors_m[0,i]) if type(hdmp_m) is float or type(hdmp_m) is np.float64 else hdmp_m[0,0]/np.square(scale_factors_m[0,i]))
                #            print("SES",se2s_m[0,0]/np.square(scale_factors_m[0,i]))
                #            print("HDMPN",hdmpn_m/np.square(scale_factors_m[0,i]) if type(hdmpn_m) is float or type(hdmpn_m) is np.float64 else hdmpn_m[0,0]/scale_factors_m[0,i])
                #            print("HDMP_HDMPN",hdmp_hdmpn_m if type(hdmp_hdmpn_m) is float or type(hdmp_hdmpn_m) is np.float64 else hdmp_hdmpn_m[0,0])
                #            print("NOW1",debug_gene_sets[j],curr_betas_t[0,0,j]/scale_factors_m[0,i])


            if sparse_solution:
                sparse_mask_t = curr_postp_t < ps_m

                if sparse_frac_betas is not None:
                    #zero out very small values relative to top or median
                    relative_value = np.max(np.abs(curr_post_means_t), axis=2)
                    sparse_mask_t = np.logical_or(sparse_mask_t, (np.abs(curr_post_means_t).T < sparse_frac_betas * relative_value.T).T)

                #don't set anything not currently computed
                sparse_mask_t[:,np.logical_not(compute_mask_v),:] = False
                log("Setting %d entries to zero due to sparsity" % (np.sum(np.logical_and(sparse_mask_t, curr_betas_t > 0))), TRACE)
                curr_betas_t[sparse_mask_t] = 0
                curr_post_means_t[sparse_mask_t] = 0

                if debug_gene_sets is not None:
                    ind = [i for i in range(len(debug_gene_sets)) if debug_gene_sets[i] in gs]
                    for i in ind:
                        print("SPARSE",debug_gene_sets[i],sparse_mask_t[0,0,i])
                        print("NOW2",debug_gene_sets[i],curr_betas_t[0,0,i]/scale_factors_m[0,i])

            curr_betas_m = np.mean(curr_post_means_t, axis=0)
            curr_postp_m = np.mean(curr_postp_t, axis=0)
            #no state should be preserved across runs, but take a random one just in case
            sample_betas_m = curr_betas_t[int(random.random() * curr_betas_t.shape[0]),:,:]
            sample_postp_m = curr_postp_t[int(random.random() * curr_postp_t.shape[0]),:,:]
            sum_betas_t[:,compute_mask_v,:] = sum_betas_t[:,compute_mask_v,:] + curr_post_means_t[:,compute_mask_v,:]
            sum_betas2_t[:,compute_mask_v,:] = sum_betas2_t[:,compute_mask_v,:] + np.square(curr_post_means_t[:,compute_mask_v,:])

            #now calculate the convergence metrics
            R_m = np.zeros(matrix_shape)
            beta_weights_m = np.zeros(matrix_shape)
            sem2_m = np.zeros(matrix_shape)
            will_break = False
            if assume_independent:
                burn_in_phase_v[:] = False
            elif gauss_seidel:
                if prev_betas_m is not None:
                    sum_diff = np.sum(np.abs(prev_betas_m - curr_betas_m))
                    sum_prev = np.sum(np.abs(prev_betas_m))
                    tot_diff = sum_diff / sum_prev
                    log("Iteration %d: gauss seidel difference = %.4g / %.4g = %.4g" % (iteration_num+1, sum_diff, sum_prev, tot_diff), TRACE)
                    if iteration_num > min_num_iter and tot_diff < eps:
                        burn_in_phase_v[:] = False
                        log("Converged after %d iterations" % (iteration_num+1), INFO)
                prev_betas_m = curr_betas_m
            elif iteration_num > min_num_iter and np.sum(burn_in_phase_v) > 0:
                def __calculate_R_tensor(sum_t, sum2_t, num, print_pc=None):

                    #mean of betas across all iterations; psi_dot_j
                    mean_t = sum_t / float(num)

                    if print_pc is not None:
                        print(mean_t[print_pc[1],print_pc[0],:10])

                    #mean of betas across replicates; psi_dot_dot
                    mean_m = np.mean(mean_t, axis=0)
                    #variances of betas across all iterators; s_j
                    var_t = (sum2_t - float(num) * np.power(mean_t, 2)) / (float(num) - 1)
                    #B_v = (float(iteration_num) / (num_chains - 1)) * np.apply_along_axis(np.sum, 0, np.apply_along_axis(lambda x: np.power(x - mean_betas_v, 2), 1, mean_betas_m))
                    B_m = (float(num) / (mean_t.shape[0] - 1)) * np.sum(np.power(mean_t - mean_m, 2), axis=0)
                    W_m = (1.0 / float(mean_t.shape[0])) * np.sum(var_t, axis=0)
                    avg_W_m = (1.0 / float(mean_t.shape[2])) * np.sum(var_t, axis=2)
                    var_given_y_m = np.add((float(num) - 1) / float(num) * W_m, (1.0 / float(num)) * B_m)
                    var_given_y_m[var_given_y_m < 0] = 0

                    R_m = np.ones(W_m.shape)
                    R_non_zero_mask_m = W_m > 0

                    var_given_y_m[var_given_y_m < 0] = 0

                    R_m[R_non_zero_mask_m] = np.sqrt(var_given_y_m[R_non_zero_mask_m] / W_m[R_non_zero_mask_m])
                    
                    return (B_m, W_m, R_m, avg_W_m, mean_t)

                #these matrices have convergence statistics in format (num_parallel, num_gene_sets)
                #WARNING: only the results for compute_mask_v are valid
                (B_m, W_m, R_m, avg_W_m, mean_t) = __calculate_R_tensor(sum_betas_t, sum_betas2_t, iteration_num)

                beta_weights_m = np.zeros((sum_betas_t.shape[1], sum_betas_t.shape[2]))
                sum_betas_t_mean = np.mean(sum_betas_t)
                if sum_betas_t_mean > 0:
                    np.mean(sum_betas_t, axis=0) / sum_betas_t_mean

                #calculate the thresholded / scaled R_v
                num_R_above_1_v = np.sum(R_m >= 1, axis=1)
                num_R_above_1_v[num_R_above_1_v == 0] = 1

                #mean for each parallel run

                R_m_above_1 = copy.copy(R_m)
                R_m_above_1[R_m_above_1 < 1] = 0
                mean_thresholded_R_v = np.sum(R_m_above_1, axis=1) / num_R_above_1_v

                #max for each parallel run
                max_index_v = np.argmax(R_m, axis=1)
                max_index_parallel = None
                max_val = None
                for i in range(len(max_index_v)):
                    if compute_mask_v[i] and (max_val is None or R_m[i,max_index_v[i]] > max_val):
                        max_val = R_m[i,max_index_v[i]]
                        max_index_parallel = i
                max_R_v = np.max(R_m, axis=1)
               

                #TEMP TEMP TEMP
                #if priors_for_convergence:
                #    curr_v = curr_betas_v
                #    s_cur2_v = np.array([curr_v[i] for i in sorted(range(len(curr_v)), key=lambda k: -np.abs(curr_v[k]))])
                #    s_cur2_v = np.square(s_cur2_v - np.mean(s_cur2_v))
                #    cum_cur2_v = np.cumsum(s_cur2_v) / np.sum(s_cur2_v)
                #    top_mask2 = np.array(cum_cur2_v < 0.99)
                #    (B_v2, W_v2, R_v2) = __calculate_R(sum_betas_m[:,top_mask2], sum_betas2_m[:,top_mask2], iteration_num)
                #    max_index2 = np.argmax(R_v2)
                #    log("Iteration %d (betas): max ind=%d; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g" % (iteration_num, max_index2, B_v2[max_index2], W_v2[max_index2], R_v2[max_index2], np.mean(R_v2), np.sum(R_v2 > r_threshold_burn_in)), TRACE)
                #END TEMP TEMP TEMP
                    
                if use_max_r_for_convergence:
                    convergence_statistic_v = max_R_v
                else:
                    convergence_statistic_v = mean_thresholded_R_v

                outlier_mask_m = np.full(avg_W_m.shape, False)
                if avg_W_m.shape[0] > 10:
                    #check the variances
                    q3, median, q1 = np.percentile(avg_W_m, [75, 50, 25], axis=0)
                    iqr_mask = q3 > q1
                    chain_iqr_m = np.zeros(avg_W_m.shape)
                    chain_iqr_m[:,iqr_mask] = (avg_W_m[:,iqr_mask] - median[iqr_mask]) / (q3 - q1)[iqr_mask]
                    #dimensions chain x parallel
                    outlier_mask_m = beta_outlier_iqr_threshold
                    if np.sum(outlier_mask_m) > 0:
                        log("Detected %d outlier chains due to oscillations" % np.sum(outlier_mask_m), DEBUG)

                if np.sum(R_m > 1) > 10:
                    #check the Rs
                    q3, median, q1 = np.percentile(R_m[R_m > 1], [75, 50, 25])
                    if q3 > q1:
                        #Z score per parallel, gene
                        R_iqr_m = (R_m - median) / (q3 - q1)
                        #dimensions of parallel x gene sets
                        bad_gene_sets_m = np.logical_and(R_iqr_m > 100, R_m > 2.5)
                        bad_gene_sets_v = np.any(bad_gene_sets_m,0)
                        if np.sum(bad_gene_sets_m) > 0:
                            #now find the bad chains
                            bad_chains = np.argmax(np.abs(mean_t - np.mean(mean_t, axis=0)), axis=0)[bad_gene_sets_m]

                            #np.where bad gene sets[0] lists parallel
                            #bad chains lists the bad chain corresponding to each parallel
                            cur_outlier_mask_m = np.zeros(outlier_mask_m.shape)
                            cur_outlier_mask_m[bad_chains, np.where(bad_gene_sets_m)[0]] = True

                            log("Found %d outlier chains across %d parallel runs due to %d gene sets with high R (%.4g - %.4g; %.4g - %.4g)" % (np.sum(cur_outlier_mask_m), np.sum(np.any(cur_outlier_mask_m, axis=0)), np.sum(bad_gene_sets_m), np.min(R_m[bad_gene_sets_m]), np.max(R_m[bad_gene_sets_m]), np.min(R_iqr_m[bad_gene_sets_m]), np.max(R_iqr_m[bad_gene_sets_m])), DEBUG)
                            outlier_mask_m = np.logical_or(outlier_mask_m, cur_outlier_mask_m)

                            #log("Outlier parallel: %s" % (np.where(bad_gene_sets_m)[0]), DEBUG)
                            #log("Outlier values: %s" % (R_m[bad_gene_sets_m]), DEBUG)
                            #log("Outlier IQR: %s" % (R_iqr_m[bad_gene_sets_m]), DEBUG)
                            #log("Outlier chains: %s" % (bad_chains), DEBUG)


                            #log("Actually in mask: %s" % (str(np.where(outlier_mask_m))))

                non_outliers_m = ~outlier_mask_m
                if np.sum(outlier_mask_m) > 0:
                    log("Detected %d total outlier chains" % np.sum(outlier_mask_m), DEBUG)
                    #dimensions are num_chains x num_parallel
                    for outlier_parallel in np.where(np.any(outlier_mask_m, axis=0))[0]:
                        #find a non-outlier chain and replace the three matrices in the right place
                        if np.sum(outlier_mask_m[:,outlier_parallel]) > 0:
                            if np.sum(non_outliers_m[:,outlier_parallel]) > 0:
                                replacement_chains = np.random.choice(np.where(non_outliers_m[:,outlier_parallel])[0], size=np.sum(outlier_mask_m[:,outlier_parallel]))
                                log("Replaced chains %s with chains %s in parallel %d" % (np.where(outlier_mask_m[:,outlier_parallel])[0], replacement_chains, outlier_parallel), DEBUG)

                                #print(sum_betas_t[np.where(outlier_mask_m[:,outlier_parallel])[0][0],outlier_parallel,:10])
                                #(B_m, W_m, R_m, avg_W_m, mean_t) = __calculate_R_tensor(sum_betas_t, sum_betas2_t, iteration_num, print_pc=(outlier_parallel, np.where(outlier_mask_m[:,outlier_parallel])[0][0]))
                                #print(np.max(R_m))

                                for tensor in [curr_betas_t, curr_postp_t, curr_post_means_t, sum_betas_t, sum_betas2_t]:
                                    tensor[outlier_mask_m[:,outlier_parallel],outlier_parallel,:] = copy.copy(tensor[replacement_chains,outlier_parallel,:])

                                #print(sum_betas_t[np.where(outlier_mask_m[:,outlier_parallel])[0][0],outlier_parallel,:10])
                                #(B_m, W_m, R_m, avg_W_m, mean_t) = __calculate_R_tensor(sum_betas_t, sum_betas2_t, iteration_num, print_pc=(outlier_parallel, np.where(outlier_mask_m[:,outlier_parallel])[0][0]))
                                #print(np.max(R_m))

                            else:
                                log("Every chain was an outlier so doing nothing", TRACE)


                log("Iteration %d: max ind=%s; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;" % (iteration_num, (max_index_parallel, max_index_v[max_index_parallel]) if num_parallel > 1 else max_index_v[max_index_parallel], B_m[max_index_parallel, max_index_v[max_index_parallel]], W_m[max_index_parallel, max_index_v[max_index_parallel]], R_m[max_index_parallel, max_index_v[max_index_parallel]], np.mean(mean_thresholded_R_v), np.sum(R_m > r_threshold_burn_in)), TRACE)

                converged_v = convergence_statistic_v < r_threshold_burn_in
                newly_converged_v = np.logical_and(burn_in_phase_v, converged_v)
                if np.sum(newly_converged_v) > 0:
                    if num_parallel == 1:
                        log("Converged after %d iterations" % iteration_num, INFO)
                    else:
                        log("Parallel %s converged after %d iterations" % (",".join([str(p) for p in np.nditer(np.where(newly_converged_v))]), iteration_num), INFO)
                    burn_in_phase_v = np.logical_and(burn_in_phase_v, np.logical_not(converged_v))

            if sum(burn_in_phase_v) == 0 or iteration_num >= max_num_burn_in:

                if return_sample:

                    frac_increase = np.sum(np.abs(curr_betas_m) > np.abs(beta_tildes_m)) / curr_betas_m.size
                    if frac_increase > 0.01:
                        warn("A large fraction of betas (%.3g) are larger than beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_increase)
                        printed_warning_increase = True

                    frac_opposite = np.sum(curr_betas_m * beta_tildes_m < 0) / curr_betas_m.size
                    if frac_opposite > 0.01:
                        warn("A large fraction of betas (%.3g) are of opposite signs than the beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_opposite)
                        printed_warning_swing = False

                    if sum(burn_in_phase_v) > 0:
                        burn_in_phase_v[:] = False
                        log("Stopping burn in after %d iterations" % (iteration_num), INFO)


                    #max_beta = None
                    #if max_beta is not None:
                    #    threshold_ravel = max_beta * scale_factors_m.ravel()
                    #    if np.sum(sample_betas_m.ravel() > threshold_ravel) > 0:
                    #        log("Capped %d sample betas" % np.sum(sample_betas_m.ravel() > threshold_ravel), DEBUG)
                    #        sample_betas_mask = sample_betas_m.ravel() > threshold_ravel
                    #        sample_betas_m.ravel()[sample_betas_mask] = threshold_ravel[sample_betas_mask]
                    #    if np.sum(curr_betas_m.ravel() > threshold_ravel) > 0:
                    #        log("Capped %d curr betas" % np.sum(curr_betas_m.ravel() > threshold_ravel), DEBUG)
                    #        curr_betas_mask = curr_betas_m.ravel() > threshold_ravel
                    #        curr_betas_m.ravel()[curr_betas_mask] = threshold_ravel[curr_betas_mask]

                    return (sample_betas_m, sample_postp_m, curr_betas_m, curr_postp_m)

                #average over the posterior means instead of samples
                #these differ from sum_betas_v because those include the burn in phase
                avg_betas_m += np.sum(curr_post_means_t, axis=0)
                avg_betas2_m += np.sum(np.power(curr_post_means_t, 2), axis=0)
                avg_postp_m += np.sum(curr_postp_t, axis=0)
                num_avg += curr_post_means_t.shape[0]

                if iteration_num >= min_num_iter and num_avg > 1:
                    if gauss_seidel:
                        will_break = True
                    else:

                        #calculate these here for trace printing
                        avg_m = avg_betas_m
                        avg2_m = avg_betas2_m
                        sem2_m = ((avg2_m / (num_avg - 1)) - np.power(avg_m / num_avg, 2)) / num_avg
                        sem2_v = np.sum(sem2_m, axis=0)
                        zero_sem2_v = sem2_v == 0
                        sem2_v[zero_sem2_v] = 1
                        total_z_v = np.sqrt(np.sum(avg2_m / num_avg, axis=0) / sem2_v)
                        total_z_v[zero_sem2_v] = np.inf

                        log("Iteration %d: sum2=%.4g; sum sem2=%.4g; z=%.3g" % (iteration_num, np.sum(avg2_m / num_avg), np.sum(sem2_m), np.min(total_z_v)), TRACE)
                        min_z_sampling_var = 10
                        if np.all(total_z_v > min_z_sampling_var):
                            log("Desired precision achieved; stopping sampling")
                            will_break=True

                        #TODO: STILL FINALIZING HOW TO DO THIS
                        #avg_m = avg_betas_m
                        #avg2_m = avg_betas2_m

                        #sem2_m = ((avg2_m / (num_avg - 1)) - np.power(avg_m / num_avg, 2)) / num_avg
                        #zero_sem2_m = sem2_m == 0
                        #sem2_m[zero_sem2_m] = 1

                        #max_avg = np.max(np.abs(avg_m / num_avg))
                        #min_avg = np.min(np.abs(avg_m / num_avg))
                        #ref_val = max_avg - min_avg
                        #if ref_val == 0:
                        #    ref_val = np.sqrt(np.var(curr_post_means_t))
                        #    if ref_val == 0:
                        #        ref_val = 1

                        #max_sem = np.max(np.sqrt(sem2_m))
                        #max_percentage_error = max_sem / ref_val

                        #log("Iteration %d: ref_val=%.3g; max_sem=%.3g; max_ratio=%.3g" % (iteration_num, ref_val, max_sem, max_percentage_error))
                        #if max_percentage_error < max_frac_sem:
                        #    log("Desired precision achieved; stopping sampling")
                        #    break
                        
            else:
                if update_hyper_p or update_hyper_sigma:
                    h2 = 0
                    for i in range(num_parallel):
                        if use_X:
                            h2 += curr_betas_m[i,:].dot(curr_betas_m[i,:])
                        else:
                            if multiple_V:
                                cur_V = V[i,:,:]
                            else:
                                cur_V = V
                            if sparse_V:
                                h2 += V.dot(curr_betas_m[i,:].T).T.dot(curr_betas_m[i,:])
                            else:
                                h2 += curr_betas_m[i,:].dot(cur_V).dot(curr_betas_m[i,:])
                    h2 /= num_parallel

                    new_p = np.mean((np.sum(curr_betas_t > 0, axis=2) + num_p_pseudo) / float(curr_betas_t.shape[2] + num_p_pseudo))

                    if self.sigma_power is not None:
                        new_sigma2 = h2 / np.mean(np.sum(np.power(scale_factors_m, self.sigma_power), axis=1))
                    else:
                        new_sigma2 = h2 / num_gene_sets

                    if num_missing_gene_sets:
                        missing_scale_factor = num_gene_sets / (num_gene_sets + num_missing_gene_sets)
                        new_sigma2 *= missing_scale_factor
                        new_p *= missing_scale_factor

                    if p_noninf_inflate != 1:
                        log("Inflating p by %.3g" % p_noninf_inflate, DEBUG)
                        new_p *= p_noninf_inflate

                    if abs(new_sigma2 - self.sigma2) / self.sigma2 < eps and abs(new_p - self.p) / self.p < eps:
                        log("Sigma converged to %.4g; p converged to %.4g" % (self.sigma2, self.p), TRACE)
                        update_hyper_sigma = False
                        update_hyper_p = False
                    else:
                        if update_hyper_p:
                            log("Updating p from %.4g to %.4g" % (self.p, new_p), TRACE)
                            if not update_hyper_sigma and adjust_hyper_sigma_p:
                                #remember, sigma is the *total* variance term. It is equal to p * conditional_sigma.
                                #if we are only updating p, and adjusting sigma, we will leave the conditional_sigma constant, which means scaling the sigma
                                new_sigma2 = self.sigma2 / self.p * new_p
                                log("Updating sigma from %.4g to %.4g to maintain constant sigma/p" % (self.sigma2, new_sigma2), TRACE)
                                #we need to adjust the total sigma to keep the conditional sigma constant
                                self.set_sigma(new_sigma2, self.sigma_power)
                            self.set_p(new_p)
                                
                        if update_hyper_sigma:
                            if not sigma_underflow:
                                log("Updating sigma from %.4g to %.4g ( sqrt(sigma2/p)=%.4g )" % (self.sigma2, new_sigma2, np.sqrt(new_sigma2 / self.p)), TRACE)

                            lower_bound = 2e-3

                            if sigma_underflow or new_sigma2 / self.p < lower_bound:
                                
                                #first, try the heuristic of setting sigma2 so that strongest gene set has maximum possible p_bar

                                max_e_beta2 = np.argmax(beta_tildes_m / ses_m)

                                max_se2 = se2s_m.ravel()[max_e_beta2]
                                max_beta_tilde = beta_tildes_m.ravel()[max_e_beta2]
                                max_beta_tilde2 = np.square(max_beta_tilde)

                                #make sigma/p easily cover the observation
                                new_sigma2 = (max_beta_tilde2 - max_se2) * self.p

                                #make sigma a little bit smaller so that the top gene set is a little more of an outlier
                                new_sigma2 /= sigma_num_devs_to_top

                                if new_sigma2 / self.p <= lower_bound:
                                    new_sigma2 = lower_bound * self.p
                                    log("Sigma underflow! Setting sigma to lower bound (%.4g * %.4g = %.4g) and no updates" % (lower_bound, self.p, new_sigma2), TRACE)
                                else:
                                    log("Sigma underflow! Setting sigma determined from top gene set (%.4g) and no updates" % new_sigma2, TRACE)

                                if self.sigma_power is not None:

                                    #gene set specific sigma is internal sigma2 multiplied by scale_factor ** power
                                    #new_sigma2 is final sigma
                                    #so store internal value as final divided by average power

                                    #use power learned from mouse
                                    #using average across gene sets makes it sensitive to distribution of gene sets
                                    #need better solution for learning; since we are hardcoding from top gene set, just use mouse value
                                    new_sigma2 = new_sigma2 / np.power(self.MEAN_MOUSE_SCALE, self.sigma_power)

                                    #if np.sum([~is_dense_gene_set_m]) > 0:
                                    #    new_sigma2 = new_sigma2 / np.power(np.mean(scale_factors_m[~is_dense_gene_set_m].ravel()), self.sigma_power)
                                    #else:
                                    #    new_sigma2 = new_sigma2 / np.power(np.mean(scale_factors_m.ravel()), self.sigma_power)

                                    #if is_dense_gene_set_m.ravel()[max_e_beta2]:
                                    #    new_sigma2 = new_sigma2 / np.power(np.mean(scale_factors_m[~is_dense_gene_set_m].ravel()), self.sigma_power)
                                    #else:
                                    #    new_sigma2 = new_sigma2 / np.power(scale_factors_m.ravel()[max_e_beta2], self.sigma_power)

                                if not update_hyper_p and adjust_hyper_sigma_p:
                                    #remember, sigma is the *total* variance term. It is equal to p * conditional_sigma.
                                    #if we are only sigma p, and adjusting p, we will leave the conditional_sigma constant, which means scaling the p
                                    new_p = self.p / self.sigma2 * new_sigma2
                                    log("Updating p from %.4g to %.4g to maintain constant sigma/p" % (self.p, new_p), TRACE)
                                    #we need to adjust the total sigma to keep the conditional sigma constant
                                    self.set_p(new_p)

                                self.set_sigma(new_sigma2, self.sigma_power)
                                sigma_underflow = True

                                #update_hyper_sigma = False
                                #restarting sampling with sigma2 fixed to initial value due to underflow
                                #update_hyper_p = False

                                #reset loop state
                                #iteration_num = 0
                                #curr_post_means_t = np.zeros(tensor_shape)
                                #curr_postp_t = np.ones(tensor_shape)
                                #curr_betas_t = scipy.stats.norm.rvs(0, np.std(beta_tildes_m), tensor_shape)                            
                                #avg_betas_m = np.zeros(matrix_shape)
                                #avg_betas2_m = np.zeros(matrix_shape)
                                #avg_postp_m = np.zeros(matrix_shape)
                                #num_avg = 0
                                #sum_betas_t = np.zeros(tensor_shape)
                                #sum_betas2_t = np.zeros(tensor_shape)
                            else:
                                self.set_sigma(new_sigma2, self.sigma_power)

                            #update the matrix forms of these variables
                            orig_sigma2_m *= new_sigma2 / np.mean(orig_sigma2_m)
                            if self.sigma_power is not None:
                                #sigma2_m = orig_sigma2_m * np.power(scale_factors_m, self.sigma_power)
                                sigma2_m = self.get_scaled_sigma2(scale_factors_m, orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

                                #for dense gene sets, scaling by size doesn't make sense. So use mean size across sparse gene sets
                                if np.sum(is_dense_gene_set_m) > 0:
                                    if np.sum(~is_dense_gene_set_m) > 0:
                                        #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m[~is_dense_gene_set_m]), self.sigma_power)
                                        sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m[~is_dense_gene_set_m]), orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
                                    else:
                                        #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m), self.sigma_power)
                                        sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m), orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
                            else:
                                sigma2_m = orig_sigma2_m

                            ps_m *= new_p / np.mean(ps_m)

            if betas_trace_out is not None:
                for parallel_num in range(num_parallel):
                    for chain_num in range(num_chains):
                        for i in range(num_gene_sets):
                            gene_set = i
                            if betas_trace_gene_sets is not None and len(betas_trace_gene_sets) == num_gene_sets:
                                gene_set = betas_trace_gene_sets[i]

                            betas_trace_fh.write("%d\t%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n" % (iteration_num, parallel_num+1, chain_num+1, gene_set, curr_post_means_t[chain_num,parallel_num,i] / scale_factors_m[parallel_num,i], curr_betas_t[chain_num,parallel_num,i] / scale_factors_m[parallel_num,i], curr_postp_t[chain_num,parallel_num,i], res_beta_hat_t[chain_num,parallel_num,i] / scale_factors_m[parallel_num,i], beta_tildes_m[parallel_num,i] / scale_factors_m[parallel_num,i], curr_betas_t[chain_num,parallel_num,i], res_beta_hat_t[chain_num,parallel_num,i], beta_tildes_m[parallel_num,i], ses_m[parallel_num,i], sigma2_m[parallel_num,i] if len(np.shape(sigma2_m)) > 0 else sigma2_m, ps_m[parallel_num,i] if len(np.shape(ps_m)) > 0 else ps_m, R_m[parallel_num,i], R_m[parallel_num,i] * beta_weights_m[parallel_num,i], sem2_m[parallel_num, i]))

                betas_trace_fh.flush()

            if will_break:
                break


        if betas_trace_out is not None:
            betas_trace_fh.close()

            #log("%d\t%s" % (iteration_num, "\t".join(["%.3g\t%.3g" % (curr_betas_m[i,0], (np.mean(sum_betas_m, axis=0) / iteration_num)[i]) for i in range(curr_betas_m.shape[0])])), TRACE)

        avg_betas_m /= num_avg
        avg_postp_m /= num_avg

        if num_parallel == 1:
            avg_betas_m = avg_betas_m.flatten()
            avg_postp_m = avg_postp_m.flatten()

        #max_beta = None
        #if max_beta is not None:
        #    threshold_ravel = max_beta * scale_factors_m.ravel()
        #    if np.sum(avg_betas_m.ravel() > threshold_ravel) > 0:
        #        log("Capped %d sample betas" % np.sum(avg_betas_m.ravel() > threshold_ravel), DEBUG)
        #        avg_betas_mask = avg_betas_m.ravel() > threshold_ravel
        #        avg_betas_m.ravel()[avg_betas_mask] = threshold_ravel[avg_betas_mask]

        frac_increase = np.sum(np.abs(curr_betas_m) > np.abs(beta_tildes_m)) / curr_betas_m.size
        if frac_increase > 0.01:
            warn("A large fraction of betas (%.3g) are larger than beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_increase)
            printed_warning_increase = True

        frac_opposite = np.sum(curr_betas_m * beta_tildes_m < 0) / curr_betas_m.size
        if frac_opposite > 0.01:
            warn("A large fraction of betas (%.3g) are of opposite signs than the beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_opposite)
            printed_warning_swing = False

        return (avg_betas_m, avg_postp_m)


    #store Y value
    #Y is whitened if Y_corr_m is not null
    def _set_Y(self, Y, Y_for_regression=None, Y_exomes=None, Y_positive_controls=None, Y_corr_m=None, store_cholesky=True, store_corr_sparse=False, skip_V=False, skip_scale_factors=False, min_correlation=0):
        log("Setting Y", TRACE)

        self.last_X_block = None
        if Y_corr_m is not None:

            if min_correlation is not None:
                #set things too far away to 0
                Y_corr_m[Y_corr_m <= 0] = 0

            #remove bands at the end that are all zero
            keep_mask = np.array([True] * len(Y_corr_m))
            for i in range(len(Y_corr_m)-1, -1, -1):
                if sum(Y_corr_m[i] != 0) == 0:
                    keep_mask[i] = False
                else:
                    break
            if sum(keep_mask) > 0:
                Y_corr_m = Y_corr_m[keep_mask]

            #scale factor for diagonal to ensure non-singularity
            self.y_corr = copy.copy(Y_corr_m)

            y_corr_diags = [self.y_corr[i,:(len(self.y_corr[i,:]) - i)] for i in range(len(self.y_corr))]
            y_corr_sparse = sparse.csc_matrix(sparse.diags(y_corr_diags + y_corr_diags[1:], list(range(len(y_corr_diags))) + list(range(-1, -len(y_corr_diags), -1))))            

            if store_cholesky:
                self.y_corr_cholesky = self._get_y_corr_cholesky(Y_corr_m)
                log("Banded cholesky matrix: shape %s, %s" % (self.y_corr_cholesky.shape[0], self.y_corr_cholesky.shape[1]), DEBUG)
                #whitened
                self.Y_w = scipy.linalg.solve_banded((self.y_corr_cholesky.shape[0]-1, 0), self.y_corr_cholesky, Y)
                na_mask = ~np.isnan(self.Y_w)
                self.y_w_var = np.var(self.Y_w[na_mask])
                self.y_w_mean = np.mean(self.Y_w[na_mask])
                self.Y_w = self.Y_w - self.y_w_mean
                #fully whitened
                self.Y_fw = scipy.linalg.cho_solve_banded((self.y_corr_cholesky, True), Y)
                na_mask = ~np.isnan(self.Y_fw)
                self.y_fw_var = np.var(self.Y_fw[na_mask])
                self.y_fw_mean = np.mean(self.Y_fw[na_mask])
                self.Y_fw = self.Y_fw - self.y_fw_mean

                #update the scale factors and mean shifts for the whitened X
                self._set_X(self.X_orig, self.genes, self.gene_sets, skip_V=skip_V, skip_scale_factors=skip_scale_factors, skip_N=True)
                if self.X_orig_missing_gene_sets is not None and not skip_scale_factors:
            
                    (self.mean_shifts_missing, self.scale_factors_missing) = self._calc_X_shift_scale(self.X_orig_missing_gene_sets, self.y_corr_cholesky)

            if store_corr_sparse:

                self.y_corr_sparse = y_corr_sparse

        na_mask = ~np.isnan(Y)
        self.y_var = np.var(Y[na_mask])
        #DO WE NEED THIS???
        #self.y_mean = np.mean(Y[na_mask])
        #self.Y = Y - self.y_mean
        self.Y = Y
        self.Y_for_regression = Y_for_regression
        self.Y_exomes = Y_exomes
        self.Y_positive_controls = Y_positive_controls

    def _get_y_corr_cholesky(self, Y_corr_m):
        Y_corr_m_copy = copy.copy(Y_corr_m)
        diag_add = 0.05
        while True:
            try:
                Y_corr_m_copy[0,:] += diag_add
                Y_corr_m_copy /= (1 + diag_add)
                y_corr_cholesky = scipy.linalg.cholesky_banded(Y_corr_m_copy, lower=True)
                return y_corr_cholesky
            except np.linalg.LinAlgError:
                pass

    def _whiten(self, matrix, corr_cholesky, whiten=True, full_whiten=False):
        if full_whiten:
            #fully whiten, by sigma^{-1}; useful for optimization
            matrix = scipy.linalg.cho_solve_banded((corr_cholesky, True), matrix, overwrite_b=True)
        elif whiten:
            #whiten X_b by sigma^{-1/2}
            matrix = scipy.linalg.solve_banded((corr_cholesky.shape[0]-1, 0), corr_cholesky, matrix, overwrite_ab=True)
        return matrix

    #return an iterator over chunks of X in dense format
    #useful when want to conduct matrix calculations for which dense arrays are much faster, but don't have enough memory to cast all of X to dense
    #full_whiten (which multiplies by C^{-1} takes precedence over whiten, which multiplies by C^{1/2}, but whiten defaults to true
    #if mean_shifts/scale_factors are passed in, then shift/rescale the blocks. This is done *before* any whitening
    def _get_X_blocks(self, whiten=True, full_whiten=False, get_missing=False, start_batch=0, mean_shifts=None, scale_factors=None):
        X_orig = self.X_orig
        if get_missing:
            X_orig = self.X_orig_missing_gene_sets

        for (X_b, begin, end, batch) in self._get_X_blocks_internal(X_orig, self.y_corr_cholesky, whiten=whiten, full_whiten=full_whiten, start_batch=start_batch, mean_shifts=mean_shifts, scale_factors=scale_factors):
            yield (X_b, begin, end, batch)

    def _get_num_X_blocks(self, X_orig, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return int(np.ceil(X_orig.shape[1] / batch_size))

    def _get_X_size_mb(self, X_orig=None):
        if X_orig is None:
            X_orig = self.X_orig
        return (self.X_orig.data.nbytes + self.X_orig.indptr.nbytes + self.X_orig.indices.nbytes) / 1024 / 1024

    def _get_X_blocks_internal(self, X_orig, y_corr_cholesky, whiten=True, full_whiten=False, start_batch=0, mean_shifts=None, scale_factors=None):

        if y_corr_cholesky is None:
            #explicitly turn these off to help with caching
            whiten = False
            full_whiten = False

        num_batches = self._get_num_X_blocks(X_orig)

        consider_cache = X_orig is self.X_orig and num_batches == 1 and mean_shifts is None and scale_factors is None

        for batch in range(start_batch, num_batches):
            log("Getting X%s block batch %s (%s)" % ("_missing" if X_orig is self.X_orig_missing_gene_sets else "", batch, "fully whitened" if full_whiten else ("whitened" if whiten else "original")), TRACE)
            begin = batch * self.batch_size
            end = (batch + 1) * self.batch_size
            if end > X_orig.shape[1]:
                end = X_orig.shape[1]

            if self.last_X_block is not None and consider_cache and self.last_X_block[1:] == (whiten, full_whiten, begin, end, batch):
                log("Using cache!", TRACE)
                yield (self.last_X_block[0], begin, end, batch)
            else:
                X_b = X_orig[:,begin:end].toarray()
                if mean_shifts is not None:
                    X_b = X_b - mean_shifts[begin:end]
                if scale_factors is not None:
                    X_b = X_b / scale_factors[begin:end]

                if whiten or full_whiten:
                    X_b = self._whiten(X_b, y_corr_cholesky, whiten=whiten, full_whiten=full_whiten)

                #only cache if we are accessing the original X
                if consider_cache:
                    self.last_X_block = (X_b, whiten, full_whiten, begin, end, batch)
                else:
                    self.last_X_block = None

                yield (X_b, begin, end, batch)

    def _get_fraction_non_missing(self):
        if self.gene_sets_missing is not None and self.gene_sets is not None:
            fraction_non_missing = float(len(self.gene_sets)) / float(len(self.gene_sets_missing) + len(self.gene_sets))        
        else:
            fraction_non_missing = 1
        return fraction_non_missing

    # TODO returns mean shoft/scale factors    
    def _calc_X_shift_scale(self, X, y_corr_cholesky=None):
        if y_corr_cholesky is None:
            mean_shifts = X.sum(axis=0).A1 / X.shape[0]
            scale_factors = np.sqrt(X.power(2).sum(axis=0).A1 / X.shape[0] - np.square(mean_shifts))
        else:
            scale_factors = np.array([])
            mean_shifts = np.array([])
            for X_b, begin, end, batch in self._get_X_blocks_internal(X, y_corr_cholesky):
                (cur_mean_shifts, cur_scale_factors) = self._calc_shift_scale(X_b)
                mean_shifts = np.append(mean_shifts, cur_mean_shifts)
                scale_factors = np.append(scale_factors, cur_scale_factors)
        return (mean_shifts, scale_factors)

    def _calc_shift_scale(self, X_b):
        mean_shifts = []
        scale_factors = []
        for i in range(X_b.shape[1]):
            X_i = X_b[:,i]
            mean_shifts.append(np.mean(X_i))
            scale_factor = np.std(X_i)
            if scale_factor == 0:
                scale_factor = 1
            scale_factors.append(scale_factor)
        return (np.array(mean_shifts), np.array(scale_factors))

    #store a (possibly unnormalized) X matrix
    #the passed in X should be a sparse matrix, with 0/1 values
    #does normalization
    def _set_X(self, X_orig, genes, gene_sets, skip_V=False, skip_scale_factors=False, skip_N=True):

        log("Setting X", TRACE)

        if X_orig is not None:
            if not len(genes) == X_orig.shape[0]:
                bail("Dimension mismatch when setting X: %d genes but %d rows in X" % (len(genes), X_orig.shape[0]))
            if not len(gene_sets) == X_orig.shape[1]:
                bail("Dimension mismatch when setting X: %d gene sets but %d columns in X" % (len(gene_sets), X_orig.shape[1]))

        if self.X_orig is not None and X_orig is self.X_orig and genes is self.genes and gene_sets is self.gene_sets and ((self.y_corr_cholesky is None and not self.scale_is_for_whitened) or (self.y_corr_cholesky is not None and self.scale_is_for_whitened)):
            return
        
        self.last_X_block = None

        self.genes = genes

        if self.genes is not None:
            self.gene_to_ind = self._construct_map_to_ind(self.genes)
        else:
            self.gene_to_ind = None

        self.gene_sets = gene_sets
        if self.gene_sets is not None:
            self.gene_set_to_ind = self._construct_map_to_ind(self.gene_sets)
        else:
            self.gene_set_to_ind = None

        self.X_orig = X_orig
        if self.X_orig is not None:
            self.X_orig.eliminate_zeros()

        if self.X_orig is None:
            self.X_orig_missing_genes = None
            self.X_orig_missing_genes_missing_gene_sets = None
            self.X_orig_missing_gene_sets = None
            self.last_X_block = None
            return

        if not skip_N:
            self.gene_N = self.get_col_sums(self.X_orig, axis=1)

        #self.X = self.X_orig.todense().astype(float)
        if not skip_scale_factors:
            self._set_scale_factors()

        #X = self.X_orig.todense().astype(float)
        #for i in range(self.X_orig.shape[1]):
        #    X[:,i] = (X[:,i] - self.mean_shifts[i]) / self.scale_factors[i]
        #self.V = X.T.dot(X) / len(self.genes)

    def _set_scale_factors(self):

        log("Calculating scale factors and mean shifts", TRACE)
        (self.mean_shifts, self.scale_factors) = self._calc_X_shift_scale(self.X_orig, self.y_corr_cholesky)

        #flag to indicate whether these scale factors correspond to X_orig or the (implicit) whitened version
        if self.y_corr_cholesky is not None:
            self.scale_is_for_whitened = True
        else:
            self.scale_is_for_whitened = False

    def _get_V(self):
        if self.X_orig is not None:
            log("Calculating internal V", TRACE)
            return self._calculate_V()
        else:
            return None

    def _calculate_V(self, X_orig=None, y_corr_cholesky=None, mean_shifts=None, scale_factors=None):
        if X_orig is None:
            X_orig = self.X_orig
        if mean_shifts is None:
            mean_shifts = self.mean_shifts
        if scale_factors is None:
            scale_factors = self.scale_factors
        if y_corr_cholesky is None:
            y_corr_cholesky = self.y_corr_cholesky
        return self._calculate_V_internal(X_orig, y_corr_cholesky, mean_shifts, scale_factors)

    def _calculate_V_internal(self, X_orig, y_corr_cholesky, mean_shifts, scale_factors, y_corr_sparse=None):
        log("Calculating V for X with dimensions %d x %d" % (X_orig.shape[0], X_orig.shape[1]), TRACE)

        #TEMP
        if y_corr_cholesky is not None:

            if self._get_num_X_blocks(X_orig) == 1:
                whiten1 = True
                full_whiten1 = False
                whiten2 = True
                full_whiten2 = False
            else:
                whiten1 = False
                full_whiten1 = True
                whiten2 = False
                full_whiten2 = False

            V = None
            if X_orig is not None:
                #X_b is whitened

                for X_b1, begin1, end1, batch1 in self._get_X_blocks_internal(X_orig, y_corr_cholesky, whiten=whiten1, full_whiten=full_whiten1, mean_shifts=mean_shifts, scale_factors=scale_factors):
                    cur_V = None

                    if y_corr_sparse is not None:
                        X_b1 = y_corr_sparse.dot(X_b1)

                    for X_b2, begin2, end2, batch2 in self._get_X_blocks_internal(X_orig, y_corr_cholesky, whiten=whiten2, full_whiten=full_whiten2, mean_shifts=mean_shifts, scale_factors=scale_factors):
                        V_block = self._compute_V(X_b1, 0, 1, X_orig2=X_b2, mean_shifts2=0, scale_factors2=1)
                        if cur_V is None:
                            cur_V = V_block
                        else:
                            cur_V = np.hstack((cur_V, V_block))
                    if V is None:
                        V = cur_V
                    else:
                        V = np.vstack((V, cur_V))
        else:
            V = self._compute_V(X_orig, mean_shifts, scale_factors)

        return V

    #calculate V between X_orig and X_orig2
    #X_orig2 can be dense or sparse, but if it is sparse than X_orig must also be sparse
    def _compute_V(self, X_orig, mean_shifts, scale_factors, rows = None, X_orig2 = None, mean_shifts2 = None, scale_factors2 = None):
        if X_orig2 is None:
            X_orig2 = X_orig
        if mean_shifts2 is None:
            mean_shifts2 = mean_shifts
        if scale_factors2 is None:
            scale_factors2 = scale_factors
        if rows is None:
            if type(X_orig) is np.ndarray or type(X_orig2) is np.ndarray:
                dot_product = X_orig.T.dot(X_orig2)                
            else:
                dot_product = X_orig.T.dot(X_orig2).toarray().astype(float)
        else:
            if type(X_orig) is np.ndarray or type(X_orig2) is np.ndarray:
                dot_product = X_orig[:,rows].T.dot(X_orig2)
            else:
                dot_product = X_orig[:,rows].T.dot(X_orig2).toarray().astype(float)
            mean_shifts = mean_shifts[rows]
            scale_factors = scale_factors[rows]

        return (dot_product/X_orig.shape[0] - np.outer(mean_shifts, mean_shifts2)) / np.outer(scale_factors, scale_factors2)
            
    #by default, find batches of uncorrelated genes (for use in gibbs)
    #option to find batches of correlated (pass in batch size)
    #this is a greedy addition method
    #if have sort_values, will greedily add from lowest value to higher value
    def _compute_gene_set_batches(self, V=None, X_orig=None, mean_shifts=None, scale_factors=None, use_sum=True, max_allowed_batch_correlation=None, find_correlated_instead=None, sort_values=None, stop_at=None):
        gene_set_masks = []

        if max_allowed_batch_correlation is None:
            if use_sum:
                max_allowed_batch_correlation = 0.5
            else:
                max_allowed_batch_correlation = 0.1

        if find_correlated_instead is not None:
            if find_correlated_instead < 1:
                bail("Need batch size of at least 1")

        if use_sum:
            combo_fn = np.sum
        else:
            combo_fn = np.max

        use_X = False
        if V is not None and len(V.shape) == 3:
            num_gene_sets = V.shape[1]
            not_included_gene_sets = np.full((V.shape[0], num_gene_sets), True)
        elif V is not None:
            num_gene_sets = V.shape[0]
            not_included_gene_sets = np.full(num_gene_sets, True)
        else:
            assert(mean_shifts.shape == scale_factors.shape)
            if len(mean_shifts.shape) > 1:
                mean_shifts = np.squeeze(mean_shifts, axis=0)
                scale_factors = np.squeeze(scale_factors, axis=0)
            if X_orig is None or mean_shifts is None or scale_factors is None:
                bail("Need X_orig or V for this operation")
            num_gene_sets = X_orig.shape[1]
            not_included_gene_sets = np.full(num_gene_sets, True)
            use_X = True

        log("Batching %d gene sets..." % num_gene_sets, INFO)
        if use_X:
            log("Using low memory mode", DEBUG)

        indices = np.array(range(num_gene_sets))

        if sort_values is None:
            sort_values = indices

        total_added = 0

        while np.any(not_included_gene_sets):
            if V is not None and len(V.shape) == 3:
                #batches if multiple_V

                current_mask = np.full((V.shape[0], num_gene_sets), False)
                #set the first gene set in each row to True
                for c in range(V.shape[0]):

                    sorted_remaining_indices = sorted(indices[not_included_gene_sets[c,:]], key=lambda k: sort_values[k])
                    #seed with the first gene not already included
                    if len(sorted_remaining_indices) == 0:
                        continue

                    first_gene_set = sorted_remaining_indices[0]
                    current_mask[c,first_gene_set] = True
                    not_included_gene_sets[c,first_gene_set] = False
                    sorted_remaining_indices = sorted_remaining_indices[1:]

                    if find_correlated_instead:
                        #WARNING: THIS HAS NOT BEEN TESTED
                        #sort by decreasing V
                        index_map = np.where(not_included_gene_sets[c,:])[0]
                        ordered_indices = index_map[np.argsort(-V[c,first_gene_set,:])[not_included_gene_sets[c,:]]]
                        indices_to_add = ordered_indices[:find_correlated_instead]
                        current_mask[c,indices_to_add] = True
                    else:
                        for i in sorted_remaining_indices:
                            if combo_fn(V[c,i,current_mask[c,:]]) < max_allowed_batch_correlation:
                                current_mask[c,i] = True
                                not_included_gene_sets[c,i] = False
            else:
                sorted_remaining_indices = sorted(indices[not_included_gene_sets], key=lambda k: sort_values[k])
                #batches if one V
                current_mask = np.full(num_gene_sets, False)
                #seed with the first gene not already included
                first_gene_set = sorted_remaining_indices[0]
                current_mask[first_gene_set] = True
                not_included_gene_sets[first_gene_set] = False
                sorted_remaining_indices = sorted_remaining_indices[1:]

                if V is not None:
                    if find_correlated_instead:
                        #sort by decreasing V
                        index_map = np.where(not_included_gene_sets)[0]
                        ordered_indices = index_map[np.argsort(-V[first_gene_set,not_included_gene_sets])]
                        #map these to the original ones
                        indices_to_add = ordered_indices[:find_correlated_instead]
                        current_mask[indices_to_add] = True
                        not_included_gene_sets[indices_to_add] = False
                    else:
                        for i in sorted_remaining_indices:
                            if combo_fn(V[i,current_mask]) < max_allowed_batch_correlation:
                                current_mask[i] = True
                                not_included_gene_sets[i] = False
                else:
                    assert(scale_factors.shape == mean_shifts.shape)

                    if find_correlated_instead:
                        cur_V = self._compute_V(X_orig[:,first_gene_set], mean_shifts[first_gene_set], scale_factors[first_gene_set], X_orig2=X_orig[:,not_included_gene_sets], mean_shifts2=mean_shifts[not_included_gene_sets], scale_factors2=scale_factors[not_included_gene_sets])
                        #sort by decreasing V
                        index_map = np.where(not_included_gene_sets)[0]
                        ordered_indices = index_map[np.argsort(-cur_V[0,:])]
                        indices_to_add = ordered_indices[:find_correlated_instead]
                        current_mask[indices_to_add] = True
                        not_included_gene_sets[indices_to_add] = False
                    else:
                        #cap out at batch_size gene sets to avoid memory of making whole V; this may reduce the batch size relative to optimal
                        #also, only add those not in mask already (since we are searching only these in V)
                        max_to_add = self.batch_size
                        V_to_generate_mask = copy.copy(not_included_gene_sets)
                        if np.sum(V_to_generate_mask) > max_to_add:
                            assert(len(sorted_remaining_indices) == np.sum(not_included_gene_sets))
                            V_to_generate_mask[sort_values > sort_values[sorted_remaining_indices[max_to_add]]] = False

                        V_to_generate_mask[first_gene_set] = True
                        cur_V = self._compute_V(X_orig[:,V_to_generate_mask], mean_shifts[V_to_generate_mask], scale_factors[V_to_generate_mask])
                        indices_not_included = indices[V_to_generate_mask]
                        sorted_cur_V_indices = sorted(range(cur_V.shape[0]), key=lambda k: sort_values[indices_not_included[k]])
                        for i in sorted_cur_V_indices:
                            if combo_fn(cur_V[i,current_mask[V_to_generate_mask]]) < max_allowed_batch_correlation:
                                current_mask[indices_not_included[i]] = True
                                not_included_gene_sets[indices_not_included[i]] = False

            gene_set_masks.append(current_mask)
            #log("Batch %d; %d gene sets" % (len(gene_set_masks), sum(current_mask)), TRACE)
            total_added += np.sum(current_mask)
            if stop_at is not None and total_added >= stop_at:
                log("Breaking at %d" % total_added, TRACE)
                break

        denom = 1
        if V is not None and len(V.shape) == 3:
            denom = V.shape[0]

        sizes = [float(np.sum(x)) / denom for x in gene_set_masks]
        log("Batched %d gene sets into %d batches; size range %d - %d" % (num_gene_sets, len(gene_set_masks), min(sizes) if len(sizes) > 0 else 0, max(sizes)  if len(sizes) > 0 else 0), DEBUG)

        return gene_set_masks

    #sort the genes in the matrices
    #does not alter genes already subseet
    def _sort_genes(self, sorted_gene_indices, skip_V=False, skip_scale_factors=False):

        log("Sorting genes", TRACE)
        if self.y_corr_cholesky is not None:
            #FIXME: subset the cholesky matrix here
            bail("Sorting genes after setting correlation matrix is not yet implemented")

        self.genes = [self.genes[i] for i in sorted_gene_indices]
        self.gene_to_ind = self._construct_map_to_ind(self.genes)

        index_map = {sorted_gene_indices[i]: i for i in range(len(sorted_gene_indices))}

        if self.X_orig is not None:
            #reset the X matrix and scale factors
            self._set_X(sparse.csc_matrix((self.X_orig.data, [index_map[x] for x in self.X_orig.indices], self.X_orig.indptr), shape=self.X_orig.shape), self.genes, self.gene_sets, skip_V=skip_V, skip_scale_factors=skip_scale_factors, skip_N=True)

        if self.X_orig_missing_gene_sets is not None:
            #if we've already removed gene sets, then we need remove the genes from them too
            
            self.X_orig_missing_gene_sets = sparse.csc_matrix((self.X_orig_missing_gene_sets.data, [index_map[x] for x in self.X_orig_missing_gene_sets.indices], self.X_orig_missing_gene_sets.indptr), shape=self.X_orig_missing_gene_sets.shape)
            #need to recompute these
            (self.mean_shifts_missing, self.scale_factors_missing) = self._calc_X_shift_scale(self.X_orig_missing_gene_sets, self.y_corr_cholesky)

        if self.huge_signal_bfs is not None:
            #reset the X matrix and scale factors
            self.huge_signal_bfs = sparse.csc_matrix((self.huge_signal_bfs.data, [index_map[x] for x in self.huge_signal_bfs.indices], self.huge_signal_bfs.indptr), shape=self.huge_signal_bfs.shape)

        if self.huge_gene_covariates is not None:
            index_map_rev = {i: sorted_gene_indices[i] for i in range(len(sorted_gene_indices))}
            self.huge_gene_covariates = self.huge_gene_covariates[[index_map_rev[x] for x in range(self.huge_gene_covariates.shape[0])],:]
            self.huge_gene_covariate_zs = self.huge_gene_covariate_zs[[index_map_rev[x] for x in range(self.huge_gene_covariate_zs.shape[0])],:]

        if self.huge_gene_covariates_mask is not None:
            index_map_rev = {i: sorted_gene_indices[i] for i in range(len(sorted_gene_indices))}
            self.huge_gene_covariates_mask = self.huge_gene_covariates_mask[[index_map_rev[x] for x in range(self.huge_gene_covariates_mask.shape[0])]]

        if self.gene_N is not None:
            self.gene_N = self.gene_N[sorted_gene_indices]
        if self.gene_ignored_N is not None:
            self.gene_ignored_N = self.gene_ignored_N[sorted_gene_indices]

        for x in [self.Y, self.Y_for_regression, self.Y_exomes, self.Y_positive_controls, self.Y_w, self.Y_fw, self.priors, self.priors_adj, self.combined_prior_Ys, self.combined_prior_Ys_adj, self.combined_Ds, self.Y_orig, self.Y_for_regression_orig, self.Y_w_orig, self.Y_fw_orig, self.priors_orig, self.priors_adj_orig]:
            if x is not None:
                x[:] = np.array([x[i] for i in sorted_gene_indices])

    def _prune_gene_sets(self, prune_value, max_size=5000, keep_missing=False, ignore_missing=False, skip_V=False):
        if self.gene_sets is None or len(self.gene_sets) == 0:
            return
        if self.X_orig is None:
            return

        keep_mask = np.array([False] * len(self.gene_sets))
        remove_gene_sets = set()

        #keep total to batch_size ** 2

        batch_size = int(max_size ** 2 / self.X_orig.shape[1])
        num_batches = int(self.X_orig.shape[1] / batch_size) + 1

        for batch in range(num_batches):
            begin = batch * batch_size
            end = (batch + 1) * batch_size
            if end > self.X_orig.shape[1]:
                end = self.X_orig.shape[1]

            X_b1  = self.X_orig[:,begin:end]

            V_block = self._compute_V(self.X_orig[:,begin:end], self.mean_shifts[begin:end], self.scale_factors[begin:end], X_orig2=self.X_orig, mean_shifts2=self.mean_shifts, scale_factors2=self.scale_factors)

            if self.p_values is not None and False:
                gene_set_key = lambda i: self.p_values[i]
            else:
                gene_set_key = lambda i: np.abs(X_b1[:,i]).sum(axis=0)

            for gene_set_ind in sorted(range(len(self.gene_sets[begin:end])), key=gene_set_key):
                absolute_ind = gene_set_ind + begin
                if absolute_ind in remove_gene_sets:
                    continue
                keep_mask[absolute_ind] = True
                remove_gene_sets.update(np.where(np.abs(V_block[gene_set_ind,:]) > prune_value)[0])
        if np.sum(~keep_mask) > 0:
            self._subset_gene_sets(keep_mask, keep_missing=keep_missing, ignore_missing=ignore_missing, skip_V=skip_V)
            log("Pruning at %.3g resulted in %d gene sets" % (prune_value, len(self.gene_sets)))

    def _subset_genes(self, gene_mask, skip_V=False, overwrite_missing=False, skip_scale_factors=False, skip_Y=False):

        if not overwrite_missing and sum(np.logical_not(gene_mask)) == 0:
            return
       
        log("Subsetting genes", TRACE)

        if overwrite_missing:
            self.genes_missing = None
            self.priors__missing = None
            self.gene_N_missing = None
            self.gene_ignored_N_missing = None
            self.X_orig_missing_genes = None
            self.X_orig_missing_genes_missing_gene_sets = None

        self.genes_missing = (self.genes_missing if self.genes_missing is not None else []) + [self.genes[i] for i in range(len(self.genes)) if not gene_mask[i]]

        self.gene_missing_to_ind = self._construct_map_to_ind(self.genes_missing)
        
        if self.priors is not None:
            self.priors_missing = (self.priors_missing if self.priors_missing is not None else []) + [self.priors[i] for i in range(len(self.priors)) if not gene_mask[i]]

        self.genes = [self.genes[i] for i in range(len(self.genes)) if gene_mask[i]]
        self.gene_to_ind = self._construct_map_to_ind(self.genes)

        remove_mask = np.logical_not(gene_mask)

        if self.gene_N is not None:
            self.gene_N_missing = np.concatenate((self.gene_N_missing if self.gene_N_missing is not None else np.array([]), self.gene_N[remove_mask]))
        if self.gene_ignored_N is not None:
            self.gene_ignored_N_missing = np.concatenate((self.gene_ignored_N_missing if self.gene_ignored_N_missing is not None else np.array([]), self.gene_ignored_N[remove_mask]))

        if self.X_orig is not None:
            #store the genes that were removed for later
            X_orig_missing_genes = self.X_orig[remove_mask,:]
            if self.X_orig_missing_genes is not None:
                self.X_orig_missing_genes = sparse.csc_matrix(sparse.vstack([self.X_orig_missing_genes, X_orig_missing_genes]))
            else:
                self.X_orig_missing_genes = X_orig_missing_genes

            #reset the X matrix and scale factors
            self._set_X(self.X_orig[gene_mask,:], self.genes, self.gene_sets, skip_V=skip_V, skip_scale_factors=skip_scale_factors, skip_N=True)


        if self.X_orig_missing_gene_sets is not None:

            X_orig_missing_genes_missing_gene_sets = self.X_orig_missing_gene_sets[remove_mask,:]
            if self.X_orig_missing_genes_missing_gene_sets is not None:
                self.X_orig_missing_genes_missing_gene_sets = sparse.csc_matrix(sparse.vstack([self.X_orig_missing_genes_missing_gene_sets, X_orig_missing_genes_missing_gene_sets]))
            else:
                self.X_orig_missing_genes_missing_gene_sets = X_orig_missing_genes_missing_gene_sets

            #if we've already removed gene sets, then we need remove the genes from them too
            self.X_orig_missing_gene_sets = self.X_orig_missing_gene_sets[gene_mask,:]
            #need to recompute these
            (self.mean_shifts_missing, self.scale_factors_missing) = self._calc_X_shift_scale(self.X_orig_missing_gene_sets, self.y_corr_cholesky)

        if self.gene_N is not None:
            self.gene_N = self.gene_N[gene_mask]
        if self.gene_ignored_N is not None:
            self.gene_ignored_N = self.gene_ignored_N[gene_mask]

        if not skip_Y:
            if self.Y is not None:
                self._set_Y(self.Y[gene_mask], self.Y_for_regression[gene_mask] if self.Y_for_regression is not None else None, self.Y_exomes[gene_mask] if self.Y_exomes is not None else None, self.Y_positive_controls[gene_mask] if self.Y_positive_controls is not None else None, Y_corr_m=self.y_corr[:,gene_mask] if self.y_corr is not None else None, store_cholesky=self.y_corr_cholesky is not None, store_corr_sparse=self.y_corr_sparse is not None, skip_V=skip_V)

            if self.huge_signal_bfs is not None:
                self.huge_signal_bfs = self.huge_signal_bfs[gene_mask,:]
            if self.huge_gene_covariates is not None:
                self.huge_gene_covariates = self.huge_gene_covariates[gene_mask,:]
            if self.huge_gene_covariate_zs is not None:
                self.huge_gene_covariate_zs = self.huge_gene_covariate_zs[gene_mask,:]
            if self.huge_gene_covariates_mask is not None:
                self.huge_gene_covariates_mask = self.huge_gene_covariates_mask[gene_mask]

            if self.priors is not None:
                self.priors = self.priors[gene_mask]
            if self.priors_adj is not None:
                self.priors_adj = self.priors_adj[gene_mask]
            if self.combined_prior_Ys is not None:
                self.combined_prior_Ys = self.combined_prior_Ys[gene_mask]
            if self.combined_prior_Ys_adj is not None:
                self.combined_prior_Ys_adj = self.combined_prior_Ys_adj[gene_mask]
            if self.combined_Ds is not None:
                self.combined_Ds = self.combined_Ds[gene_mask]
            if self.Y_orig is not None:
                self.Y_orig = self.Y_orig[gene_mask]
            if self.Y_for_regression_orig is not None:
                self.Y_for_regression_orig = self.Y_for_regression_orig[gene_mask]
            if self.Y_w_orig is not None:
                self.Y_w_orig = self.Y_w_orig[gene_mask]
            if self.Y_fw_orig is not None:
                self.Y_fw_orig = self.Y_fw_orig[gene_mask]
            if self.priors_orig is not None:
                self.priors_orig = self.priors_orig[gene_mask]
            if self.priors_adj_orig is not None:
                self.priors_adj_orig = self.priors_adj_orig[gene_mask]

        #for x in [self.priors, self.combined_prior_Ys, self.Y_orig, self.Y_w_orig, self.Y_fw_orig, self.priors_orig, self.combined_prior_Ys_orig]:
        #    if x is not None:
        #        x[:] = np.concatenate((x[gene_mask], x[~gene_mask]))


    #subset the current state of the class to a reduced set of gene sets
    def _subset_gene_sets(self, subset_mask, keep_missing=True, ignore_missing=False, skip_V=False, skip_scale_factors=False):

        if subset_mask is None or np.sum(~subset_mask) == 0:
            return
        if self.gene_sets is None:
            return

        log("Subsetting gene sets", TRACE)

        remove_mask = np.logical_not(subset_mask)

        if ignore_missing:
            keep_missing = False

            if self.gene_sets is not None:
                if self.gene_sets_ignored is None:
                    self.gene_sets_ignored = []
                self.gene_sets_ignored = self.gene_sets_ignored + [self.gene_sets[i] for i in range(len(self.gene_sets)) if remove_mask[i]]

            if self.gene_set_labels is not None:
                if self.gene_set_labels_ignored is None:
                    self.gene_set_labels_ignored = []
                self.gene_set_labels_ignored = np.append(self.gene_set_labels_ignored, self.gene_set_labels[remove_mask])

            if self.scale_factors is not None:
                if self.scale_factors_ignored is None:
                    self.scale_factors_ignored = np.array([])
                self.scale_factors_ignored = np.append(self.scale_factors_ignored, self.scale_factors[remove_mask])

            if self.mean_shifts is not None:
                if self.mean_shifts_ignored is None:
                    self.mean_shifts_ignored = np.array([])
                self.mean_shifts_ignored = np.append(self.mean_shifts_ignored, self.mean_shifts[remove_mask])

            if self.beta_tildes is not None:
                if self.beta_tildes_ignored is None:
                    self.beta_tildes_ignored = np.array([])
                self.beta_tildes_ignored = np.append(self.beta_tildes_ignored, self.beta_tildes[remove_mask])

            if self.p_values is not None:
                if self.p_values_ignored is None:
                    self.p_values_ignored = np.array([])
                self.p_values_ignored = np.append(self.p_values_ignored, self.p_values[remove_mask])

            if self.ses is not None:
                if self.ses_ignored is None:
                    self.ses_ignored = np.array([])
                self.ses_ignored = np.append(self.ses_ignored, self.ses[remove_mask])

            if self.z_scores is not None:
                if self.z_scores_ignored is None:
                    self.z_scores_ignored = np.array([])
                self.z_scores_ignored = np.append(self.z_scores_ignored, self.z_scores[remove_mask])

            if self.se_inflation_factors is not None:
                if self.se_inflation_factors_ignored is None:
                    self.se_inflation_factors_ignored = np.array([])
                self.se_inflation_factors_ignored = np.append(self.se_inflation_factors_ignored, self.se_inflation_factors[remove_mask])

            if self.huge_gene_covariates is not None:
                if self.total_qc_metrics_ignored is None:
                    self.total_qc_metrics_ignored = self.total_qc_metrics[remove_mask,:]
                    self.mean_qc_metrics_ignored = self.mean_qc_metrics[remove_mask]
                else:
                    self.total_qc_metrics_ignored = np.vstack((self.total_qc_metrics_ignored, self.total_qc_metrics[remove_mask,:]))
                    self.mean_qc_metrics_ignored = np.append(self.mean_qc_metrics_ignored, self.mean_qc_metrics[remove_mask])

            #need to record how many ignored
            if self.X_orig is not None:
                if self.col_sums_ignored is None:
                    self.col_sums_ignored = np.array([])
                self.col_sums_ignored = np.append(self.col_sums_ignored, self.get_col_sums(self.X_orig[:,remove_mask]))

                gene_ignored_N = self.get_col_sums(self.X_orig[:,remove_mask], axis=1)
                if self.gene_ignored_N is None:
                    self.gene_ignored_N = gene_ignored_N
                else:
                    self.gene_ignored_N += gene_ignored_N
                if self.gene_N is not None:
                    self.gene_N -= gene_ignored_N

        elif keep_missing:
            self.gene_sets_missing = [self.gene_sets[i] for i in range(len(self.gene_sets)) if remove_mask[i]]

            if self.beta_tildes is not None:
                self.beta_tildes_missing = self.beta_tildes[remove_mask]
            if self.p_values is not None:
                self.p_values_missing = self.p_values[remove_mask]
            if self.z_scores is not None:
                self.z_scores_missing = self.z_scores[remove_mask]
            if self.ses is not None:
                self.ses_missing = self.ses[remove_mask]
            if self.se_inflation_factors is not None:
                self.se_inflation_factors_missing = self.se_inflation_factors[remove_mask]
            if self.beta_tildes_orig is not None:
                self.beta_tildes_missing_orig = self.beta_tildes_orig[remove_mask]
            if self.p_values_orig is not None:
                self.p_values_missing_orig = self.p_values_orig[remove_mask]
            if self.z_scores_orig is not None:
                self.z_scores_missing_orig = self.z_scores_orig[remove_mask]
            if self.ses_orig is not None:
                self.ses_missing_orig = self.ses_orig[remove_mask]

            if self.total_qc_metrics is not None:
                self.total_qc_metrics_missing = self.total_qc_metrics[remove_mask]

            if self.mean_qc_metrics is not None:
                self.mean_qc_metrics_missing = self.mean_qc_metrics[remove_mask]

            if self.inf_betas is not None:
                self.inf_betas_missing = self.inf_betas[remove_mask]

            if self.betas_uncorrected is not None:
                self.betas_uncorrected_missing = self.betas_uncorrected[remove_mask]

            if self.betas is not None:
                self.betas_missing = self.betas[remove_mask]
            if self.non_inf_avg_cond_betas is not None:
                self.non_inf_avg_cond_betas_missing = self.non_inf_avg_cond_betas[remove_mask]
            if self.non_inf_avg_postps is not None:
                self.non_inf_avg_postps_missing = self.non_inf_avg_postps[remove_mask]

            if self.inf_betas_orig is not None:
                self.inf_betas_missing_orig = self.inf_betas_orig[remove_mask]
            if self.betas_orig is not None:
                self.betas_missing_orig = self.betas_orig[remove_mask]
            if self.betas_uncorrected_orig is not None:
                self.betas_uncorrected_missing_orig = self.betas_uncorrected_orig[remove_mask]
            if self.non_inf_avg_cond_betas_orig is not None:
                self.non_inf_avg_cond_betas_missing_orig = self.non_inf_avg_cond_betas_orig[remove_mask]
            if self.non_inf_avg_postps_orig is not None:
                self.non_inf_avg_postps_missing_orig = self.non_inf_avg_postps_orig[remove_mask]

            if self.is_dense_gene_set is not None:
                self.is_dense_gene_set_missing = self.is_dense_gene_set[remove_mask]

            if self.gene_set_batches is not None:
                self.gene_set_batches_missing = self.gene_set_batches[remove_mask]

            if self.gene_set_labels is not None:
                self.gene_set_labels_missing = self.gene_set_labels[remove_mask]

            if self.ps is not None:
                self.ps_missing = self.ps[remove_mask]
            if self.sigma2s is not None:
                self.sigma2s_missing = self.sigma2s[remove_mask]


            if self.X_orig is not None:
                #store the removed gene sets for later
                if keep_missing:
                    self.X_orig_missing_gene_sets = self.X_orig[:,remove_mask]
                    self.mean_shifts_missing = self.mean_shifts[remove_mask]
                    self.scale_factors_missing = self.scale_factors[remove_mask]

        #now do the subsetting to keep

        if self.beta_tildes is not None:
            self.beta_tildes = self.beta_tildes[subset_mask]
        if self.p_values is not None:
            self.p_values = self.p_values[subset_mask]
        if self.z_scores is not None:
            self.z_scores = self.z_scores[subset_mask]
        if self.ses is not None:
            self.ses = self.ses[subset_mask]
        if self.se_inflation_factors is not None:
            self.se_inflation_factors = self.se_inflation_factors[subset_mask]

        if self.beta_tildes_orig is not None:
            self.beta_tildes_orig = self.beta_tildes_orig[subset_mask]
        if self.p_values_orig is not None:
            self.p_values_orig = self.p_values_orig[subset_mask]
        if self.z_scores_orig is not None:
            self.z_scores_orig = self.z_scores_orig[subset_mask]
        if self.ses_orig is not None:
            self.ses_orig = self.ses_orig[subset_mask]


        if self.total_qc_metrics is not None:
            self.total_qc_metrics = self.total_qc_metrics[subset_mask]

        if self.mean_qc_metrics is not None:
            self.mean_qc_metrics = self.mean_qc_metrics[subset_mask]

        if self.inf_betas is not None:
            self.inf_betas = self.inf_betas[subset_mask]

        if self.betas_uncorrected is not None:
            self.betas_uncorrected = self.betas_uncorrected[subset_mask]

        if self.betas is not None:
            self.betas = self.betas[subset_mask]
        if self.non_inf_avg_cond_betas is not None:
            self.non_inf_avg_cond_betas = self.non_inf_avg_cond_betas[subset_mask]
        if self.non_inf_avg_postps is not None:
            self.non_inf_avg_postps = self.non_inf_avg_postps[subset_mask]

        if self.inf_betas_orig is not None:
            self.inf_betas_orig = self.inf_betas_orig[subset_mask]
        if self.betas_orig is not None:
            self.betas_orig = self.betas_orig[subset_mask]
        if self.betas_uncorrected_orig is not None:
            self.betas_uncorrected_orig = self.betas_uncorrected_orig[subset_mask]
        if self.non_inf_avg_cond_betas_orig is not None:
            self.non_inf_avg_cond_betas_orig = self.non_inf_avg_cond_betas_orig[subset_mask]
        if self.non_inf_avg_postps_orig is not None:
            self.non_inf_avg_postps_orig = self.non_inf_avg_postps_orig[subset_mask]

        if self.is_dense_gene_set is not None:
            self.is_dense_gene_set = self.is_dense_gene_set[subset_mask]

        if self.gene_set_batches is not None:
            self.gene_set_batches = self.gene_set_batches[subset_mask]

        if self.gene_set_labels is not None:
            self.gene_set_labels = self.gene_set_labels[subset_mask]

        if self.ps is not None:
            self.ps = self.ps[subset_mask]
        if self.sigma2s is not None:
            self.sigma2s = self.sigma2s[subset_mask]

        self.gene_sets = list(itertools.compress(self.gene_sets, subset_mask))
        self.gene_set_to_ind = self._construct_map_to_ind(self.gene_sets)

        if self.X_orig is not None:
            #never update V; if it exists it will be updated below
            self._set_X(self.X_orig[:,subset_mask], self.genes, self.gene_sets, skip_V=True, skip_scale_factors=skip_scale_factors, skip_N=True)

        if self.X_orig_missing_genes is not None:
            #if we've already removed genes, then we need to remove the gene sets from them
            if keep_missing:
                self.X_orig_missing_genes_missing_gene_sets = self.X_orig_missing_genes[:,remove_mask]
            self.X_orig_missing_genes = self.X_orig_missing_genes[:,subset_mask]

        #need to update the scale factor for sigma2
        #sigma2 is always relative to just the non missing gene sets
        if self.sigma2 is not None:
            self.set_sigma(self.sigma2, self.sigma_power, sigma2_osc=self.sigma2_osc)
        if self.p is not None:
            self.set_p(self.p)

    def _unsubset_gene_sets(self, skip_V=False, skip_scale_factors=False):
        if self.gene_sets_missing is None or self.X_orig_missing_gene_sets is None:
            return(np.array([True] * len(self.gene_sets)))

        log("Un-subsetting gene sets", TRACE)

        #need to update the scale factor for sigma2
        #sigma2 is always relative to just the non missing gene sets
        fraction_non_missing = self._get_fraction_non_missing()

        subset_mask = np.array([True] * len(self.gene_sets) + [False] * len(self.gene_sets_missing))

        self.gene_sets += self.gene_sets_missing
        self.gene_sets_missing = None
        self.gene_set_to_ind = self._construct_map_to_ind(self.gene_sets)

        #if self.sigma2 is not None:
        #    old_sigma2 = self.sigma2
        #    self.set_sigma(self.sigma2 * fraction_non_missing, self.sigma_power, sigma2_osc=self.sigma2_osc)
        #    log("Changing sigma from %.4g to %.4g" % (old_sigma2, self.sigma2))
        #if self.p is not None:
        #    old_p = self.p
        #    self.set_p(self.p * fraction_non_missing)
        #    log("Changing p from %.4g to %.4g" % (old_p, self.p))

        if self.beta_tildes_missing is not None:
            self.beta_tildes = np.append(self.beta_tildes, self.beta_tildes_missing)
            self.beta_tildes_missing = None
        if self.p_values_missing is not None:
            self.p_values = np.append(self.p_values, self.p_values_missing)
            self.p_values_missing = None
        if self.z_scores_missing is not None:
            self.z_scores = np.append(self.z_scores, self.z_scores_missing)
            self.z_scores_missing = None
        if self.ses_missing is not None:
            self.ses = np.append(self.ses, self.ses_missing)
            self.ses_missing = None
        if self.se_inflation_factors_missing is not None:
            self.se_inflation_factors = np.append(self.se_inflation_factors, self.se_inflation_factors_missing)
            self.se_inflation_factors_missing = None

        if self.total_qc_metrics_missing is not None:
            self.total_qc_metrics = np.vstack((self.total_qc_metrics, self.total_qc_metrics_missing))
            self.total_qc_metrics_missing = None

        if self.mean_qc_metrics_missing is not None:
            self.mean_qc_metrics = np.append(self.mean_qc_metrics, self.mean_qc_metrics_missing)
            self.mean_qc_metrics_missing = None

        if self.beta_tildes_missing_orig is not None:
            self.beta_tildes_orig = np.append(self.beta_tildes_orig, self.beta_tildes_missing_orig)
            self.beta_tildes_missing_orig = None
        if self.p_values_missing_orig is not None:
            self.p_values_orig = np.append(self.p_values_orig, self.p_values_missing_orig)
            self.p_values_missing_orig = None
        if self.z_scores_missing_orig is not None:
            self.z_scores_orig = np.append(self.z_scores_orig, self.z_scores_missing_orig)
            self.z_scores_missing_orig = None
        if self.ses_missing_orig is not None:
            self.ses_orig = np.append(self.ses_orig, self.ses_missing_orig)
            self.ses_missing_orig = None

        if self.inf_betas_missing is not None:
            self.inf_betas = np.append(self.inf_betas, self.inf_betas_missing)
            self.inf_betas_missing = None

        if self.betas_uncorrected_missing is not None:
            self.betas_uncorrected = np.append(self.betas_uncorrected, self.betas_uncorrected_missing)
            self.betas_uncorrected_missing = None

        if self.betas_missing is not None:
            self.betas = np.append(self.betas, self.betas_missing)
            self.betas_missing = None
        if self.non_inf_avg_cond_betas_missing is not None:
            self.non_inf_avg_cond_betas = np.append(self.non_inf_avg_cond_betas, self.non_inf_avg_cond_betas_missing)
            self.non_inf_avg_cond_betas_missing = None
        if self.non_inf_avg_postps_missing is not None:
            self.non_inf_avg_postps = np.append(self.non_inf_avg_postps, self.non_inf_avg_postps_missing)
            self.non_inf_avg_postps_missing = None

        if self.inf_betas_missing_orig is not None:
            self.inf_betas_orig = np.append(self.inf_betas_orig, self.inf_betas_missing_orig)
            self.inf_betas_missing_orig = None
        if self.betas_missing_orig is not None:
            self.betas_orig = np.append(self.betas_orig, self.betas_missing_orig)
            self.betas_missing_orig = None
        if self.betas_uncorrected_missing_orig is not None:
            self.betas_uncorrected_orig = np.append(self.betas_uncorrected_orig, self.betas_uncorrected_missing_orig)
            self.betas_uncorrected_missing_orig = None
        if self.non_inf_avg_cond_betas_missing_orig is not None:
            self.non_inf_avg_cond_betas_orig = np.append(self.non_inf_avg_cond_betas_orig, self.non_inf_avg_cond_betas_missing_orig)
            self.non_inf_avg_cond_betas_missing_orig = None
        if self.non_inf_avg_postps_missing_orig is not None:
            self.non_inf_avg_postps_orig = np.append(self.non_inf_avg_postps_orig, self.non_inf_avg_postps_missing_orig)
            self.non_inf_avg_postps_missing_orig = None

        if self.X_orig_missing_gene_sets is not None:
            self.X_orig = sparse.hstack((self.X_orig, self.X_orig_missing_gene_sets), format="csc")
            self.X_orig_missing_gene_sets = None
            self.mean_shifts = np.append(self.mean_shifts, self.mean_shifts_missing)
            self.mean_shifts_missing = None
            self.scale_factors = np.append(self.scale_factors, self.scale_factors_missing)
            self.scale_factors_missing = None
            self.is_dense_gene_set = np.append(self.is_dense_gene_set, self.is_dense_gene_set_missing)
            self.is_dense_gene_set_missing = None
            self.gene_set_batches = np.append(self.gene_set_batches, self.gene_set_batches_missing)
            self.gene_set_batches = None
            self.gene_set_labels = np.append(self.gene_set_labels, self.gene_set_labels_missing)
            self.gene_set_labels = None


            if self.ps is not None:
                self.ps = np.append(self.ps, self.ps_missing)
                self.ps_missing = None
            if self.sigma2s is not None:
                self.sigma2s = np.append(self.sigma2s, self.sigma2s_missing)
                self.sigma2s_missing = None

        self._set_X(self.X_orig, self.genes, self.gene_sets, skip_V=skip_V, skip_scale_factors=skip_scale_factors, skip_N=False)

        if self.X_orig_missing_genes_missing_gene_sets is not None:
            #if we've already removed genes, then we need to remove the gene sets from them
            self.X_orig_missing_genes = sparse.hstack((self.X_orig_missing_genes, self.X_orig_missing_genes_missing_gene_sets), format="csc")
            self.X_orig_missing_genes_missing_gene_sets = None

        return(subset_mask)


    #utility function to create a mapping from name to index in a list
    def _construct_map_to_ind(self, gene_sets):
        return dict([(gene_sets[i], i) for i in range(len(gene_sets))])

    #utility function to map names or indices to column indicies
    def _get_col(self, col_name_or_index, header_cols, require_match=True):
        try:
            if col_name_or_index is None:
                raise ValueError
            return(int(col_name_or_index))
        except ValueError:
            matching_cols = [i for i in range(0,len(header_cols)) if header_cols[i] == col_name_or_index]
            if len(matching_cols) == 0:
                if require_match:
                    bail("Could not find match for column %s in header: %s" % (col_name_or_index, "\t".join(header_cols)))
                else:
                    return None
            if len(matching_cols) > 1:
                bail("Found two matches for column %s in header: %s" % (col_name_or_index, "\t".join(header_cols)))
            return matching_cols[0]

    # inverse_matrix calculations
    def _invert_matrix(self,matrix_in):
        inv_matrix=np.linalg.inv(matrix_in) 
        return inv_matrix

    def _invert_sym_matrix(self,matrix_in):
        cho_factor = scipy.linalg.cho_factor(matrix_in)
        return scipy.linalg.cho_solve(cho_factor, np.eye(matrix_in.shape[0]))

    def _invert_matrix_old(self,matrix_in):
        sparsity=(matrix_in.shape[0]*matrix_in.shape[1]-matrix_in.count_nonzero())/(matrix_in.shape[0]*matrix_in.shape[1])
        log("Sparsity of matrix_in in invert_matrix %s" % sparsity, INFO)
        if sparsity>0.65:
            inv_matrix=np.linalg.inv(matrix_in.toarray()) # works efficiently for sparse matrix_in
        else:
            inv_matrix=sparse.linalg.inv(matrix_in) 
        return inv_matrix


##This function is for labelling clusters. Update it with your favorite LLM if desired
def query_lmm(query, auth_key=None):

    import requests

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer %s' % auth_key,
    }

    json_data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {
                'role': 'user',
                'content': '%s' % query,
            },
        ],
    }
    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data).json()
        if "choices" in response and len(response["choices"]) > 0 and "message" in response["choices"][0] and "content" in response["choices"][0]["message"]:
            return response["choices"][0]["message"]["content"]
        else:
            log("LMM response did not match the expected format; returning none. Response: %s" % response); 
            return None
    except Exception:
        log("LMM call failed; returning None"); 
        return None


def main():

    if not options.hide_opts:
        log("Python version: %s" % sys.version)
        log("Numpy version: %s" % np.__version__)
        log("Scipy version: %s" % scipy.__version__)
        log("Options: %s" % options, DEBUG)

    g = GeneSetData(batch_size=options.batch_size)

    #g.read_X(options.X_in)
    #y = []
    #for line in open("c"):
    #    a = line.strip().split()
    #    y.append(a)
    #y = np.array(y)
    #print(g.X_orig.shape, y.shape)
    #print(g._compute_logistic_beta_tildes(g.X_orig, y, 1, 0, append_pseudo=False, convert_to_dichotomous=False))
    #bail("")

    if options.top_gene_set_prior:
        g.set_sigma(g.convert_prior_to_var(options.top_gene_set_prior, options.num_gene_sets_for_prior if options.num_gene_sets_for_prior is not None else len(g.gene_sets)), options.sigma_power, convert_sigma_to_internal_units=True)
        log("Setting sigma=%.4g (external=%.4g) given top of %d gene sets prior of %.4g" % (g.get_sigma2(), g.get_sigma2(convert_sigma_to_external_units=True), options.num_gene_sets_for_prior, options.top_gene_set_prior))
        #we overrode this
        options.sigma2_cond = None
    elif options.sigma2_ext:
        g.set_sigma(options.sigma2_ext, options.sigma_power, convert_sigma_to_internal_units=True)
        #we overrode this
        options.sigma2_cond = None
    elif options.sigma2:
        g.set_sigma(options.sigma2, options.sigma_power, convert_sigma_to_internal_units=False)
        #we overrode this
        options.sigma2_cond = None

    #sigma calculations
    if options.const_sigma:
        options.sigma_power = 2

    if options.update_hyper.lower() == "both":
        options.update_hyper_p = True
        options.update_hyper_sigma = True
    elif options.update_hyper.lower() == "p":
        options.update_hyper_p = True
        options.update_hyper_sigma = False
    elif options.update_hyper.lower() == "sigma2" or options.update_hyper.lower() == "sigma":
        options.update_hyper_p = False
        options.update_hyper_sigma = True
    elif options.update_hyper.lower() == "none":
        options.update_hyper_p = False
        options.update_hyper_sigma = False
    else:
        bail("Invalid value for --update-hyper (both, p, sigma2, or none)")

    if options.gene_map_in:
        g.read_gene_map(options.gene_map_in, options.gene_map_orig_gene_col, options.gene_map_orig_gene_col)
    if options.gene_loc_file:
        g.init_gene_locs(options.gene_loc_file)

    if run_huge or run_beta_tilde or run_sigma or run_beta or run_priors or run_gibbs or run_factor:
        if options.gene_bfs_in:
            g.read_Y(gene_bfs_in=options.gene_bfs_in,show_progress=not options.hide_progress, gene_bfs_id_col=options.gene_bfs_id_col, gene_bfs_log_bf_col=options.gene_bfs_log_bf_col, gene_bfs_combined_col=options.gene_bfs_combined_col, gene_bfs_prior_col=options.gene_bfs_prior_col, hold_out_chrom=options.hold_out_chrom)
        elif options.gwas_in or options.exomes_in:
            g.read_Y(gwas_in=options.gwas_in,show_progress=not options.hide_progress, gwas_chrom_col=options.gwas_chrom_col, gwas_pos_col=options.gwas_pos_col, gwas_p_col=options.gwas_p_col, gwas_beta_col=options.gwas_beta_col, gwas_se_col=options.gwas_se_col, gwas_n_col=options.gwas_n_col, gwas_n=options.gwas_n, gwas_units=options.gwas_units, gwas_freq_col=options.gwas_freq_col, gwas_locus_col=options.gwas_locus_col, gwas_ignore_p_threshold=options.gwas_ignore_p_threshold, gwas_low_p=options.gwas_low_p, gwas_high_p=options.gwas_high_p, gwas_low_p_posterior=options.gwas_low_p_posterior, gwas_high_p_posterior=options.gwas_high_p_posterior, detect_low_power=options.gwas_detect_low_power, detect_high_power=options.gwas_detect_high_power, detect_adjust_huge=options.gwas_detect_adjust_huge, learn_window=options.learn_window, closest_gene_prob=options.closest_gene_prob, max_closest_gene_prob=options.max_closest_gene_prob, scale_raw_closest_gene=options.scale_raw_closest_gene, cap_raw_closest_gene=options.cap_raw_closest_gene, cap_region_posterior=options.cap_region_posterior, scale_region_posterior=options.scale_region_posterior, phantom_region_posterior=options.phantom_region_posterior, allow_evidence_of_absence=options.allow_evidence_of_absence, correct_huge=options.correct_huge, gws_prob_true=options.gws_prob_true, max_closest_gene_dist=options.max_closest_gene_dist, signal_window_size=options.signal_window_size, signal_min_sep=options.signal_min_sep, signal_max_logp_ratio=options.signal_max_logp_ratio, credible_set_span=options.credible_set_span, min_n_ratio=options.min_n_ratio, max_clump_ld=options.max_clump_ld, exomes_in=options.exomes_in, exomes_gene_col=options.exomes_gene_col, exomes_p_col=options.exomes_p_col, exomes_beta_col=options.exomes_beta_col, exomes_se_col=options.exomes_se_col, exomes_n_col=options.exomes_n_col, exomes_n=options.exomes_n, exomes_units=options.exomes_units, exomes_low_p=options.exomes_low_p, exomes_high_p=options.exomes_high_p, exomes_low_p_posterior=options.exomes_low_p_posterior, exomes_high_p_posterior=options.exomes_high_p_posterior, positive_controls_in=options.positive_controls_in, positive_controls_id_col=options.positive_controls_id_col, positive_controls_prob_col=options.positive_controls_prob_col, gene_loc_file=options.gene_loc_file_huge if options.gene_loc_file_huge is not None else options.gene_loc_file, hold_out_chrom=options.hold_out_chrom, exons_loc_file=options.exons_loc_file_huge, min_var_posterior=options.min_var_posterior, s2g_in=options.s2g_in, s2g_chrom_col=options.s2g_chrom_col, s2g_pos_col=options.s2g_pos_col, s2g_gene_col=options.s2g_gene_col, s2g_prob_col=options.s2g_prob_col, credible_sets_in=options.credible_sets_in, credible_sets_id_col=options.credible_sets_id_col, credible_sets_chrom_col=options.credible_sets_chrom_col, credible_sets_pos_col=options.credible_sets_pos_col, credible_sets_ppa_col=options.credible_sets_ppa_col)
        elif options.gene_percentiles_in:
            g.read_Y(gene_percentiles_in=options.gene_percentiles_in,show_progress=not options.hide_progress, gene_percentiles_id_col=options.gene_percentiles_id_col, gene_percentiles_value_col=options.gene_percentiles_value_col, gene_percentiles_higher_is_better=options.gene_percentiles_higher_is_better, top_posterior=options.top_posterior, hold_out_chrom=options.hold_out_chrom)
        elif options.gene_zs_in:
            g.read_Y(gene_zs_in=options.gene_zs_in,show_progress=not options.hide_progress, gene_zs_id_col=options.gene_zs_id_col, gene_zs_value_col=options.gene_zs_value_col, gws_threshold=options.gws_threshold, gws_prob_true=options.gws_prob_true, max_mean_posterior=options.max_mean_posterior, hold_out_chrom=options.hold_out_chrom)
        #else:
        #    bail("Need --gwas-in or --exomes-in or --gene-bfs-in or --gene-percentiles-in or --gene-zs-in")

    if not run_huge:

        ignore_filters = False
        gene_set_ids = None
        if run_factor and options.gene_set_stats_in is not None:
            #get the IDs we'll keep
            gene_set_ids = g.read_gene_set_statistics(options.gene_set_stats_in, stats_id_col=options.gene_set_stats_id_col, stats_exp_beta_tilde_col=options.gene_set_stats_exp_beta_tilde_col, stats_beta_tilde_col=options.gene_set_stats_beta_tilde_col, stats_p_col=options.gene_set_stats_p_col, stats_se_col=options.gene_set_stats_se_col, stats_beta_col=options.gene_set_stats_beta_col, stats_beta_uncorrected_col=options.gene_set_stats_beta_uncorrected_col, ignore_negative_exp_beta=options.ignore_negative_exp_beta, max_gene_set_p=options.max_gene_set_p, min_gene_set_beta=options.min_gene_set_beta, min_gene_set_beta_uncorrected=options.min_gene_set_beta_uncorrected, return_only_ids=True)
            ignore_filters = True

        #read in the matrices
        if options.X_in is not None or options.X_list is not None or options.Xd_in is not None or options.Xd_list is not None:
            filter_gene_set_p = options.filter_gene_set_p
            force_reread = False
            while True:
                g.read_X(options.X_in, Xd_in=options.Xd_in, X_list=options.X_list, Xd_list=options.Xd_list, V_in=options.V_in, min_gene_set_size=options.min_gene_set_size, max_gene_set_size=options.max_gene_set_size, only_ids=gene_set_ids, prune_gene_sets=options.prune_gene_sets, x_sparsify=options.x_sparsify, add_ext=options.add_ext, add_top=options.add_top, add_bottom=options.add_bottom, filter_negative=options.filter_negative, threshold_weights=options.threshold_weights, max_gene_set_p=options.max_gene_set_p, filter_gene_set_p=filter_gene_set_p if not ignore_filters else 1, increase_filter_gene_set_p=options.increase_filter_gene_set_p, max_num_gene_sets=options.max_num_gene_sets, filter_gene_set_metric_z=options.filter_gene_set_metric_z if not ignore_filters else None, initial_p=options.p_noninf, initial_sigma2=g.sigma2, initial_sigma2_cond=options.sigma2_cond, sigma_power=options.sigma_power, sigma_soft_threshold_95=options.sigma_soft_threshold_95, sigma_soft_threshold_5=options.sigma_soft_threshold_5, run_logistic=True, run_gls=options.gls, run_corrected_ols=not options.ols and not options.gls, gene_loc_file=options.gene_loc_file, gene_cor_file=options.gene_cor_file, gene_cor_file_gene_col=options.gene_cor_file_gene_col, gene_cor_file_cor_start_col=options.gene_cor_file_cor_start_col, update_hyper_p=options.update_hyper_p, update_hyper_sigma=options.update_hyper_sigma, batch_all_for_hyper=options.batch_all_for_hyper, first_for_hyper=options.first_for_hyper, first_for_sigma_cond=options.first_for_sigma_cond, sigma_num_devs_to_top=options.sigma_num_devs_to_top, p_noninf_inflate=options.p_noninf_inflate, batch_separator=options.batch_separator, file_separator=options.file_separator, max_num_burn_in=options.max_num_burn_in, max_num_iter_betas=options.max_num_iter_betas, min_num_iter_betas=options.min_num_iter_betas, num_chains_betas=options.num_chains_betas, r_threshold_burn_in_betas=options.r_threshold_burn_in_betas, use_max_r_for_convergence_betas=options.use_max_r_for_convergence_betas, max_frac_sem_betas=options.max_frac_sem_betas, max_allowed_batch_correlation=options.max_allowed_batch_correlation, sparse_solution=options.sparse_solution, sparse_frac_betas=options.sparse_frac_betas, betas_trace_out=options.betas_trace_out, show_progress=not options.hide_progress, skip_V=(options.max_gene_set_p is not None), force_reread=force_reread)
                if gene_set_ids is not None:
                    break
                if options.min_num_gene_sets is None or filter_gene_set_p is None or filter_gene_set_p >= 1 or g.gene_sets is None or len(g.gene_sets) >= options.min_num_gene_sets:
                    break
                if filter_gene_set_p < 1:
                    fraction_to_increase = float(options.min_num_gene_sets) / len(g.gene_sets)
                    assert(fraction_to_increase > 1)
                    #add in a fudge factor
                    filter_gene_set_p *= fraction_to_increase * 1.2
                    log("Only read in %d gene sets; scaled --filter-gene-set-p to %.3g and re-reading gene sets" % (len(g.gene_sets), filter_gene_set_p))
                    force_reread = True
                else:
                    break
                    
        run_gibbs_for_factor = False
        if options.gene_set_stats_in is not None:
            g.read_gene_set_statistics(options.gene_set_stats_in, stats_id_col=options.gene_set_stats_id_col, stats_exp_beta_tilde_col=options.gene_set_stats_exp_beta_tilde_col, stats_beta_tilde_col=options.gene_set_stats_beta_tilde_col, stats_p_col=options.gene_set_stats_p_col, stats_se_col=options.gene_set_stats_se_col, stats_beta_col=options.gene_set_stats_beta_col, stats_beta_uncorrected_col=options.gene_set_stats_beta_uncorrected_col, ignore_negative_exp_beta=options.ignore_negative_exp_beta, max_gene_set_p=options.max_gene_set_p, min_gene_set_beta=options.min_gene_set_beta, min_gene_set_beta_uncorrected=options.min_gene_set_beta_uncorrected)
        elif run_beta_tilde or run_sigma or run_beta or run_priors or run_gibbs or run_factor:
            if run_factor:
                run_gibbs_for_factor = True
            g.calculate_gene_set_statistics(max_gene_set_p=options.filter_gene_set_p, run_gls=options.gls, run_logistic=True, run_corrected_ols=not options.ols and not options.gls, correct_betas=options.correct_betas, gene_loc_file=options.gene_loc_file, gene_cor_file=options.gene_cor_file, gene_cor_file_gene_col=options.gene_cor_file_gene_col, gene_cor_file_cor_start_col=options.gene_cor_file_cor_start_col)

        if options.X_out:
            g.write_X(options.X_out)
        if options.Xd_out:
            g.write_Xd(options.Xd_out)
        if options.V_out:
            g.write_V(options.V_out)

        if run_sigma or ((run_beta or run_priors or run_gibbs or run_gibbs_for_factor) and g.sigma2 is None):
            g.calculate_sigma(options.sigma_power, options.chisq_threshold, options.chisq_dynamic, options.desired_intercept_difference)

        #gene set betas
        if options.gene_set_betas_in:
            g.read_betas(options.gene_set_betas_in)
        elif run_beta or run_priors or run_gibbs or run_gibbs_for_factor:
            #if False:
            #    g.calculate_inf_betas(update_hyper_sigma=options.update_hyper_sigma)
            #update hyper was done above while while reading x
            g.calculate_non_inf_betas(options.p_noninf if g.p is None else g.p, max_num_burn_in=options.max_num_burn_in, max_num_iter=options.max_num_iter_betas, min_num_iter=options.min_num_iter_betas, num_chains=options.num_chains_betas, r_threshold_burn_in=options.r_threshold_burn_in_betas, use_max_r_for_convergence=options.use_max_r_for_convergence_betas, max_frac_sem=options.max_frac_sem_betas, max_allowed_batch_correlation=options.max_allowed_batch_correlation, gauss_seidel=options.gauss_seidel_betas, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=options.sparse_solution, sparse_frac_betas=options.sparse_frac_betas, pre_filter_batch_size=options.pre_filter_batch_size, pre_filter_small_batch_size=options.pre_filter_small_batch_size, betas_trace_out=options.betas_trace_out)

        #priors
        if run_priors:
            g.calculate_priors(max_gene_set_p=options.filter_gene_set_p, num_gene_batches=options.priors_num_gene_batches, correct_betas=options.correct_betas, gene_loc_file=options.gene_loc_file, gene_cor_file=options.gene_cor_file, gene_cor_file_gene_col=options.gene_cor_file_gene_col, gene_cor_file_cor_start_col=options.gene_cor_file_cor_start_col, p_noninf=options.p_noninf if g.p is None else g.p, max_num_burn_in=options.max_num_burn_in, max_num_iter=options.max_num_iter_betas, min_num_iter=options.min_num_iter_betas, num_chains=options.num_chains_betas, r_threshold_burn_in=options.r_threshold_burn_in_betas, use_max_r_for_convergence=options.use_max_r_for_convergence_betas, max_frac_sem=options.max_frac_sem_betas, max_allowed_batch_correlation=options.max_allowed_batch_correlation, gauss_seidel=options.gauss_seidel_betas, sparse_solution=options.sparse_solution, sparse_frac_betas=options.sparse_frac_betas)

        if run_gibbs or run_gibbs_for_factor:
            g.do_gibbs(min_num_iter=options.min_num_iter, max_num_iter=options.max_num_iter, num_chains=options.num_chains, num_mad=options.num_mad, r_threshold_burn_in=options.r_threshold_burn_in, max_frac_sem=options.max_frac_sem, use_max_r_for_convergence=options.use_max_r_for_convergence, p_noninf=options.p_noninf if g.p is None else g.p, increase_hyper_if_betas_below=options.increase_hyper_if_betas_below, update_huge_scores=options.update_huge_scores, top_gene_prior=options.top_gene_prior, max_num_burn_in=options.max_num_burn_in, max_num_iter_betas=options.max_num_iter_betas, min_num_iter_betas=options.min_num_iter_betas, num_chains_betas=options.num_chains_betas, r_threshold_burn_in_betas=options.r_threshold_burn_in_betas, use_max_r_for_convergence_betas=options.use_max_r_for_convergence_betas, max_frac_sem_betas=options.max_frac_sem_betas, use_mean_betas=not options.use_sampled_betas_in_gibbs, sparse_frac_gibbs=options.sparse_frac_gibbs, sparse_solution=options.sparse_solution, sparse_frac_betas=options.sparse_frac_betas, pre_filter_batch_size=options.pre_filter_batch_size, pre_filter_small_batch_size=options.pre_filter_small_batch_size, max_allowed_batch_correlation=options.max_allowed_batch_correlation, gauss_seidel=options.gauss_seidel, gauss_seidel_betas=options.gauss_seidel_betas, num_gene_batches=options.priors_num_gene_batches, num_batches_parallel=options.gibbs_num_batches_parallel, max_mb_X_h=options.gibbs_max_mb_X_h, initial_linear_filter=options.initial_linear_filter, correct_betas=options.correct_betas, gene_set_stats_trace_out=options.gene_set_stats_trace_out, gene_stats_trace_out=options.gene_stats_trace_out, betas_trace_out=options.betas_trace_out)

    if options.gene_set_stats_out:
        g.write_gene_set_statistics(options.gene_set_stats_out)

    if options.gene_stats_out:
        g.write_gene_statistics(options.gene_stats_out)

    if options.gene_gene_set_stats_out:
        g.write_gene_gene_set_statistics(options.gene_gene_set_stats_out)

    if options.gene_set_overlap_stats_out:
        g.write_gene_set_overlap_statistics(options.gene_set_overlap_stats_out)

    if options.gene_covs_out:
        g.write_huge_gene_covariates(options.gene_covs_out)

    if options.gene_effectors_out:
        g.write_gene_effectors(options.gene_effectors_out)

    if run_factor:
        g.run_factor(max_num_factors=options.max_num_factors, alpha0=options.alpha0, beta0=options.beta0, gene_set_filter_type=options.gene_set_filter_type, gene_set_filter_value=options.gene_set_filter_value, gene_filter_type=options.gene_filter_type, gene_filter_value=options.gene_filter_value, gene_set_multiply_type=options.gene_set_multiply_type, gene_multiply_type=options.gene_multiply_type, run_transpose=not options.no_transpose, lmm_auth_key=options.lmm_auth_key)

    if options.factors_out is not None or options.marker_factors_out is not None or options.gene_set_factors_out is not None or options.gene_factors_out is not None:
        g.write_matrix_factors(options.factors_out, options.gene_set_factors_out, options.gene_factors_out, options.marker_factors_out)

    if options.gene_set_clusters_out is not None or options.gene_clusters_out is not None:
        g.write_clusters(options.gene_set_clusters_out, options.gene_clusters_out)

    if options.params_out:
        g.write_params(options.params_out)

if __name__ == '__main__':
    #cProfile.run('main()')
    main()

