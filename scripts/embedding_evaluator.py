#!/usr/bin/env python3

import sys
sys.path.extend(['.', '..'])

import argparse
import os

import gensim
import numpy as np
import pandas as pd
from scipy import stats
from Bio import pairwise2
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')

from dna2vec.multi_k_model import MultiKModel

# Helper functions

def get_mers(k, num_mers):
    '''
    This produces <num_mers> random sequences of length k.
    '''
    bases = ['A', 'T', 'C', 'G']
    temp = np.random.choice(bases, size=(num_mers, k))
    return [''.join(x) for x in temp]


def most_similar(in_model, vocab,
                 same_k=True, is_vector=False,
                 topn=10):
    
    # Note this only works for returning k-mers of the same length.
    if same_k:
        return in_model.data[len(vocab)].model.most_similar(vocab, topn=topn)
    else:
        # Make vector representation of what we're searching for if needed.
        if is_vector == False:
            vec = in_model.vector(vocab)
        else:
            vec = vocab
            
        # These are for sorting the output.
        dtype = [('kmer', 'S10'), ('cosine', float)]
        scores = []
        for model_wrap in in_model.data.values():
            temp = model_wrap.model.similar_by_vector(vec, topn=topn)
            scores.append(np.array(temp, dtype=dtype))
        return np.sort(np.concatenate(scores, axis=0), axis=0, order='cosine')[:-(topn + 1):-1]
 

# Experiments in preprint    

def cosine_alignment_experiment(num_mers=10000):
     
    test_mers0 = get_mers(7, num_mers)
    test_mers1 = get_mers(7, num_mers)

    ts = []
    for i in range(len(test_mers0)):
        cos = (mk_model.cosine_distance(test_mers0[i], test_mers1[i]))
        align_sim = pairwise2.align.globalxx(test_mers0[i], test_mers1[i], score_only=True)
        ts.append([cos, align_sim])

    ts = np.array(ts)
    cor, pval = stats.spearmanr(ts[:, 1], ts[:, 0])
    
    return ts, cor, pval
        
    
def arithmetic_experiment(operands=[(3,3), (4,4)], 
                          n_nns=[1, 5, 10], 
                          concatenation='weak', 
                          samples=1000, 
                          use_random_snippet=False):
    

    """
    Searches nearest neighbors of a vector made by concatenating i-mer + j-mer 
    with the i-mer + j-mer as a concatenated string.
    
    WARNING: As currently implemented this is walk-away-from-your-computer slow for ~1000 samples.
    
    :operands: List of (i,j) k-mers to use
    :n_nns: Number of nearest neighbors to search
    :samples: Number of searches to perform
    :concatenation: Concatenation style. Weak concatentation is order independent. Strong concatenation is in order i->j.
    :use_random_snippet: (CHANGES OUTPUT) Also report the result of a random snippet replacing the j-mer.
    
    """
        
    results_arithmetic = {}
    max_topn = np.max(n_nns)
    
    # Short function for comparions
    def if_concatenation_is_nn(L_0, L_1, nn_list, concatenation=concatenation):
        
        """
        Returns True if a concatenation of two strings in the list of nearest neighbors
        """
        
        if concatenation == 'strong':
            concat_query = (L_0 + L_1).encode("utf-8")
            if concat_query in nn_list:
                is_nn = True
            else:
                is_nn = False

        elif concatenation == 'weak':
            concat_queries = ((L_0 + L_1).encode("utf-8"), (L_1 + L_0).encode("utf-8"))
            if len(set(concat_queries).intersection(set(nn_list))) > 0:
                is_nn = True
            else:
                is_nn = False
                
        return is_nn
     
    # Run experiment
    for l_operands in operands:

        # Initialize data
        matches = np.zeros((samples, len(n_nns)))        
        if use_random_snippet:
            snippet_matches = np.zeros((samples, len(n_nns)))
            
        for s in range(samples):

            # Make k-mers (equal length)
            L_0 = get_mers(k=l_operands[0], num_mers=1)[0]
            L_1 = get_mers(k=l_operands[1], num_mers=1)[0]
            snippet = get_mers(k=l_operands[1], num_mers=1)[0]
            
            # Generate vector
            query_vec = mk_model.vector(L_0) + mk_model.vector(L_1)    

            # Get top N nearest neighbors of vector
            nns = most_similar(mk_model, query_vec, is_vector=True, same_k=False, topn=max_topn)
            nn_list = pd.DataFrame(nns)['kmer'].tolist()
            
            for n, topn in enumerate(n_nns):
                
                # Is query string among those top N nearest neighbors of vector?
                matches[s, n] = if_concatenation_is_nn(L_0, L_1, nn_list[0:topn], concatenation=concatenation)
                
                if use_random_snippet:
                    snippet_matches[s, n] = if_concatenation_is_nn(L_0, snippet, nn_list[0:topn], concatenation=concatenation)
       
        matching_fractions = (pd.DataFrame(matches).sum() / pd.DataFrame(matches).count()).values
        
        if use_random_snippet == False:
            results_arithmetic[l_operands] = matching_fractions
        else:
            snippet_fractions = (pd.DataFrame(snippet_matches).sum() / pd.DataFrame(snippet_matches).count()).values
            results_arithmetic[l_operands] = [matching_fractions, snippet_fractions]

    return results_arithmetic



# Generates tables and figures from preprint

def generate_figure1(num_mers=10000):

    ts, cor, pval = cosine_alignment_experiment(num_mers)
    
    df = pd.DataFrame(ts, columns=['cosine', 'alignment'])

    df.to_csv(f'{outdir}/figure-1-data.csv')
    
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(by='alignment',
               ax=ax)
    ax.set_title('')
    ax.set_title('Relationship Between Cosine Similarity\nand Alignment Score\n(spearman cor = {})'.format(round(cor, 3)),
                 loc='left', ha='left')
    ax.set_xlabel('Sequence Alignment Score')
    ax.set_ylabel('Cosine Similarity')
    fig.suptitle('')
    fig.set_facecolor('white')
    plt.savefig(f'{outdir}/figure-1.png', dpi=200, bbox_inches="tight")


def generate_table1(n_samples=1000):
    
    # Experiment: recapitulation of table 1 from dna2vec
    n_nns = [1, 5, 10, 30]
    experiment_1st = arithmetic_experiment(operands=[(3,3), (3,4), (3,5), (4,4)],
                                           n_nns=n_nns, 
                                           samples=n_samples, 
                                           concatenation = 'weak', 
                                           use_random_snippet=False)


    df_1st = pd.DataFrame(experiment_1st).T * 100
    df_1st.columns = [str(n)+'-NN' for n in n_nns]
    df_1st['Concatenated'] = df_1st.reset_index()[['level_0', 'level_1']].values.sum(axis=1)
    df_1st[['Concatenated'] + [str(n)+'-NN' for n in n_nns]]
    df_1st.to_csv(f'{outdir}/table-1.csv')

    
def generate_table2(n_samples=1000):

    # Experiment: recapitulation of table 2 from dna2vec

    experiment_2nd_weak = arithmetic_experiment(operands=[(3,3), (4,3), (5,3), (4,4)],
                                               n_nns=[1, 5, 10, 30], 
                                               samples=n_samples, 
                                               concatenation = 'weak', 
                                               use_random_snippet=True)

    experiment_2nd_strong = arithmetic_experiment(operands=[(3,3), (4,3), (5,3), (4,4)],
                                               n_nns=[1, 5, 10, 30], 
                                               samples=n_samples, 
                                               concatenation = 'strong', 
                                               use_random_snippet=True)


    df_2nd = pd.concat([pd.DataFrame(experiment_2nd_weak).T, pd.DataFrame(experiment_2nd_strong).T], axis=1)
    df_2nd.columns = ['weak-concat analogy 1/5/10/30-NN', 'weak-concat scrambled-snippet 1/5/10/30-NN',                  
                      'strong-concat analogy 1/5/10/30-NN','strong-concat scrambled-snippet 1/5/10/30-NN',]
    col_order = ['weak-concat scrambled-snippet 1/5/10/30-NN',  'weak-concat analogy 1/5/10/30-NN',                  
                      'strong-concat scrambled-snippet 1/5/10/30-NN', 'strong-concat analogy 1/5/10/30-NN',]

    df_2nd = df_2nd[col_order]
    df_2nd.to_csv(f'{outdir}/table-2.csv')


def generate_figure3(n_samples=1000):
    
    # Experiment: recapitulation of figure 3 from dna2vec
    n_nns = range(1,100,2)
    experiment_3rd = arithmetic_experiment(operands=[(5,3)],
                                           n_nns=n_nns, 
                                           samples=n_samples, 
                                           concatenation = 'strong', 
                                           use_random_snippet=True)


    df_3rd = pd.DataFrame({'n-Nearest Neighbors' : n_nns,
                          'analogy' : list(experiment_3rd.values())[0][0], 
                          'scrambled-snippet' : list(experiment_3rd.values())[0][1]})


    df_3rd.to_csv(f'{outdir}/figure-3-data.csv')
    
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(df_3rd['n-Nearest Neighbors'], 1000*df_3rd['analogy'], c='firebrick')
    ax.plot(df_3rd['n-Nearest Neighbors'], 1000*df_3rd['scrambled-snippet'], c='slateblue')

    ax.set_xlabel('n-Nearest Neighbors')
    ax.set_ylabel('count (out of 1000 samples)')

    fig.set_facecolor('white')
    plt.savefig(f'{outdir}/figure-3.png', dpi=200, bbox_inches="tight")

    
def main():

    global mk_model
    global outdir
    
    parser = argparse.ArgumentParser(description="""Generate tables and figures from 'dna2vec: Consistent 
    vector representations of variable-length k-mers'""")

    parser.add_argument('-n', '--num-samples', dest='n_samples', default=1000, nargs=1,
                        help='Number of samples for arithmetic experiments (tables 1 and 2, figure 3)')
    parser.add_argument('-e', '--embedding', dest='embed_to_read', nargs=1, help='Output directory')
    parser.add_argument('-o', '--output-dir', dest='outdir', nargs=1, help='Output directory')
    
    args = parser.parse_args()
    n_samples = int(args.n_samples[0])
    embed_to_read = args.embed_to_read[0]
    outdir = args.outdir[0]
    
    if outdir not in os.listdir():
        os.mkdir(outdir)
    
    # Used previously
    # outdir = 'epoch1'
    # embed_to_read = '/data/mwiest/dna2vec-20200825-2123-k3to8-100d-10c-32980Mbp-sliding-9bf_epoch2.w2v'
    # n_samples = 1000
    
    mk_model = MultiKModel(embed_to_read)
    
    print('Generating Figure 1...')
    generate_figure1(num_mers=10000)
    print('Generating Table 1 (slow)...')
    generate_table1(n_samples=n_samples)
    print('Generating Table 2 (very slow)...')
    generate_table2(n_samples=n_samples)
    print('Generating Figure 3 (slow)...')
    generate_figure3(n_samples=n_samples)
    print('... Done!')
    
if __name__ == "__main__":
    main()
