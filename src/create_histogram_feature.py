import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

print "Run store_histogram_feature in case not done already"

def get_best_histogram(pair_score):  
    return np.histogram(pair_score.values(), bins="auto") 


pair_score = pickle.load(open("./feature/pair.p", "r"))
hist, bins = get_best_histogram(pair_score)

def get_0_score_idx():
    return np.digitize([0], bins)[0]

def get_feature(pair_list):
    '''
    @param pair_list: List of cross product features between user and question
    @param feature: feature = char/tag
     
    Given list of pair calculates a vector by using histogram
    '''    
    pair_list_score = []
    for pair in pair_list:
        if pair in pair_score:
            pair_list_score.append(pair_score[pair])
        
    set_indices = np.digitize(pair_list_score, bins)
    feature_vec = [0]*len(bins)
    for set_index in set_indices:
        
        feature_vec[set_index-1] += 1.0
    
    return feature_vec
    
def plot_histogram(x, var_name, bins=10):
    '''
    plots histogram of x array, 10 bins equal size
    '''
    _, hist_data, _ = plt.hist(x, bins=bins)
    print type(hist_data), hist_data
    plt.plot(x=hist_data)
    plt.savefig(var_name, linewidth=0)
    plt.close()

def plot_best_histogram(pair_score):
    hist, bins = get_best_histogram(pair_score)
    
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    
    plt.savefig("best-freq_0-freq_1", linewidth=0)
    plt.close()
    
    

if __name__ == '__main__':
#     for bins in range(0,151,10):
#         if bins == 0:
#             bins = 10
#         plot_histogram(pair_score_word.values(), str(bins) + "-freq_0-freq_1-word", bins)
#     
#     plot_best_histogram(pair_score)
    count=0
    for score in pair_score.values():
#         print score
        if score == -1.0:
            count+=1
            
    print "1 count", count   
    print len(bins), bins
    print len(hist), hist
    
    print list(np.digitize([-1,-0.95,-0.8,-0.5,0,0.5,0.95,1], bins))
    print hist[0],hist[-1]
    
    print get_feature([1,2,3,4,5,6,7])
    
