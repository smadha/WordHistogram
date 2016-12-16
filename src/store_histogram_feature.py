'''
compute Cartesian product of user_id and question_id characters and tags
'''

from collections import Counter
import cPickle as pickle
import load_data_set as ld

# Each class has it's own Counter to keep track of pair frequency in that class
pairs_class_i = [Counter([]), Counter([])]
list_all_pairs = []
# score for all pairs 
pair_score = {}

def add_pair(pair, class_i):
    '''
    Adds a pair to Counter of class_i tuples
    class_i = 0/1 based on class id
    pair = One pair of character observed in question_id, user_id
    '''
    class_i = int(class_i)
    pairs_class_i[class_i].update([pair])


if __name__ == '__main__':

    # Each class has it's own Counter to keep track of pair frequency in that class
    pairs_class_i = [Counter([]), Counter([])]
    list_all_pairs = []
    # score for all pairs 
    pair_score = {}

    count = 0
    for idx,training_data in enumerate(ld.X_train):
        
        label = ld.y_train[idx]
                
        for word in training_data:
            add_pair(word, label)
            
        count += 1
        if(count % 5000) == 0:
            print count, "processed"
                
                
    #list_all_pairs is all possible list of pairs found training data
    list_all_pairs = set(pairs_class_i[0].keys() + pairs_class_i[1].keys())
    
    print "Unique pairs in class0", len(pairs_class_i[0])
    print "Unique pairs in class1", len(pairs_class_i[1])
    print "Unique pairs in training data", len(list_all_pairs)
        
    for pair in list_all_pairs:
        freq_0 = pairs_class_i[0][pair]
        freq_1 = pairs_class_i[1][pair]
        score = 1.0 * (freq_0 - freq_1) / sum([freq_0,freq_1])
        pair_score[pair] = score
        
    
    print "pairs generated for", len(pair_score)
    
    pickle.dump( pair_score, open("./feature/pair.p", "wb"), protocol=2 )
    

