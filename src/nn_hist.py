
import load_data_set as ld
from create_histogram_feature import get_feature

from keras.layers import Dense
from keras.models import Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Dropout

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import f1_score, make_scorer

def genmodel(num_units, actfn='relu', reg_coeff=0.0, last_act='softmax'):
    ''' Generate a neural network model of approporiate architecture
    Args:
        num_units: architecture of network in the format [n1, n2, ... , nL]
        actfn: activation function for hidden layers ('relu'/'sigmoid'/'linear'/'softmax')
        reg_coeff: L2-regularization coefficient
        last_act: activation function for final layer ('relu'/'sigmoid'/'linear'/'softmax')
    Output:
        model: Keras sequential model with appropriate fully-connected architecture
    '''

    model = Sequential()
    for i in range(1, len(num_units)):
        if i == 1 and i < len(num_units) - 1:
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=actfn, 
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
            model.add(Dropout(0.2))
        elif i == 1 and i == len(num_units) - 1:
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=last_act, 
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
            model.add(Dropout(0.2))
        elif i < len(num_units) - 1:
            model.add(Dense(output_dim=num_units[i], activation=actfn, 
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
            model.add(Dropout(0.2))
        elif i == len(num_units) - 1:
            model.add(Dense(output_dim=num_units[i], activation=last_act, 
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        
    return model

def transform_label(labels):
    labels_new = []
    for label in labels:
        label_new = [0.0,0.0]
        label_new[int(label)]=1.0
        labels_new.append(label_new)
    
    return labels_new

def original_label(label):
    return [ 0*l[0] + 1*l[1] for l in label]


def normalize(X_tr, X_te):
    ''' Normalize training and test data features
    Args:
        X_tr: Unnormalized training features
        X_te: Unnormalized test features
    Output:
        X_tr: Normalized training features
        X_te: Normalized test features
    '''
    X_mu = np.mean(X_tr, axis=0)
    X_tr = X_tr - X_mu
    X_sig = np.std(X_tr, axis=0)
    X_tr = X_tr/X_sig
    X_te = (X_te - X_mu)/X_sig
    return X_tr, X_te

def score_f1_class1(ground_truth, predictions):
    '''
    Returns f1 score for class 1
    '''
    return f1_score(ground_truth, predictions, average='binary', pos_label=1)

X_tr, X_te = [],[]

for train in ld.X_train :
    X_tr.append(get_feature(train))
    
for test in ld.X_test:
    X_te.append(get_feature(test))

X_tr, X_te = np.array(X_tr), np.array(X_te)

col_deleted = np.nonzero((X_tr==0).sum(axis=0) > (len(X_tr)-1000))
# col_deleted = col_deleted[0].tolist() + range(6,22) + range(28,44)
print col_deleted
X_tr = np.delete(X_tr, col_deleted, axis=1)
X_te = np.delete(X_te, col_deleted, axis=1)

X_tr, X_te = normalize(X_tr, X_te)
y_tr, y_te = transform_label(ld.y_train), ld.y_test

print "training and testing data normalised ", len(y_tr), len(y_te)

momentum = 0.99
eStop = True
sgd_Nesterov = True
sgd_lr = 1e-5
batch_size=1000
nb_epoch=50
verbose=True

def run_NN(arch, reg_coeff, sgd_deca):
        
    call_ES = EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='auto')
    
    # Generate Model
    model = genmodel(num_units=arch, reg_coeff=reg_coeff )
    # Compile Model
    sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=momentum, 
        nesterov=sgd_Nesterov)
    
    # sgd = RMSprop(lr=sgd_lr, rho=0.9, epsilon=1e-08, decay=sgd_decay)
    
    model.compile(loss='binary_crossentropy', optimizer=sgd, 
        metrics=['accuracy'])
    # Train Model
    if eStop:
        model.fit(X_tr, y_tr, nb_epoch=nb_epoch, batch_size=batch_size, 
        verbose=verbose, callbacks=[call_ES], validation_split=0.1, 
        validation_data=None, shuffle=True)
    else:
        model.fit(X_tr, y_tr, nb_epoch=nb_epoch, batch_size=batch_size, 
            verbose=verbose)
    
    labels_pred = model.predict_classes(X_te)
    
    y_true, y_pred = y_te, labels_pred
    
    print y_true[0], y_pred[0]
    print "arch, reg_coeff, sgd_decay", arch, reg_coeff, sgd_decay

    report = classification_report(y_true, y_pred)
    print report
    with open("results_nn.txt", "a") as f:
        f.write(report)
        f.write("\n")
        f.write(" ".join([str(s) for s in ["arch, reg_coeff, sgd_decay", arch, reg_coeff, sgd_decay]]))
        f.write("\n")

arch_range = [[len(X_tr[0]),1024,2], [len(X_tr[0]),1024,512,2], [len(X_tr[0]),1024,1024,2]]
reg_coeffs_range = [1e-6, 5e-6, 1e-5, 5e-5, 5e-4 ]
sgd_decays_range = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4 ]
class_weight_0_range = [1]
subsample_size_range = [2,2.5,3]

#GRID SEARCH ON BEST PARAM
for arch in arch_range:
    for reg_coeff in reg_coeffs_range:
        for sgd_decay in sgd_decays_range:
            run_NN(arch, reg_coeff, sgd_decay)

print "done"    
    
