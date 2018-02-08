import csv # for reading a csv file
import copy # for deep copies of data (so rows are also copied, not just referenced)
import traceback
from random import shuffle
from operator import itemgetter # for sorting lists by a specific index (sorting data by a specific feature)
from numpy import log
from colorama import init, deinit, Fore # for coloring certain debug messages

# Configurable Decision Tree Settings
LABEL_COL = 1 # which column of data contains label
ID_COL = 0 # which column of data contains ID
LABELS = ('B', 'M') # this would not work for continuous labels :P (ever deal with continuous labels?)
IGNORED_COLS = [LABEL_COL, ID_COL] # Code depends on IGNORED_COLS being the first COLS of the dataset (that is, if you choose to ignore COL 0 and COL 5, the code will not work!)

# Miscellaneous Settings
CROSS_VALIDATE = 5 # number of partitions for cross validation
TEST_SIZE = 103 # irrelevant (for testing without cross validation)

# Debug versions (can only debug one version at a time)
# See debug() definition
# Some functions in code are assigned a debug_level and contain debug statements
# These debug_levels allow you to debug one function at a time (you may want to create your own debug_level to debug a specific problem)  
DEBUG_THRESH = 3
DEBUG_FEATENTS = 2
DEBUG_PROCREATE = 4
DEBUG_PREDICT = 6
DEBUG_CROSS = 9
DEBUG_VERSION = 10 # set debug version here 
# You can safely ignore any debug() calls while reading my code 

### prob(data)
## Goal: 
# Calculates the probability of a row in the given data having LABELS[0] ('B' for this assignment) as its label 
## How it works:
# Just count the number of rows in data with label 'B' and divide by the total number of rows
## Inputs: 
# - data: given data to test 
## Returns: 
# The probability (float from 0 to 1) of a row in data having label LABELS[0] ('B')
def prob(data):
    hits = 0
    for row in data:
        if row[LABEL_COL] == LABELS[0]:
            hits += 1
    p = hits / len(data)
    if p < 0 or p > 1:
        print('EXCEPTION: prob < 0 or > 1') # I hope this doesn't happen! 
        quit()
    return p

### entropy(data) 
## Inputs: 
# - data: given data  
## Returns: 
# Entropy of data (float >= 0) based on probabilities of each label 
def entropy(data):
    p = prob(data) 
    if p == 0 or p == 1: 
        return 0 # entropy is 0 in these cases; avoid dividing by 0 ;)
    return - p * log(p) - (1-p) * log(1-p) 

### sortByFeature(data, feature) 
## Inputs: 
# - data: given data  
# - feature: feature/attribute to sort by
## Returns: 
# Given data, sorted by given feature
def sortByFeature(data, feature):
    for row in data:
        row[feature] = float(row[feature]) # don't sort strings! ('100' < '80')
    if feature is None:
        EXCEPTION('feature is None in sortByFeature()')
    if feature >= numFeatures(data):
        EXCEPTION('feature index out of bounds for data in sortByFeature()')
    return sorted(data, key=itemgetter(feature)) # itemgetter is from the imported operator module
    #return sorted(data, key=lambda x: x[feature])

### featureEntropy(data, feature)
## How it works:
# 1. Sort data by feature
# 2. Entropy on one side of threshold
# 3. Entropy on other side of threshold
# 4. Multiply each by probability of each
# 5. Add 'em up (weighted average)
## Inputs: 
# - data: given data
# - feature: feature to split on
## Returns:
# Entropy of data after splitting on feature at optimal threshold 
def featureEntropy(data, feature):
    debug_level = DEBUG_FEATENTS 
    newData = sortByFeature(data, feature) # threshold() expects data to be sorted by feature to split on 
    thresh = threshold(newData, feature) # see definition of threshold()
    total = len(newData) 
    debug(DEBUG_THRESH, 'thresh is ' + str(thresh))
    one_side = entropy(newData[:thresh]) * ( thresh/total ) # 
    debug(debug_level, 'entropy one_side (unweighted): ' + str(one_side / (thresh/total)))
    two_side = entropy(newData[thresh:]) * ( (total-thresh)/total ) 
    debug(debug_level, 'entropy two_side (unweighted): ' + str(two_side / ((total-thresh)/total)))
    debug(debug_level, 'featureEntropy is ' + str(one_side + two_side))
    return one_side + two_side

### featureEntropies(data)
## Inputs:
# - data: given data
## Returns:
# List of entropies after splitting on each feature. Caller will likely call min(featureEntropies(data))
## Note:
# featureEntropies skips IGNORED_COLS, such as ID and label, so the entropies list returned will not have as many columns as a full row of data
# Therefore, the caller will probably want to add len(IGNORED_COLS) to the index of min(featureEntropies(data)), so as to get the actual column index of the feature 
def featureEntropies(data):
    count = 0
    for row in data:
        if len(row) != numFeatures(data):
            count += 1
    if count > 0:
        EXCEPTION('A row is missing features ')
    debug_level = DEBUG_FEATENTS 
    entropies = []
    debug(debug_level, 'Calculating featureEntropies')
    for feature in range(0, numFeatures(data)): # assumes all rows in data are same length, see definition of numFeatures()
        if feature in IGNORED_COLS: # ignore cols like ID or label
            debug(debug_level, 'Skipping feature ' + str(feature))
            continue # next/skip
        debug(debug_level, 'Calculating featureEntropy for feature ' + str(feature))
        entropies.append(featureEntropy(data, feature)) # Calculate entropy after splitting on this feature and add to list of entropies
    return entropies

### threshold(data, feature)
## How it works: 
# Uses a binary search to approximate the position of the optimal threshold (maybe a little rough, but gets good results and is VERY fast)
## Notes:
# Assumes data has been sorted according to the feature to split on (use sortByFeature(data, feature) before calling threshold(data, feature))
# Assumes len(data) > 1
# Assumes entropy(data) is not 0 (this is important!)
## Inputs:
# - data
# - feature: feature to split on and calculate threshold for
## Returns:
# The INDEX (row number) of the optimal threshold to split on (not the actual value)
## Note:
def threshold(data, feature):
    debug_level = DEBUG_THRESH 
    debug(debug_level, 'Calculating threshold for feature ' + str(feature))
    if len(data) <= 1:
        EXCEPTION('len(data) <= 1 in threshold()')
    start = 0
    end = len(data)
    prev_thresh = int((end - start)/2) # try threshold in the middle between start and end of data
    total = len(data)
    if prev_thresh == 0: # equivalent to end == 1
        EXCEPTION('initial prev_thresh == 0 in threshold()')
    prev_ent = entropy(data[:prev_thresh])*prev_thresh/total + entropy(data[prev_thresh:])*(total-prev_thresh)/total
    # sorry, no recursion here :( 
    while start + 1 < end: # if start is 1 less than end or greater, we're done - return prev_thresh
        debug(debug_level, 'start: ' + str(start) + ' end: ' + str(end) + ' prev_thresh: ' + str(prev_thresh))
        left = int(1/4*(end-start)) + 1 # threshold on left to try
        left_ent = entropy(data[:left])*left/total + entropy(data[left:])*(total-left)/total # entropy of data splitting at left
        right = int(3/4*(end-start)) # threshold on right to try
        right_ent = entropy(data[:right])*right/total + entropy(data[right:])*(total-right)/total # entropy of data splitting at right
        if left_ent < prev_ent and left_ent <= right_ent: # if left_ent is good, go left
            debug(debug_level, 'going left to ' + str(left))
            end = prev_thresh
            prev_thresh = left 
            prev_ent = left_ent
        elif right_ent < prev_ent: # if right_ent is good, go right
            debug(debug_level, 'going right to ' + str(right))
            start = prev_thresh
            prev_thresh = right 
            prev_ent = right_ent
        else: # prev_ent is better than left_ent and right_ent, return
            return prev_thresh
    return prev_thresh

### numFeatures(data)
## Inputs:
# - data
## Returns:
# Number of features in the first row of data (assumes all rows have the same number of features)
def numFeatures(data):
    ret = len(data[0])
    i = 0
    for row in data:
        if len(row) != ret:
            EXCEPTION('numFeatures broken at ' + str(i))
        i += 1
    return len(data[0])

### printSummary(data)
## Inputs:
# - data
# Print useful info about the given data, including:
# Size of data
# Probability of each label
# Entropy of data 
def printSummary(data):
    print('----------------')
    print('Size of data: ' + str(len(data)))
    hits = 0
    for row in data:
        if row[LABEL_COL] == LABELS[0]:
            hits += 1
    p = hits / len(data)
    print('Probability of label ' + LABELS[0] + ': ' + str(p))
    print('Probability of label ' + LABELS[1] + ': ' + str((1-p)))
    print('Entropy of data: ' + str(entropy(data)))

### debug(debug_version, obj)
## Inputs:
# - debug_version: only print debug message if given debug_version equals the set DEBUG_VERSION 
# - obj: the object to print
def debug(debug_version, obj):
    if debug_version == DEBUG_VERSION:
        print('DEBUG ' + str(debug_version) + ': ')
        print(obj)

### EXCEPTION(msg)
## Inputs:
# - msg: error message to print
## Print error message in red and quit 
def EXCEPTION(msg):
    init() # colorama
    print(Fore.RED + 'EXCEPTION: ' + msg + Fore.RESET)
    deinit()
    traceback.print_stack()
    quit()

### askForFile()
# Ask user to input path to data file
def askForFile():
    path = input('Enter path to data file or hit ENTER for default ("wdbc.data"):')
    if path == '':
        path = 'wdbc.data'
    return path

class DecisionTree:
### DecisionTree constructor
# Creates a DecisionTree node
## Inputs:
# - data: data for this node (root node gets full training data, children will get split data)
# - root: boolean, is this the root node? (optional, default=True)
# - level: what level of the tree is this (root level==0)? Mostly used for debugging (optional, default=0 to match default root)
# - child: string, indicates if this node is a 'Left' or 'Right' child of its parent (root has no parent, so child=='NA') (optional, default='NA' to match default root)
    def __init__(self, data, root=True, level=0, child='NA', parent=None):
        checkDups(data)
        self.data = copy.deepcopy(data) # procreate will delete features from data, so we need a copy of data to avoid changing the original (deepcopy because the rows are also lists)
        checkDups(self.data)
        self.root = root
        self.level = level
        self.child = child
        self.isLeaf = False
        self.splitFeat = None # which feature does this node split on (to be set in procreate, will remain None for leaf nodes)
        self.splitVal = None # what VALUE (not index) of threshold did this node split on (to be set in procreate, remains None for leaf nodes)
        self.left = None
        self.right = None
        self.parent = parent

### procreate()
# Creates the whole decision tree starting from a DecisionTree node. 
# The root node is given a set of data when it's constructed 
# 1. Determine the best feature and threshold to split on
# 2. Create left child with left side of data after split, and right child with right side of data
# 3. Call procreate on children (you get a recursion)
## Base cases:
# - Length of data is <= 1 (nothing to split; also we want to avoid breaking threshold())
# - Entropy of data is 0 (no reason to split)
    def procreate(self):
        debug_level = DEBUG_PROCREATE
        debug(6, 'Procreate data size: ' + str(len(self.data)))
        if len(self.data) <= 1: # base case, return
            self.isLeaf = True 
        #    self.printMe()
            return
        if entropy(self.data) == 0: # base case, return
            self.isLeaf = True
        #    self.printMe()
            return
        feat_ents = featureEntropies(self.data) 
        splitFeat = feat_ents.index(min(feat_ents)) + len(IGNORED_COLS) # index of feature with minimum entropy (feature to split on). See featureEntropies() definition for why need to add len(IGNORED_COLS) 
        sortedData = sortByFeature(self.data, splitFeat) # has to be a deepcopy for when the feature is deleted from the children (sortedData will be split and passed to children)
        thresh = threshold(sortedData, splitFeat) 
        feats = []
        i = 0
        for row in sortedData:
            feats.append(row[splitFeat])
            i += 1
            if i == thresh:
                feats.append('|')
        #print(feats)
        leftData = copy.deepcopy(sortedData[:thresh]) # data to pass to left child
        rightData = copy.deepcopy(sortedData[thresh:]) # data to pass to right child
        self.splitFeat = splitFeat
        debug(6, 'Taking average of ' + str(float(leftData[-1][splitFeat])) + ' and ' + str(float(rightData[0][splitFeat])))
        self.splitVal = (float(leftData[-1][splitFeat]) + float(rightData[0][splitFeat])) / 2 # take average of values surrounding threshold INDEX to get split VALUE
        debug(6, 'Splitting on ' + str(splitFeat) + ' with splitVal ' + str(self.splitVal))
        #self.printMe()
        for row in leftData:
            del row[splitFeat] # delete the feature that was split on from left child's data
        if numFeatures(leftData) != numFeatures(self.data) - 1:
            EXCEPTION('remove feature failed')
        for row in rightData:
            del row[splitFeat] # delete the feature that was split on from right child's data
        if numFeatures(rightData) != numFeatures(self.data) - 1:
            EXCEPTION('remove feature failed')
        #self.printMe()
        self.left = DecisionTree(leftData, root=False, level=self.level+1, child='Left', parent=self) # create left child
        self.right = DecisionTree(rightData, root=False, level=self.level+1, child='Right', parent=self) # create right child
        self.left.procreate() # procreate
        self.right.procreate() # procreate

### printMe()
# Print useful info about DecisionTree node
    def printMe(self):
        #if not self.root:
         #   return
        printSummary(self.data)
        for row in self.data:
            if row[ID_COL] == '842302':
                print('Found the missing row here')
        if self.isLeaf:
            print('Leaf')
        print('Level: ' + str(self.level), ' Child: ' + self.child)
        print('Splitting on feature: ' + str(self.splitFeat) + ' (out of ' + str(len(self.data[0])) + ' features) with threshold value: ' + str(self.splitVal))
        print('-----------------------')

### iHave(_id)
# Used for debugging 
## Inputs: 
# - _id: ID of a row 
## Returns:
# True if this DecisionTree node contains the row with ID == _id
    def iHave(self, _id):
        for row in self.data:
            if row[ID_COL] == _id:
                return True
        return False

### predict(_row)
# Predict the label for a row of data by traversing the DecisionTree
## Inputs:
# - _row: row of data with label to predict  
# - debugMe: boolean, sets the debug_level of this function to DEBUG_VERSION (forces debug() to run on a specific call to predict())
## Returns:
# Probability that _row has label LABELS[0] ('B')
    def predict(self, _row, debugMe=False):
        row = copy.copy(_row) # create copy of _row (features need to be deleted when traversing tree to make a prediction, to stay consistent with the data stored in the tree, which has deleted features)
        debug_level = DEBUG_PREDICT
        if debugMe: 
            debug_level = DEBUG_VERSION
        if self.root:
            debug(debug_level, 'ROOT Predicting...')
        if self.isLeaf: # Base case: make prediction here 
            debug(debug_level, 'Prediction: ' + str(prob(self.data)) + ' ' + LABELS[0])
            debug(debug_level, 'Actual label: ' + str(row[LABEL_COL]))
            #return prob(self.data)
            return self
        debug(debug_level, 'row[' + str(self.splitFeat) + ']: ' + str(row[self.splitFeat]) + ' splitVal: ' + str(self.splitVal))
        if float(row[self.splitFeat]) < self.splitVal: # test this DecisionTree node's threshold/split VALUE of the feature it splits on against the given row's value of the same feature 
            debug(debug_level, 'going left') # if the row's value is < the threshold, go left
            del row[self.splitFeat] # delete the feature from the (copy of) the row
            return self.left.predict(row, debugMe) # keep going through left child (pass it row with deleted split feature and debugMe)
        else:
            debug(debug_level, 'going right') # if the row's value is < the threshold, go right 
            del row[self.splitFeat] # delete the feature from the (copy of) the row
            return self.right.predict(row, debugMe) # keep going through left child (pass it row with deleted split feature and debugMe)

### test(data)
# Make a prediction for each row of data (predict(row)) 
## Inputs:
# - data
# - skip_from, skip_to: skip rows from index skip_from to index skip_to (use these to skip testing on the training partition)
# no return, just print results
    def test(self, data, skip_from, skip_to):
        true_p = true_n = false_p = false_n = 0
        total = 0
        for row in data:
            if total >= skip_from and total < skip_to:
                total += 1
                continue
            resultNode = self.predict(row)
            if prob(resultNode.data) > 0.5:
                if row[LABEL_COL] == LABELS[0]:
                    true_n += 1
                else:
                    false_n += 1
            elif prob(resultNode.data) < 0.5:
                if row[LABEL_COL] == LABELS[1]:
                    true_p += 1
                else:
                    false_p += 1
            total += 1
        print('True positives: ' + str(true_p))
        print('True negative: ' + str(true_n))
        print('False positives: ' + str(false_p))
        print('False negative: ' + str(false_n))
        print(Fore.GREEN + 'Accuracy: ' + str((true_p+true_n)/total))
        if true_p + false_p != 0:
            print('Precision: ' + str(true_p/(true_p+false_p)))
        else:
            print('Precision: 0.0')
        if true_p + false_n != 0:
            print('Recall: ' + str(true_p/(true_p+false_n)) + '\n' + Fore.RESET)
        else:
            print('Recall: 0.0' + Fore.RESET)


###########################################

### checkDups(data)
# Check if data has duplicate rows (for debugging)
def checkDups(data):
    i = 0
    for row in data:
        if i == 0:
            continue
        if row[ID_COL] == data[0][ID_COL]:
            print(row)
            print(data[0])
            EXCEPTION('duplicate rows in data')
        i += 1

### cross_validate(data, num_partitions)
# Splits the data into random partitions 
# (creating partitions randomly should produce a roughly even distribution of labels among the partitions)
# Trains a DecisionTree on each partition and tests on the rest of the data
# Prints results 
## Inputs:
# - data
# - num_partitions: number of partitions 
def cross_validate(_data, num_partitions):
    print('Cross validating with ' + str(num_partitions) + ' partitions...')
    debug_level = DEBUG_CROSS
    data = copy.deepcopy(_data) # getting the hang of this yet? sure took me a while
    print('Shuffling data')
    shuffle(data) # randomizing order of data should give roughly even distributions of labels when partitioned
    #checkDups(data)
    partition_size = int(len(data) / num_partitions)
    print('Size of each partition: ' + str(partition_size) + '\n')
    partitions = []
    prev_i = 0
    for i in range(1, num_partitions + 1):
        print(Fore.CYAN + 'Training DecisionTree on partition ' + str(i-1) + Fore.RESET)
        print('--Using data rows ' + str(prev_i*partition_size) + ' to ' + str(i*partition_size))
        partition = data[prev_i*partition_size : i*partition_size]
        root = DecisionTree(partition) # create a DecisionTree with this partition's data
        root.procreate() # train the tree on this partition
        print('Root node: ')
        root.printMe()
        print('--Testing DecisionTree on rest of data')
        root.test(data, prev_i*partition_size, i*partition_size)
        prev_i = i
        
### main (yeah, I know, it's WEIRD in python, shoulda used perl)
# This is so that if I import this file somewhere else, the following won't automatically run 
if __name__ == "__main__":
    while True:
        path = askForFile() # ask for path to data file
        try:
            with open(path, newline='') as csvFile: # try to open the file (with just closes it automatically at the end or if there's an error)
                reader = csv.reader(csvFile, delimiter=',') 
                data = [] # put data in here
                for row in reader:
                    data.append(row) # each row is a list of ID, label, and 30 features 
                init()
                cross_validate(data, CROSS_VALIDATE)
                print(Fore.CYAN + 'BONUS: Training on only two rows!' + Fore.RESET)
                print('Randomly selecting one \'B\' row and one \'M\' row to train on') 
                shuffle(data)
                b_row = m_row = None
                for row in data:
                    if b_row is None and row[LABEL_COL] == LABELS[0]:
                        b_row = row
                    elif m_row is None and row[LABEL_COL] == LABELS[1]:
                        m_row = row
                restof_data = data
                data = [b_row, m_row]
                for row in data:
                    debug(1, row)
                root = DecisionTree(data)
                root.procreate()
                root.printMe()
                true_p = true_n = false_p = false_n = 0
                total = 0
                for row in restof_data:
                    guessNode = root.predict(row)
                    if prob(guessNode.data) > 0.5:
                        if row[LABEL_COL] == 'B':
                            true_n += 1
                        else:
                            false_n += 1
                    elif prob(guessNode.data) < 0.5:
                        if row[LABEL_COL] == 'M':
                            true_p += 1
                        else:
                            false_p += 1
                    total += 1
                print(Fore.GREEN + 'Accuracy: ' + str((true_p+true_n)/total))
                if true_p + false_p != 0:
                    print('Precision: ' + str(true_p/(true_p+false_p)))
                else:
                    print('Precision: 0.0')
                if true_p + false_n != 0:
                    print('Recall: ' + str(true_p/(true_p+false_n)) + '\n' + Fore.RESET)
                else:
                    print('Recall: 0.0' + Fore.RESET)
                deinit() # colorama
            break
        except IOError:
            print('Could not open file, IOError')

