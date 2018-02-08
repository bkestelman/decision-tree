from decisionTree import *

# All featureEntropies should be <= entropy(data) since they are conditional
def test_featureEntropies(data):
    _entropy = entropy(data)
    _featureEntropies = featureEntropies(data)
    for e in _featureEntropies:
        if e > _entropy:
            return False
    return True

def printResult(testName, result):
    print('TEST ' + testName)
    if result:
        print('TEST PASSED')
    else:
        print('TEST FAILED')

def test_numFeatures(data):
    num = numFeatures(data)
    i = 0
    for row in data:
        #print(len(row))
        if len(row) != num:
            print('Row ' + str(i) + ' only has ' + str(len(row)) + ' features')
            return False
        i += 1
    return True 

def self_test(data):
    root = DecisionTree(data)
    print('PROCREATE')
    print('********************')
    root.procreate() # train
    print('********************')
    correct = 0
    i = 0
    for row in data: 
        node = root.predict(row)
        guess = prob(node.data)
        if guess == 1:
            if row[LABEL_COL] == LABELS[0]:
                correct += 1
        elif guess == 0:
            if row[LABEL_COL] == LABELS[1]:
                correct += 1
        else: 
            print('Self test encountered an uncertain prediction at node')
            node.printMe()
            return False 
        i += 1
        if i != correct:
            print('Incorrect prediction for row ')
            print(row)
            print('at node')
            node.printMe()
            print('node contains data ' )
            print(node.data)
            if search_for(row, node.data):
                print('This node contains the row')
            while node.parent is not None:
                node = node.parent
                node.printMe()
                if search_for(row, node.data):
                    print('This node contains the row')
            return False
    if correct == len(data):
        return True
    return False

def search_for(_row, data):
    for row in data:
        if row[ID_COL] == _row[ID_COL]:
            return True
    return False

# Run all tests
def runAllTests():
    while True:
        path = askForFile()
        try:
            with open(path, newline='') as csvFile:
                reader = csv.reader(csvFile, delimiter=',')
                data = []
                for row in reader:
                    data.append(row)
                printResult('featureEntropies', test_featureEntropies(data))
                printResult('numFeatures', test_numFeatures(data))
                printResult('self_test', self_test(data[:TEST_SIZE]))
            break
        except IOError:
            print('Could not open file, IOError')
         
