import os
import sys
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics



#change
def get_content(train_data, test_data):
    num_files_train=len(train_data.data)
    num_files_test = len(test_data.data)
    for i in range(num_files_train):
        lines = train_data.data[i].splitlines()git
        flag=0
        train_data.data[i] = '\n'
        for j, line in enumerate(lines):
            if 'Lines:' in str(line):
                flag=1
            elif flag==1:
                train_data.data[i] += str(line)

    for i in range(num_files_test):
        lines = test_data.data[i].splitlines()
        flag = 0
        test_data.data[i] = '\n'
        for j, line in enumerate(lines):
            if 'Lines:' in str(line):
                flag = 1
            elif flag == 1:
                test_data.data[i] += str(line)

    return train_data, test_data

def tokenizer(train_data,test_data):

    uni = CountVectorizer(ngram_range=(1,1))

    bi = CountVectorizer(ngram_range=(1,2))

    uni_train_data = uni.fit_transform(train_data.data)
    uni_test_data = uni.transform(test_data.data)

    bi_train_data = bi.fit_transform(train_data.data)
    bi_test_data = bi.transform(test_data.data)

    return uni_train_data, uni_test_data, bi_train_data, bi_test_data

def classify_all_models(training_data,train_data, testing_data, test_data, gram, table_entries):
    Label= ''
    if gram is 'unigram':
        Label= 'UB'

    elif gram is 'bigrams':
        Label= 'BB'

    print('Naive Bayes '+ Label)
    NB = MultinomialNB().fit(training_data, train_data.target)
    pred_class = NB.predict(testing_data)
    table_entries.append(['NB', Label,float(metrics.precision_score(test_data.target, pred_class, average='macro')),float(metrics.recall_score(test_data.target, pred_class, average='macro')),float(metrics.f1_score(test_data.target, pred_class, average='macro'))])

    print('Logistic regression ' + Label)
    NB = LogisticRegression().fit(training_data, train_data.target)
    pred_class = NB.predict(testing_data)
    table_entries.append(['LR', Label,float(metrics.precision_score(test_data.target, pred_class, average='macro')),float(metrics.recall_score(test_data.target, pred_class, average='macro')),float(metrics.f1_score(test_data.target, pred_class, average='macro'))])

    print('Random Forrest ' + Label)
    NB = RandomForestClassifier().fit(training_data, train_data.target)
    pred_class = NB.predict(testing_data)
    table_entries.append(['RF',Label,float(metrics.precision_score(test_data.target, pred_class, average='macro')),float(metrics.recall_score(test_data.target, pred_class, average='macro')),float(metrics.f1_score(test_data.target, pred_class, average='macro'))])

    print('SVM ' + Label)
    NB = LinearSVC().fit(training_data, train_data.target)
    pred_class = NB.predict(testing_data)
    table_entries.append(['SVM', Label,float(metrics.precision_score(test_data.target, pred_class, average='macro')),float(metrics.recall_score(test_data.target, pred_class, average='macro')),float(metrics.f1_score(test_data.target, pred_class, average='macro'))])


def write_to_file(output,table_entries):
    file=open(output,'w')
    temp=''
    for entry in table_entries:
        for item in entry:
            temp= temp+ str(item)+","
        temp=temp.rstrip(',')
        temp=temp+"\n"
        print(temp)
        file.write(temp)
        temp=''


    file.close()

if __name__ == '__main__':
    trainset = str(sys.argv[1])
    testset = str(sys.argv[2])
    output = str(sys.argv[3])
    display_LC = str(sys.argv[4])

    train_data= load_files(trainset, shuffle=False)
    test_data = load_files(testset, shuffle=False)
    train_data, test_data= get_content(train_data, test_data)
    uni_train_data, uni_test_data, bi_train_data, bi_test_data= tokenizer(train_data,test_data)

    #print('uni:',uni_train_data)
    #print('uni test',uni_test_data)
    #print('bi:', bi_train_data)
    #print('bi test', bi_test_data)

    table_entries=[]

    training_data=uni_train_data
    testing_data=uni_test_data
    classify_all_models(training_data, train_data, testing_data,test_data,'unigram', table_entries)

    training_data = bi_train_data
    testing_data = bi_test_data
    classify_all_models(training_data, train_data, testing_data, test_data, 'bigrams', table_entries)

    print(table_entries)
    write_to_file(output,table_entries)

    #print('train data:', train_data.data)
    #print('test data:', test_data.data)
    #print('hey')
