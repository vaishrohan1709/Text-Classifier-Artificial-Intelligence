import os
import sys
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.stem import PorterStemmer
from copy import deepcopy
from nltk.corpus import stopwords

def get_content(data):
    stop_words= set (stopwords.words('english'))
    num_files=len(data.data)
    for i in range(num_files):
        lines = data.data[i].splitlines()
        flag=0
        data.data[i] = '\n'
        for line in lines:
            if 'Lines:' in str(line):
                flag=1
            elif flag==1:
                data.data[i] += str(line)
    j=0
    to_stem = True
    to_stop = True
    stemmer = PorterStemmer()
    while j<4:
        if j==0:
            to_stem = True
            to_stop = True
        elif j==1:
            to_stem = True
            to_stop = False
        elif j==2:
            to_stem = False
            to_stop = True
        elif j==3:
            to_stem = False
            to_stop = False
        back_up_data=deepcopy(data)
        for i in range(num_files):
            lines = back_up_data.data[i].splitlines()
            new_lines = ''
            back_up_data.data[i] = '\n'
            for line in lines:
                old_tokens = line.split(' ')
                new_line = ' '
                for token in old_tokens:
                    # token = token.decode('utf-8', 'ignore')
                    if to_stop==True:
                        if (token and token.lower()) not in stop_words:
                            if to_stem==True:
                                token = stemmer.stem(token)
                                new_line += str(token) + ' '
                            else:
                                new_line += str(token) + ' '
                    else:
                        if to_stem==True:
                            token = stemmer.stem(token)
                            new_line += str(token) + ' '
                        else:
                            new_line += str(token) + ' '
                new_lines+=new_line
            back_up_data.data[i] += new_lines
        if j == 0:
            data_stem_stop = back_up_data
        elif j==1:
            data_stem_Nstop = back_up_data
        elif j==2:
            data_Nstem_stop =back_up_data
        elif j==3:
            data_Nstem_Nstop = back_up_data
        j+=1

    return data_stem_stop, data_stem_Nstop, data_Nstem_stop, data_Nstem_Nstop

def tokenizer_train(data, type):

    uni_data = type.fit_transform(data.data)
    return uni_data

def tokenizer_test(data, type):
    uni_data = type.transform(data.data)
    return uni_data

def classify(train_data,train_data_truth,test_data,test_data_truth,table_entries, type):
    algo_name='NB'
    algo=MultinomialNB(alpha=0.01)
    print(algo_name + type)
    var = algo.fit(train_data, train_data_truth.target)
    pred_class = var.predict(test_data)
    table_entries.append(
        [algo_name,type,float(metrics.precision_score(test_data_truth.target, pred_class, average='macro')),
        float(metrics.recall_score(test_data_truth.target, pred_class, average='macro')),
        float(metrics.f1_score(test_data_truth.target, pred_class, average='macro'))])
    '''LC_dict.setdefault(algo_name+type, {'F1': []})
    LC_dict[algo_name+type]['F1'].append(float(metrics.f1_score(test_data_truth.target, pred_class, average='macro')))'''

def write_to_file(output,table_entries):
    file=open(output,'w')
    temp=''
    for entry in table_entries:
        for item in entry:
            temp= temp+ str(item)+","
        temp=temp.rstrip(',')
        temp=temp+"\n"
        file.write(temp)
        temp=''
    file.close()

if __name__ == '__main__':
    trainset = str(sys.argv[1])
    testset = str(sys.argv[2])
    output = str(sys.argv[3])

    train_data = load_files(trainset)
    test_data = load_files(testset)

    train_data_stem_stop, train_data_stem_Nstop, train_data_Nstem_stop, train_data_Nstem_Nstop= get_content(train_data)
    test_data_stem_stop, test_data_stem_Nstop, test_data_Nstem_stop, test_data_Nstem_Nstop= get_content(test_data)

    CV_uni_YY = CountVectorizer(ngram_range=(1, 1), decode_error='ignore')
    CV_uni_YN = CountVectorizer(ngram_range=(1, 1), decode_error='ignore')
    CV_uni_NY = CountVectorizer(ngram_range=(1, 1), decode_error='ignore')
    CV_uni_NN = CountVectorizer(ngram_range=(1, 1), decode_error='ignore')


    TV_uni_YY = TfidfVectorizer(ngram_range=(1, 1), decode_error='ignore')
    TV_uni_YN = TfidfVectorizer(ngram_range=(1, 1), decode_error='ignore')
    TV_uni_NY= TfidfVectorizer(ngram_range=(1, 1), decode_error='ignore')
    TV_uni_NN = TfidfVectorizer(ngram_range=(1, 1), decode_error='ignore')

    CV_train_data_stem_stop = tokenizer_train(train_data_stem_stop,CV_uni_YY)
    CV_train_data_stem_Nstop = tokenizer_train(train_data_stem_Nstop,CV_uni_YN)
    CV_train_data_Nstem_stop = tokenizer_train(train_data_Nstem_stop,CV_uni_NY)
    CV_train_data_Nstem_Nstop = tokenizer_train(train_data_Nstem_Nstop,CV_uni_NN)

    CV_test_data_stem_stop = tokenizer_test(test_data_stem_stop,CV_uni_YY)
    CV_test_data_stem_Nstop = tokenizer_test(test_data_stem_Nstop,CV_uni_YN)
    CV_test_data_Nstem_stop = tokenizer_test(test_data_Nstem_stop,CV_uni_NY)
    CV_test_data_Nstem_Nstop = tokenizer_test(test_data_Nstem_Nstop,CV_uni_NN)

    TV_train_data_stem_stop = tokenizer_train(train_data_stem_stop,TV_uni_YY)
    TV_train_data_stem_Nstop = tokenizer_train(train_data_stem_Nstop,TV_uni_YN)
    TV_train_data_Nstem_stop = tokenizer_train(train_data_Nstem_stop,TV_uni_NY)
    TV_train_data_Nstem_Nstop = tokenizer_train(train_data_Nstem_Nstop,TV_uni_NN)

    TV_test_data_stem_stop = tokenizer_test(test_data_stem_stop,TV_uni_YY)
    TV_test_data_stem_Nstop = tokenizer_test(test_data_stem_Nstop,TV_uni_YN)
    TV_test_data_Nstem_stop = tokenizer_test(test_data_Nstem_stop,TV_uni_NY)
    TV_test_data_Nstem_Nstop = tokenizer_test(test_data_Nstem_Nstop,TV_uni_NN)

    LC_dict={}
    table_entries=[]
    classify(CV_train_data_stem_stop,train_data_stem_stop,CV_test_data_stem_stop,test_data_stem_stop,table_entries,'CountVectorizer, Stemmer and Stop Words Removed')
    classify(CV_train_data_stem_Nstop, train_data_stem_Nstop, CV_test_data_stem_Nstop, test_data_stem_Nstop, table_entries, 'CountVectorizer, Stemmer and No Stop Words Removed')
    classify(CV_train_data_Nstem_stop, train_data_Nstem_stop, CV_test_data_Nstem_stop, test_data_Nstem_stop,
             table_entries, 'CountVectorizer, No Stemmer and Stop Words Removed')
    classify(CV_train_data_Nstem_Nstop, train_data_Nstem_Nstop, CV_test_data_Nstem_Nstop, test_data_Nstem_Nstop,
             table_entries, 'CountVectorizer, No Stemmer and No Stop Words Removed')

    classify(TV_train_data_stem_stop, train_data_stem_stop, TV_test_data_stem_stop, test_data_stem_stop,
             table_entries, 'TfidfVectorizer, Stemmer and Stop Words Removed')
    classify(TV_train_data_stem_Nstop, train_data_stem_Nstop, TV_test_data_stem_Nstop, test_data_stem_Nstop,
             table_entries, 'TfidfVectorizer, Stemmer and No Stop Words Removed')
    classify(TV_train_data_Nstem_stop, train_data_Nstem_stop, TV_test_data_Nstem_stop, test_data_Nstem_stop,
             table_entries, 'TfidfVectorizer, No Stemmer and Stop Words Removed')
    classify(TV_train_data_Nstem_Nstop, train_data_Nstem_Nstop, TV_test_data_Nstem_Nstop, test_data_Nstem_Nstop,
             table_entries, 'TfidfVectorizer, No Stemmer and No Stop Words Removed')
    print(table_entries)


    write_to_file(output,table_entries)

