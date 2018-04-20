import sys
from matplotlib import pyplot as plt
from matplotlib import colors
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
        lines = train_data.data[i].splitlines()
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

def classify_all_models(training_data,train_data, testing_data, test_data, gram, table_entries, LC_dict, step_size):
    Label= ''
    if gram is 'unigram':
        Label= 'UB'

    elif gram is 'bigrams':
        Label= 'BB'


    i=0
    algo=''
    algo_name=''
    while i<4:
        if i==0:
            algo_name='NB'
            algo=MultinomialNB()
        elif i==1:
            algo_name='LR'
            algo=LogisticRegression()
        elif i==2:
            algo_name='RF'
            algo=RandomForestClassifier()
        elif i==3:
            algo_name='SVM'
            algo=LinearSVC()
        print(algo_name + Label)
        var = algo.fit(training_data[:step_size], train_data.target[:step_size])
        pred_class = var.predict(testing_data)
        table_entries.append(
            [algo_name, Label, float(metrics.precision_score(test_data.target, pred_class, average='macro')),
             float(metrics.recall_score(test_data.target, pred_class, average='macro')),
             float(metrics.f1_score(test_data.target, pred_class, average='macro'))])
        LC_dict.setdefault(algo_name, {'Size': [], 'F1': []})
        LC_dict[algo_name]['Size'].append(step_size)
        LC_dict[algo_name]['F1'].append(float(metrics.f1_score(test_data.target, pred_class, average='macro')))
        i+=1

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


def plot_LC(LC_dict,display_LC):
    fig = plt.figure()
    plt.title('LC Plot')
    #plt.ylim(0.5, 1)
    plt.xlabel("Training Data Size")
    plt.ylabel("F1 Score")
    #plt.grid()
    i=0
    for name in LC_dict:
        if i==0:
            color=colors.cnames['red']
        elif i==1:
            color = colors.cnames['blue']
        elif i==2:
            color = colors.cnames['green']
        elif i==3:
            color = colors.cnames['yellow']
        i+=1
        plt.plot(LC_dict[name]['Size'], LC_dict[name]['F1'],'o-', color=color, label=name)

    lgd = plt.legend(loc=2, bbox_to_anchor=(0.5, -0.1))
    fig.savefig('plot', bbox_inches='tight')
    if int(display_LC)==1:
        plt.tight_layout()
        plt.show(lgd)


if __name__ == '__main__':
    trainset = str(sys.argv[1])
    testset = str(sys.argv[2])
    output = str(sys.argv[3])
    display_LC = str(sys.argv[4])

    train_data= load_files(trainset)
    test_data = load_files(testset)
    train_data, test_data= get_content(train_data, test_data)
    uni_train_data, uni_test_data, bi_train_data, bi_test_data= tokenizer(train_data,test_data)

    table_entries=[]
    LC_dict={}
    training_data=uni_train_data
    testing_data=uni_test_data
    classify_all_models(training_data, train_data, testing_data,test_data,'unigram', table_entries,LC_dict,uni_train_data.shape[0])

    training_data = bi_train_data
    testing_data = bi_test_data
    classify_all_models(training_data, train_data, testing_data, test_data, 'bigrams', table_entries,LC_dict,bi_train_data.shape[0])
    write_to_file(output, table_entries)

    #plotting
    training_data = uni_train_data
    testing_data = uni_test_data
    step_size=200
    train_size = uni_train_data.shape[0]
    LC_dict={}
    while step_size<= train_size:
        classify_all_models(training_data, train_data, testing_data, test_data, 'unigram', table_entries, LC_dict,step_size)
        step_size+=200


    if step_size>train_size and train_size%step_size is not 0:
        print('inside')
        classify_all_models(training_data, train_data, testing_data, test_data, 'unigram', table_entries, LC_dict, train_size)

    print("dict",LC_dict)
    plot_LC(LC_dict,display_LC)


