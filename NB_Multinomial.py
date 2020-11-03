

import numpy as np

class CountVectorizer():
    def __init__(self,corpus,max_df=1.0,minLen=2):
        '''
        Parameters
        ----------
        corpus: input corpus
        max_df : float, default=1.0
            Tokens that have a document frequency strictly higher than the given threshold (corpus-specific
            stop words) will be ignored.
        minLen: int, default = 2
            A token is defined as a combination of minLen or more alphanumeric characters.
        '''

        self.corpus = corpus
        self.max_df = max_df
        self.minLen = 2
        self.vocabulary_dic= {}
        self.word_token_rule = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _")


    def Doc2List(self,document,ifSort=True):

        values = self.word_token_rule

        def remover(aString = ""):
            for item in aString:
                if item not in values:
                    aString = aString.replace(item, " ")
            return aString

        listFromDoc=[element.lower() for element in remover(document).split() if len(element)>=self.minLen]

        if ifSort:
            listFromDoc.sort()
        return listFromDoc



    def BuildVocabularyDic(self):
        # Build Vocabulary from a corpus,
        # Return type is dictionary

        initial_list=[]
        for document in self.corpus:
            string_from_doc=self.Doc2List(document,ifSort=False)
            initial_list=initial_list+string_from_doc
        initial_list=list(set(initial_list))
        initial_list.sort()

        vocabulary_dic_len=0
        for element in initial_list:
            self.vocabulary_dic[element]=vocabulary_dic_len
            vocabulary_dic_len += 1

        return None


    def BuildWordCountDocument(self,document):
        n_row=1
        n_col=len(self.vocabulary_dic)
        CountVectorizer_vector = np.zeros((n_row,n_col)).astype('int64')
        string_from_doc=self.Doc2List(document)
        for string_element in string_from_doc:
            if string_element in self.vocabulary_dic:
                CountVectorizer_vector[0,self.vocabulary_dic[string_element]] +=1

        return CountVectorizer_vector


    def BuildWordCountCorpus(self,*args):

        if len(args) ==0:

            n_row=len(self.corpus)
            n_col=len(self.vocabulary_dic)
            self.CountVectorizer_array = np.zeros((n_row,n_col)).astype('int')

            for (i,document) in enumerate(self.corpus):
                self.CountVectorizer_array[i,:] = self.BuildWordCountDocument(document)

            if self.max_df < 1.0:
                self.__cutoff_with_maxdf()

            return None

        elif len(args) == 1:
            corpus = args[0]

            n_row=len(corpus)
            n_col=len(self.vocabulary_dic)
            CountVectorizer_array = np.zeros((n_row,n_col)).astype('int')

            for (i,document) in enumerate(corpus):
                CountVectorizer_array[i,:] = self.BuildWordCountDocument(document)


            return CountVectorizer_array
        else:
            raise("Too many arguments.")


    def __cutoff_with_maxdf(self):

        ifExist = self.CountVectorizer_array > 0
        docNumber = self.CountVectorizer_array.shape[0]

        df = np.sum(ifExist,axis=0) / docNumber
        inds = np.argwhere(df <= self.max_df)
        inds =inds.reshape(len(inds),)
        VocDic_cutoff = {key: value for key,value in self.vocabulary_dic.items() if value in inds }

        CountVectorizer_array_cutoff = self.CountVectorizer_array[:,inds]

        self.vocabulary_dic = VocDic_cutoff
        self.CountVectorizer_array = CountVectorizer_array_cutoff

        return None

    def get_feature_names(self):
        return list(self.vocabulary_dic.keys())

###---Multinomial distribution---###
class Multinomial:
    def __init__(self, alpha = 1.0):
        self.alpha = alpha
    def train(self, train_data):
        self.train_data = train_data
        self.n_features_ = train_data.shape[1]
        self.feature_count_ = np.sum(train_data,axis=0)

        Nyi = self.feature_count_
        Ny = np.sum(Nyi)

        self.log_prob = (np.log(Nyi+self.alpha)-np.log(Ny+self.alpha*self.n_features_))

    def loglikelihood(self, test_data):
        test_log_prob = test_data.dot(self.log_prob)
        return test_log_prob

###- Bayes Classifier####

class BayesClassifier:
    def __init__(self, alpha):
        self.alpha = alpha

        self.maximum_likelihood_models=[]
        self.priors = None
        self.n_classes = None


    def train(self,X_train,Y_train):

        label_list=np.unique(Y_train)

        total_num = len(Y_train)
        self.n_classes = len(label_list)

        model_ml=[]
        priors= np.zeros(self.n_classes )

        for i in label_list:
            X_train_i = X_train[Y_train == i,:]
            priors[i] = len(X_train_i)/total_num
            model_class_i = Multinomial(alpha=self.alpha)
            model_class_i.train(X_train_i)
            model_ml.append(model_class_i)


        self.maximum_likelihood_models = model_ml
        self.priors = priors

        assert len(self.maximum_likelihood_models) == len(self.priors)

        return None


    # Returns a matrix of size number of test ex. times number of classes containing the log
    # probabilities of each test example under each model, trained by ML.
    def loglikelihood(self, test_data):

        log_pred = np.zeros((test_data.shape[0], self.n_classes))

        for i in range(self.n_classes):
            # Here, we will have to use maximum_likelihood_models[i] and priors to fill in
            # each column of log_pred (it's more efficient to do a entire column at a time)
            log_pred[:, i] = self.maximum_likelihood_models[i].loglikelihood(test_data)+np.log(self.priors[i])

        return log_pred

    def get_pred(self,test_data):
        # Calculate the log-probabilities according to our model
        logprob = self.loglikelihood(test_data) #

        # Predict labels
        classes_pred = np.argmax(logprob,axis=1)#
        return classes_pred
