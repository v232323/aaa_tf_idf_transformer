import math


class CountVectorizer():
    """
    Convert a collection of text documents to a matrix of token counts.
    """

    def __init__(self):
        self._features_names = []

    def fit_transform(self, corpus):
        """
        Learn the vocabulary dictionary and return document-term matrix.
        :param corpus: Some list of strings
        :return: Document-term matrix.
        """
        feature_names_set = set()
        for str in corpus:
            str_list = str.split(' ')
            for word in str_list:
                feature_names_set.add(word.lower())
        self._features_names = list(feature_names_set)

        result = []
        for str in corpus:
            str_list = str.lower().split(' ')
            matrix_row = [0] * len(list(feature_names_set))
            for word in str_list:
                for num, feature in enumerate(list(feature_names_set)):
                    if word == feature:
                        matrix_row[num] += 1
            result.append(matrix_row)
        return result

    def get_feature_names(self):
        """
        Array mapping from feature integer indices to feature name.
        :return: A list of feature names.
        """
        return self._features_names


def tf_transform(count_matrix):
    result = []
    for row in count_matrix:
        mas = []
        for el in row:
            mas.append(el / sum(row))
        result.append(mas)
    return result


# inverse document_frequency
# idf = ln ((всего документов+1)/(документов со словом+1)) + 1
def idf_transform(count_matrix):
    count_all = len(count_matrix)
    count_with_word = [0] * len(count_matrix[0])
    for row in count_matrix:
        for i, el in enumerate(row):
            if el > 0:
                count_with_word[i] += 1

    idf = []
    for el in count_with_word:
        idf.append(math.log((count_all + 1) / (el + 1)) + 1)
    return idf


class TfidfTransformer():
    """
    Transform a count_matrix to tf-idf
    """

    def fit_transform(self, count_matrix):
        tf = tf_transform(count_matrix)
        idf = idf_transform(count_matrix)
        result = []
        for row in tf:
            mas = []
            for el1, el2 in zip(row, idf):
                mas.append(el1 * el2)
            result.append(mas)
        self.tfidf = result
        return result


class TfidVectorizer(CountVectorizer):
    """
    Convert a collection of raw documents to a matrix of TF-IDF features.
    """

    def __init__(self):
        super().__init__()
        self.tfifd = TfidfTransformer()

    def fit_transform(self, corpus):
        count_matrix = super().fit_transform(corpus)  # получаем метод из родителя CountVectorizer
        return self.tfifd.fit_transform(count_matrix)  # используем метод из TfidfTransformer


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]

    #vectorizer = CountVectorizer()
    #count_matrix = vectorizer.fit_transform(corpus)
    #print(vectorizer.get_feature_names())
    #print(count_matrix)
    #print(tf_transform(count_matrix))
    #print(idf_transform(count_matrix))
    #transformer = TfidfTransformer()
    #tfidf_matrix = transformer.fit_transform(count_matrix)
    #print(tfidf_matrix)

    vectorizer = TfidVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(tfidf_matrix)
