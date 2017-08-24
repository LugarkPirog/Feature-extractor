# import pickle
import numpy as np
from itertools import combinations, permutations


class FeatureExtractor(object):
    def __init__(self):
        self.cache = None
        self.categ = None

    def fit_transform(self, array):
        if type(array) == np.ndarray:
            arr = array
            # print('ndarray')
        elif type(array) == list:
            arr = np.array(array)
            # print(arr[:, 0])
        else:
            raise TypeError('Not a valuable array given!')
        labels = arr[:, -1]
        features = self.extract_feats(arr[:, 0])
        self.categ = self.count_categorical(features)
        out = self.cat_to_numeric(features, labels)
        return out

    def extract_feats(self, arr):
        feat_arr = []
        for string in arr:
            result = self.extract_feats_from_string(string)
            feat_arr.append(result)
        feat_arr = np.array(feat_arr)
        return feat_arr

    def extract_feats_from_string(self, string):
        """
        Tuned to detect weapon thematics in messages

        :param string: A string that needs to be featurized
        :return: An array of features

        """
        feature_arr = []
        voc1 = ['купить', 'покупать', 'брать', 'есть', 'помочь', 'нужный']
        voc2 = ['травмат', 'винтовка', 'мосина', 'газовый', 'пистолет', 'бронежилет', 'бронник', 'боевой',
                'травматический', 'оружие', 'мм', 'патрон', 'шт', 'наган', 'нож', 'огнестрел', 'травматический', 'тт',
                'оса', 'пм', 'двухстволк', 'пистолет', 'разрешение', 'макарыч', 'патрон', 'иж', 'травмат', 'калаш',
                'калашников', 'ствол']
        word_arr = []
        for w1 in voc1:
            a = [w1]
            for w2 in voc2:
                a.append(w2)
            word_arr.append([(str(i[0]) + ' ' + str(i[1])) for i in combinations(a, 2)])
        word_arr.append([(str(i[0]) + ' ' + str(i[1])) for i in permutations(voc2, 2)])
        res = []
        for i in word_arr:
            res += i
        word_arr = list(set(np.array(res).ravel()))
        #  ---------------  доработать тут надо, какой-нибудь permutations вставить или combinations
        #  ---------------  UPD: добавил
        for word in word_arr:
            q = 1 if word in string else 0
            feature_arr.append(q)

        return feature_arr

    def cat_to_numeric(self, features, labels):
        r, c = features.shape
        y_ = sum(int(val) for val in labels)
        mean_y = y_ / len(labels)
        alpha = 10
        cache = []
        for col in self.categ:
            z = sum([1 if i == 0 else 0 for i in features[:, col]])
            o = sum([1 if i == 1 else 0 for i in features[:, col]])
            l0 = (sum([1 if features[i, col] == 0 and str(labels[i]) == '1' else 0 for i in range(r)])/y_*z + mean_y * alpha) / (z + alpha)
            l1 = (sum([1 if features[i, col] == 1 and str(labels[i]) == '1' else 0 for i in range(r)])/y_*o + mean_y * alpha) / (o + alpha)
            cache.append((l0, l1))
            for el in range(r):
                features[el, col] = l0 if features[el, col] == 0 else l1
        self.cache = cache
        return features

    def cat_to_numeric_with_cache(self, features):
        for i, col in enumerate(self.categ):
            for num in range(len(features[:, col])):
                features[num, col] = self.cache[i][0] if features[num, col] == 0 else self.cache[i][1]
        return features

    @staticmethod
    def count_categorical(feats):
        unique = []
        # print('$$$$ ', feats.shape[1], ' $$$$')
        for i in range(feats.shape[1]):
            unique.append(len(np.unique(feats[:, i])))
        catcols = [i for i, j in enumerate(unique) if j == 2]
        return catcols

    def transform(self, array):
        feats = self.extract_feats(array)
        out = self.cat_to_numeric_with_cache(feats)
        return out
