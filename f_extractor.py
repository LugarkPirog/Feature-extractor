# import pickle
import numpy as np


class FeatureExtractor(object):
    def __init__(self):
        self.cache = None

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
        features, categorical = self.extract_feats(arr[:, 0])
        out, cache = self.cat_to_numeric(features, labels, categorical)
        return out

    def extract_feats(self, arr):
        feat_arr = []
        for string in arr:
            result = self.extract_feats_from_string(string)
            feat_arr.append(result)
        feat_arr = np.array(feat_arr)
        categ = self.count_categorical(feat_arr)
        return feat_arr, categ

    def extract_feats_from_string(self, string):
        pass

    def cat_to_numeric(self, features, catcols, labels):
        r, c = features.shape
        y_ = sum(int(val) for val in labels)
        mean_y = y_ / len(labels)
        alpha = 10
        cache = []
        for col in catcols:
            z = sum([1 if i == 0 else 0 for i in features[:, col]])
            o = sum([1 if i == 1 else 0 for i in features[:, col]])
            l0 = (sum([1 if features[i, col] == 0 and str(labels[i]) == '1' else 0 for i in range(r)])/y_*z + mean_y * alpha) / (z + alpha)
            l1 = (sum([1 if features[i, col] == 1 and str(labels[i]) == '1' else 0 for i in range(r)])/y_*o + mean_y * alpha) / (o + alpha)
            cache.append((l0, l1))
            for el in range(r):
                features[el, col] = l0 if features[el, col] == 0 else l1
        self.cache = cache
        return features

    def cat_to_numeric_with_cache(self, features, catcols):
        for i, col in enumerate(catcols):
            for num in range(len(features[:, col])):
                features[num, col] = self.cache[i][0] if features[num, col] == 0 else self.cache[i][1]
        return features

    def count_categorical(self, feats):
        unique = []
        # print('$$$$ ', feats.shape[1], ' $$$$')
        for i in range(feats.shape[1]):
            unique.append(len(np.unique(feats[:, i])))
        catcols = [i for i, j in enumerate(unique) if j == 2]
        return catcols

    def transform(self, array, cache):
        pass
