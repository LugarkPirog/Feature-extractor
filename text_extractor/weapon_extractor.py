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
        voc1 = ['купить', 'покупать', 'брать', 'есть', 'помочь', 'нужный', 'пневматический', 'травматический']
        voc2 = ['heckler koch', 'двухстволк', 'пулемёт', 'нитрометан', 'нитронафталин', 'нитроглицерин', 'colt',
                'калаш', 'газовый', 'мосина', 'огнестрел', 'карабин, ствол, тт', 'нитрат аммоний', 'стечкин', 'патрон',
                'мр', 'browning', 'mp', 'калашников', 'ппш, аммиачный селитра, тротил', 'нож', 'шашка тротиловый', 'ак',
                'иж', 'винтовка', 'револьвер', 'бронежилет', 'акм', 'нитрогликоль', 'ствол', 'гранат ф', 'шт',
                'нитрат целлюлоза', 'гексоген', 'винтовка ввс', 'гремучий ртуть', 'карабин тигр', 'тт', 'хлорат калия',
                'азид свинец', 'оружие', 'glock', 'пистолет макаров', 'жидкий кислород', 'макарыч', 'walther', 'ргд',
                'нитротолуол', 'свд', 'наган', 'перхлорат, винтовка', 'орсис', 'пзрк, рпк', 'ярыгин', 'бронник',
                'марголин', 'мм', 'октоген', 'апс', 'маузер', 'акс', 'пероксид ацетон', 'оса', 'свл', 'травмат',
                'огнестрела', 'пм', 'barret', 'herstal', 'боевой', 'птур', 'асвк', 'травматический', 'beretta',
                'бердыш', 'пистолет', 'разрешение', 'гранатомёт', 'свт', '']
        voc3 = ['проба', 'концерн', 'амф', 'меф', 'кристал', 'закладк', 'курьер', 'аккаунт', 'cs go', 'mdma',
                'кокаин', 'дроп', 'клад', 'кладмен', 'гашиш', 'шишк', 'bitcoin', 'криптовалюта', 'работа',
                'заработок', 'гаш', 'мг', 'диллер', 'репорт', 'трип', 'трип репорт', 'розница', 'опт', 'опт розница',
                'кс го', 'акк', 'банк', 'карта', 'смс оплата подтверждение', 'спс', 'оплата', 'снятие средств',
                'скрипт', 'снилс']
        word_arr = []
        for w1 in voc1:
            a = [w1]
            a.extend(voc2)
            word_arr.extend([(str(i[0]) + ' ' + str(i[1])) for i in combinations(a, 2)])
        word_arr.extend(voc2)
        word_arr.extend(voc3)

        word_arr = list(set(np.array(word_arr).ravel()))
        #  ---------------  доработать тут надо, какой-нибудь permutations вставить или combinations
        #  ---------------  UPD: добавил
        for word in word_arr:
            q = 1 if word in string else 0
            feature_arr.append(q)

        return feature_arr

    def cat_to_numeric(self, features, labels):
        # Надо учитывать разницу в кол-ве отрицательных и положительных примеров !!!
        r, c = features.shape
        y_ = sum(int(float(val)) for val in labels)
        mean_y = y_ / r
        alpha = 20
        cache = []
        for col in self.categ:
            z = sum([1 if i == 0 else 0 for i in features[:, col]])
            o = sum([1 if i == 1 else 0 for i in features[:, col]])
            # div = z / o
            # ATTENTION HERE !!! -------------------------
            l0 = (sum([1 if features[i, col] == 0 and str(labels[i]) == '1' else 0 for i in range(r)])/y_*o + mean_y * alpha) / (z + alpha)
            l1 = (sum([1 if features[i, col] == 1 and str(labels[i]) == '1' else 0 for i in range(r)])/y_*z + mean_y * alpha) / (o + alpha)
            # THANKS !!! ---------------------------------
            # l0, l1 = l0 / div, l1 * div
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

    def __del__(self):
        del self.cache
        del self.categ

if __name__ == '__main__':
    import pandas as pd
    s = [['собака кошка травмат купить броник', '1'], ['купить мм патроны нужен разрешение', '1'],
         ['qewrio nreljkqw nnmd', '0']]
    f = FeatureExtractor()
    # a = f.fit_transform(s)
    L = pd.read_csv('d:/NN1/data/real/test2.csv', sep=';', encoding='windows-1251', usecols=[0], skip_blank_lines=True,
                    error_bad_lines=False).values.astype('str')
    labels = pd.read_csv('d:/NN1/data/real/test2.csv', sep=';', usecols=[1], encoding='windows-1251',
                         skip_blank_lines=True, error_bad_lines=False).values
    labels = np.array([1 if i == 1 else 0 for i in labels], np.int32).reshape([-1, 1])
    data = np.array(L[:, 0]).reshape((-1, 1))
    transform = np.hstack((data, labels))
    a = f.fit_transform(transform)
    print(f.cache)
    print(a)

    # data = L
