from itertools import combinations

import numpy


def extract_feats_from_string(string):
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
        a.extend(voc2)
        word_arr.extend([(str(i[0]) + ' ' + str(i[1])) for i in combinations(a, 2)])
    word_arr.extend(voc2)

    word_arr = list(set(np.array(word_arr).ravel()))
    #  ---------------  доработать тут надо, какой-нибудь permutations вставить или combinations
    #  ---------------  UPD: добавил
    for word in word_arr:
        q = 1 if word in string else 0
        feature_arr.append(q)

    return feature_arr