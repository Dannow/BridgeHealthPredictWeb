from re import search, I

# 用于正则表达式模糊匹配
class RegexMap:
    def __init__(self, n_dic, val):
        self._items = n_dic
        self.__val = val

    def __getitem__(self, key):
        for regex in self._items.keys():
            if search(regex, key, I):
                return self._items[regex]
        return self.__val