# lines = open('example.txt', 'r').readlines()
# test_dictionary = dict()
# for sentence in lines:
#     item = sentence.split()
#     for i in item:
#         if i in test_dictionary:
#             a = test_dictionary.get(i)
#             a+=1
#             test_dictionary[i] = a
#         else:
#             test_dictionary[i] = 1
# print(test_dictionary)

import operator
from collections import defaultdict
lines = open('example.txt', 'r').readlines()
print(lines)
lines = [line.strip() for line in lines] #lines를 짤라서 개행을 지우고 리스트형태로 만들었다.
print(lines)
# words_dict = {}
words_dict = defaultdict(int)

for sentence in lines:              #리스트 형태의 문장을 for문을 통해 하나씩 가져와서
    for word in sentence.split(' '):#띄어쓰기 단위로 나누고
        words_dict[word] += 1       #해당 키의 벨류값을 1씩 증가시킨다.
print(words_dict)
print(sorted(words_dict))

sdict = sorted(words_dict.items(), key=operator.itemgetter(0))
for i in sdict:
    if i[1] >= 10:
        print(i)
# for word in sorted(words_dict.keys()):
#      if words_dict[word] >= 10:
#          print(word, words_dict[word])