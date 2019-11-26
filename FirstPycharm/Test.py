# list1=['a', 'b', 'c', 'd']
# list2=list1[:]
#
# print(id(list1), id(list2)) #리스트는 뮤터블 따라서 서로 다른 주소값을 생선한다.
# list2[1] ='x'
# print(list1, list2)



# list1=['a','b',['qwer', 'xy']]
# list2=list1[:]
#
# print(id(list1), id(list2))
# print(id(list1[2][1]), id(list2[2][1]))
# list2[2][1] = 'd'
# print(list1, list2)



list1=[1,2,3]
list2=list1
print(id(list1), id(list2))

list1.append([1,2])
list2.append([1,2])
print(id(list1), id(list2))

list1.extend([3,4])
list2.extend([3,4])
print(id(list1), id(list2))

print(list1, list2)
list1 = list1 + list2
list2 = list2 + list1
print(id(list1), id(list2)) #당연히 바뀔거임 왜? 뮤터블이니까. 새로운 걸 생성한거지.