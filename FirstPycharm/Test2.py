import copy

list1=[10,20,30,40,30,20,10]
list2 = copy.deepcopy(list1)

print(list1, list2)

list3 = list()
for i in range(0, 101):
    if i%2 == 0:
        list3.append(i)
print(list3)

set1 = {1,2,3}
set2 = {2,3,4}

plus = set2.union(set1)
inter = set2.intersection(set1)

list4 = list(plus)
list5 = list(inter)
list4.sort()
list5.sort()
print(list4, list5)