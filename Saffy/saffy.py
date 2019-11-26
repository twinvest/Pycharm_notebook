import re
count = 0

list1 = list()
N = input()
N = int(N)
for i in range(1, N+1):
    list1.append(str(i))

print(list1)

regex = re.compile('[0-9]*1+')
for i in list1:
    if regex.match(i):
        print(i)
        count = count+1

print(count)