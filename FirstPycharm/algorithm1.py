def pick(n, list, toPick):
    if(toPick == 0):
        print(list)
    smallest = 0 if not list else list[-1] + 1
    for next in range(smallest, n):
        list.append(next)
        pick(n, list, toPick-1)
        list.pop()

pick(4, [], 2)