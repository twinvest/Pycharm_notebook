def convert(x):
    return 9.0*x/5.0 + 32

def converter(temperature, convert):
    list2 = list()
    for i in temperature:
        list2.append(convert(i))
    return list2

if __name__ == "__main__":
    temper=[10, 20, 30, 40, 50]
    list1 = converter(temper, convert)
    print(list1)
