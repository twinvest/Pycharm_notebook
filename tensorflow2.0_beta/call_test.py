#
#
# my_car = Car("Red", 2017)
# # 새로운 Car 인스턴스가 생성되었습니다.
#
# my_car.get_info()
# # color : Red, year: 2017
# callable(my_car)
#
# # list1=[1,2,3]
# # list2=[1,2,3]
# #
# #
# # car1, car2 = my_car(list1, list2)
# # print(car1, car2)

class Adder:
    def __init__(self, n):
         self.n = n
         print(self.n)
    def call(self, x, y):
        return self.n + x + y

plus_3 = Adder(3)
call_func = plus_3(4, 5)
print(call_func)

