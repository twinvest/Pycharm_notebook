class Car:
    honk = "빵빵"
    def __init__(self, color, year):
        self.color = color
        self.year = year
        print("새로운 Car 인스턴스가 생성되었습니다.")

    def get_info(self):
        print("color : %s ,year: %d" % (self.color, self.year))

    def call(self, tw1, tw2):
        output1 = self.tw1
        output2 = self.tw2
        #output3 = self.tw3
        return output1, output2