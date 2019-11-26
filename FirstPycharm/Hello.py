import random

def draw_a_card(cardset):
    select = random.randint(0, 14)
    return cardset.pop(select)
    '''
    전체 카드의 리스트를 argument로 받음
    카드 리스트에서 임의의 카드 한장을 뽑고 이를 리스트에서 지움
    random.randint() 함수 활용, list 내장함수 pop() 활용
    :return: 뽑은 카드 값
    '''
    #pass

def get_score(score, name):
    score = sum(score)
    print(score, name)
    return score
    '''
    가지고 있는 카드들의 리스트를 argument로 받음
    카드 소유주를 나타내는 문자열을 argument로 받음
    합계 점수를 계산, sum() 함수 활용
    현재 보유중인 카드 리스트를 출력
    점수 합산 결과를 카드 소유주가 누구인지와 함께 출력 (print)
    :return: 계산된 합계 점수
    '''
    #pass

def print_result(a, b):
    if(a>b):
        print('내가이김')
    else:
        print('딜러가이김')
    '''
    나의 점수와 딜러의 점수를 arguments로 받음
    대소 비교를 통해 결과를 출력 (print)
    :return: None
    '''
    pass

cards = list(range(1,14)) * 4
print(cards)
my_cards, dealer_cards = [], []

for i in range(2):
    my_cards.append(draw_a_card(cards))
    dealer_cards.append(draw_a_card(cards))

my_sum = get_score(my_cards, 'me')
dealer_sum = get_score(dealer_cards, 'dealer')

print_result(my_sum, dealer_sum)