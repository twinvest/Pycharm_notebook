import pyautogui

def menustart():
    pyautogui.click(50, 1050)
    pyautogui.typewrite('memo', interval=0.01)
    pyautogui.press('enter')
    pyautogui.click(500, 500)
def make_print():
    pyautogui.click(500, 500)
    pyautogui.doubleClick()
    pyautogui.hotkey('win', 'up')
    for a in range(1, 10):
        pyautogui.typewrite('y o u d i e!\nI\'m D E V I L\n', interval=0.001)


def click_windows():
    pyautogui.moveTo(50, 1050)
    for a in range(101):
        pyautogui.click(50, 1050)
        print(a)


def main():
    #windows1()
    menustart()
    make_print()
    #click_windows() d

    # for a in range(100, 1000, 200):
    #     for b in range(100, 400, 50):
    #         pyautogui.moveTo(a, b)
    #         pyautogui.click(a, b)
    #         print(a)

if __name__ == "__main__":
    main()