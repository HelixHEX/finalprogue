from pyfirmata import Arduino, util

import time

def aruduino():
    board = Arduino("COM3")
    loopTimes = input(5)
    print("Blinking " + loopTimes + " times.")

    for x in range(int(loopTimes)):
        board.digital[13].write(1)
        time.sleep(0.2)
        board.digital[13].write(0)
        time.sleep(0.2)

aruduino()
