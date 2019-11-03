from pyfirmata import Arduino, util

import warnings
import serial
import serial.tools.list_ports


import time


#
# print ('\n\nPYSERIAL\n')
#
# sPort = '/dev/cu.usbmodemFA131'           #On Mac - find this using >ls /dev/cu.usb*
#
# aSerialData = serial.Serial(sPort,9600)     #COM port object
#

def arduino():
    ser = serial.Serial('/dev/cu.usbmodemFA131', 9800, timeout=1)
    loopTime = 1
    time.sleep(1)

    for x in range(int(loopTime)):
        ser.writelines(b'H')
        time.sleep(0.5)
        ser.writelines(b'L')
        time.sleep(0.5)
    ser.close()

arduino()
arduino()
arduino()
arduino()
arduino()
# def aruduino():
#     board = Arduino("/dev/cu.usbmodemFA131")
#     loopTimes = input(5)
#     print("Blinking " + loopTimes + " times.")
#
#     for x in range(int(loopTimes)):
#         board.digital[13].write(1)
#         time.sleep(0.2)
#         board.digital[13].write(0)
#         time.sleep(0.2)
#
# aruduino()
