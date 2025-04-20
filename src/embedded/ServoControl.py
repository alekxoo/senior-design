#Servo motor control library
#Written by: Abdullah Alwakeel
#Date: Apr 12 2025

import threading
from Focuser import Focuser
#use threads for I/O (blocking) calls
f = Focuser(7)
f.set(Focuser.OPT_MODE, 0x01) #turn on

SERVO_ADDR = Focuser.OPT_MOTOR_Y
servoReading = f.get(SERVO_ADDR)

def vel_y(v):
	t = threading.Thread(target=set_vel_y, args=(v,))
	t.start()

def set_vel_y(v):
	global servoReading
	# old_r = f.get(SERVO_ADDR)
	servoReading = max(65, min(145, int(servoReading + v))) #clamp between 0 and 95)
	f.set(SERVO_ADDR, servoReading, flag=0) #flag=0 means don't wait for response from device


if __name__ == "__main__":
	while True:
		try:
			inp = input("Enter change in angle (int) (non-numeric value = exit): ")
			i = int(inp)
			vel_y(i)
		except:
			break