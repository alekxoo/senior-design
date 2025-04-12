#Servo motor control library
#Written by: Abdullah Alwakeel
#Date: Apr 12 2025

import threading
from Focuser import Focuser
#use threads for I/O (blocking) calls
f = Focuser(7)

SERVO_ADDR = Focuser.OPT_MOTOR_Y
servoReading = f.get(SERVO_ADDR)

def vel_y(v):
	t = threading.Thread(target=set_vel_y, args=(v,))
	t.start()

def set_vel_y(v):
	global servoReading
	old_r = f.get(SERVO_ADDR)
	servoReading = max(0, min(95, old_r + v)) #clamp between 0 and 95
	f.set(SERVO_ADDR, servoReading)