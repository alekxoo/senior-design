#PID control library
#Written by: Abdullah Alwakeel
#Date: Apr 12 2025

from ServoControl import vel_y
from StepperControl import vel_x

PX = 3.0
IX = 1.5

PY = -20.0
IY = 0.0

i_x_acc = 0.0 #accumulated i_x value
i_y_acc = 0.0 #accumulated i_y value


time_since_last_detection = 0 #frames

def PID_reset():
	print("PID resetting, .....")
	global i_x_acc, i_y_acc
	i_x_acc = 0.0
	i_y_acc = 0.0
	vel_x(0.0)
	vel_y(0.0)

def PID(x_norm, y_norm, delta_time, detection):
	global i_y_acc, i_x_acc
	global time_since_last_detection
	if not detection:
		time_since_last_detection += 1
		if time_since_last_detection > 60 and abs(i_x_acc) > 0 and abs(i_y_acc) > 0:
			PID_reset()
			time_since_last_detection = 0
	else:
		time_since_last_detection = 0
		x_diff = (x_norm - 0.5)
		y_diff = (y_norm - 0.5)
		vel_x(IX*i_x_acc + PX*x_diff)
		vel_y(IY*i_y_acc + PY*y_diff)
		i_x_acc += (delta_time * x_diff)
		i_y_acc += (delta_time * y_diff)