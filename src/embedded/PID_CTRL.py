#PID ctrl code

from ServoControl import vel_x

PX = 0.5
IX = 0.5

PY = 0.5
IY = 0.5

i_x_acc = 0.0 #accumulated i_x value
i_y_acc = 0.0 #accumulated i_y value


def PID_reset():
	global i_x_acc, i_y_acc
	i_x_acc = 0.0
	i_y_acc = 0.0

def PID(x_norm, y_norm, delta_time):
	global i_y_acc, i_x_acc
	x_diff = (x_norm - 0.5)
	y_diff = (y_norm - 0.5)
	vel_x(IX*i_x_acc + PX*x_diff)
	vel_y(IY*i_y_acc + PY*y_diff)
	i_x_acc += (delta_time * x_diff)
	i_y_acc += (delta_time * y_diff)