#Stepper motor controller library
#Written by: Abdullah Alwakeel
#Date: Apr 12 2025

import Jetson.GPIO as gpio

gpio.setmode(gpio.BOARD)


dir_pin = 33
step_pwm_pin = 15
sleep_pin = 32 #TODO find a good pin

gpio.setup(step_pwm_pin, gpio.OUT, initial=gpio.LOW)
gpio.setup(dir_pin, gpio.OUT, initial=gpio.LOW)
gpio.setup(sleep_pin, gpio.OUT, initial=gpio.LOW)


step = gpio.PWM(step_pwm_pin, 200) #freq is irrelevant here, since we change it to determine stepping speed
dir_ = gpio.PWM(dir_pin, 10)
sleep_ = gpio.PWM(sleep_pin, 10) #comment these out if PWM is not needed

dir_.start(100) #duty cycle (100% = always on)
sleep_.start(100)
step.start(0)

num_steps = 210 # steps per revolution
microsteps = 32 #microstep setting (fixed with soldering)

def calc_pwm_freq(hz): #hz: motor revolutions per second
	return hz * num_steps * microsteps

def set_dir(d):
	if d:
		dir_.ChangeDutyCycle(100)
	else:
		dir_.ChangeDutyCycle(0)


#one step per second = minimal speed
def set_speed(v):
	if v < (1.0 / num_steps): #minimal speed (Hz)
		step.ChangeDutyCycle(0)
		sleep_.ChangeDutyCycle(0) #sleep is active low (0 = sleeping, 1 = not sleeping)
	else:
		sleep_.ChangeDutyCycle(100)
		step.ChangeFrequency(calc_pwm_freq(v))
		step.ChangeDutyCycle(50)

def stepper_shutdown():
	sleep_.ChangeDutyCycle(0)


def vel_x(v):
	if v > 0:
		set_dir(True)
		set_speed(v)
	else:
		set_dir(False)
		set_speed(-v)



if __name__ == "__main__":
	while True:
		try:
			inp = input("Enter speed in Hz (float) (non-numeric value = exit): ")
			i = float(inp)
			vel_x(i)
		except:
			break