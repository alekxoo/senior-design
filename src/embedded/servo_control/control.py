import Jetson.GPIO as GPIO
from time import sleep
from typing import Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

class Motor(ABC):
    """Abstract base class for motor control"""
    
    def __init__(self, min_position: float, max_position: float):
        self.min_position = min_position
        self.max_position = max_position
        self.current_position = min_position
    
    @abstractmethod
    def move_to_position(self, position: float) -> None:
        """Move to absolute position"""
        pass
    
    def get_current_position(self) -> float:
        """Get current position"""
        return self.current_position
    
    def move_to_minimum(self) -> None:
        """Move to minimum allowed position"""
        self.move_to_position(self.min_position)
    
    def move_to_maximum(self) -> None:
        """Move to maximum allowed position"""
        self.move_to_position(self.max_position)
    
    def increase_position(self, amount: float) -> None:
        """Increase position by specified amount"""
        new_pos = min(self.current_position + amount, self.max_position)
        self.move_to_position(new_pos)
    
    def decrease_position(self, amount: float) -> None:
        """Decrease position by specified amount"""
        new_pos = max(self.current_position - amount, self.min_position)
        self.move_to_position(new_pos)
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass

class ServoMG92B(Motor):
    """MG92B Servo motor controller"""
    
    def __init__(self, pin: int, min_angle: float = 0, max_angle: float = 180):
        super().__init__(min_angle, max_angle)
        self.pin = pin
        
        # Initialize GPIO for servo
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, 50)  # 50Hz frequency
        self.pwm.start(0)
    
    def _angle_to_duty_cycle(self, angle: float) -> float:
        """Convert angle to PWM duty cycle"""
        pulse_width = ((angle / 180.0) * 2000) + 500  # 500-2500 microseconds
        duty_cycle = pulse_width / 20000 * 100  # Convert to percentage
        return duty_cycle
    
    def move_to_position(self, position: float) -> None:
        """Move servo to specified angle"""
        position = max(self.min_position, min(position, self.max_position))
        duty_cycle = self._angle_to_duty_cycle(position)
        
        self.pwm.ChangeDutyCycle(duty_cycle)
        self.current_position = position
        sleep(0.3)  # Allow servo to reach position
    
    def cleanup(self) -> None:
        """Stop PWM and cleanup GPIO"""
        self.pwm.stop()
        GPIO.cleanup(self.pin)

class StepperDRV8834(Motor):
    """DRV8834 stepper motor controller"""
    
    def __init__(self, 
                 step_pin: int, 
                 dir_pin: int, 
                 enable_pin: int,
                 microsteps: int = 32,
                 steps_per_rev: int = 200,
                 min_angle: float = -180,
                 max_angle: float = 180):
        super().__init__(min_angle, max_angle)
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.enable_pin = enable_pin
        self.microsteps = microsteps
        self.steps_per_rev = steps_per_rev
        self.current_steps = 0
        self.total_steps = steps_per_rev * microsteps
        
        # Initialize GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.step_pin, GPIO.OUT)
        GPIO.setup(self.dir_pin, GPIO.OUT)
        GPIO.setup(self.enable_pin, GPIO.OUT)
        GPIO.output(self.enable_pin, GPIO.LOW)  # Enable driver
    
    def _step(self, direction: bool) -> None:
        """Perform one step in specified direction"""
        GPIO.output(self.dir_pin, GPIO.HIGH if direction else GPIO.LOW)
        GPIO.output(self.step_pin, GPIO.HIGH)
        sleep(0.000001)  # 1 microsecond pulse
        GPIO.output(self.step_pin, GPIO.LOW)
        sleep(0.001)  # Delay between steps
    
    def move_to_position(self, position: float) -> None:
        """Move to specified angle"""
        position = max(self.min_position, min(position, self.max_position))
        target_steps = int((position / 360.0) * self.total_steps)
        steps_to_move = target_steps - self.current_steps
        
        direction = steps_to_move > 0
        for _ in range(abs(steps_to_move)):
            self._step(direction)
            self.current_steps += 1 if direction else -1
        
        self.current_position = position
    
    def cleanup(self) -> None:
        """Disable driver and cleanup GPIO"""
        GPIO.output(self.enable_pin, GPIO.HIGH)  # Disable driver
        GPIO.cleanup([self.step_pin, self.dir_pin, self.enable_pin])

@dataclass
class Position:
    """Camera position data"""
    pan: float
    tilt: float

class CameraModule:
    """Camera pan/tilt control module"""
    
    def __init__(self, pan_motor: Motor, tilt_motor: Motor):
        self.pan_motor = pan_motor
        self.tilt_motor = tilt_motor
    
    def pan_to(self, angle: float) -> None:
        """Pan camera to specified angle"""
        self.pan_motor.move_to_position(angle)
    
    def tilt_to(self, angle: float) -> None:
        """Tilt camera to specified angle"""
        self.tilt_motor.move_to_position(angle)
    
    def pan_to_minimum(self) -> None:
        """Pan to minimum position"""
        self.pan_motor.move_to_minimum()
    
    def pan_to_maximum(self) -> None:
        """Pan to maximum position"""
        self.pan_motor.move_to_maximum()
    
    def tilt_to_minimum(self) -> None:
        """Tilt to minimum position"""
        self.tilt_motor.move_to_minimum()
    
    def tilt_to_maximum(self) -> None:
        """Tilt to maximum position"""
        self.tilt_motor.move_to_maximum()
    
    def increase_pan(self, amount: float) -> None:
        """Increase pan by specified amount"""
        self.pan_motor.increase_position(amount)
    
    def decrease_pan(self, amount: float) -> None:
        """Decrease pan by specified amount"""
        self.pan_motor.decrease_position(amount)
    
    def increase_tilt(self, amount: float) -> None:
        """Increase tilt by specified amount"""
        self.tilt_motor.increase_position(amount)
    
    def decrease_tilt(self, amount: float) -> None:
        """Decrease tilt by specified amount"""
        self.tilt_motor.decrease_position(amount)
    
    def get_position(self) -> Position:
        """Get current camera position"""
        return Position(
            pan=self.pan_motor.get_current_position(),
            tilt=self.tilt_motor.get_current_position()
        )
    
    def cleanup(self) -> None:
        """Cleanup all resources"""
        self.pan_motor.cleanup()
        self.tilt_motor.cleanup()

def main():
    """Example usage of camera control system"""
    try:
        # Initialize motors
        pan_motor = StepperDRV8834(
            step_pin=23,
            dir_pin=24,
            enable_pin=25,
            microsteps=32,
            steps_per_rev=200,
            min_angle=-180,
            max_angle=180
        )
        
        tilt_motor = ServoMG92B(
            pin=18,
            min_angle=0,
            max_angle=180
        )
        
        # Create camera module
        camera = CameraModule(pan_motor, tilt_motor)
        
        # Example movement sequence
        camera.pan_to_minimum()  # Move to -180 degrees
        sleep(1)
        
        camera.increase_pan(45)  # Pan right 45 degrees
        sleep(1)
        
        camera.tilt_to_maximum()  # Move to 180 degrees
        sleep(1)
        
        camera.decrease_tilt(30)  # Tilt down 30 degrees
        
        # Get and print current position
        pos = camera.get_position()
        print(f"Current position: Pan={pos.pan:.1f}°, Tilt={pos.tilt:.1f}°")
        
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        camera.cleanup()

if __name__ == "__main__":
    main()