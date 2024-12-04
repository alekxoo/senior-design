// motor_control.hpp
#pragma once
#include <chrono>
#include <thread>
#include <memory>
#include <JetsonGPIO.h>

class Motor {
protected:
    const float minPosition;
    const float maxPosition;
    float currentPosition;

public:
    Motor(float min, float max) : minPosition(min), maxPosition(max), currentPosition(min) {}
    virtual ~Motor() = default;

    virtual void moveToPosition(float position) = 0;
    virtual float getCurrentPosition() const { return currentPosition; }
    
    void moveToMinimum() { moveToPosition(minPosition); }
    void moveToMaximum() { moveToPosition(maxPosition); }
    
    void increasePosition(float amount) {
        float newPos = std::min(currentPosition + amount, maxPosition);
        moveToPosition(newPos);
    }
    
    void decreasePosition(float amount) {
        float newPos = std::max(currentPosition - amount, minPosition);
        moveToPosition(newPos);
    }
    
    virtual void cleanup() = 0;
};

class ServoMG92B : public Motor {
private:
    const int pwmPin;
    const int pwmFreq = 50;  // 50Hz for servo
    std::unique_ptr<GPIO::PWM> pwm;

public:
    ServoMG92B(int pin, float minAngle = 0, float maxAngle = 180)
        : Motor(minAngle, maxAngle), pwmPin(pin) {
        GPIO::setmode(GPIO::BOARD);
        pwm = std::make_unique<GPIO::PWM>(pwmPin, pwmFreq);
        pwm->start(0);
    }

    void moveToPosition(float position) override {
        position = std::clamp(position, minPosition, maxPosition);
        float pulseWidth = ((position / 180.0f) * 2000.0f) + 500.0f; // 500-2500us
        float dutyCycle = pulseWidth / 20000.0f * 100.0f;
        
        pwm->ChangeDutyCycle(dutyCycle);
        currentPosition = position;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    void cleanup() override {
        pwm->stop();
        GPIO::cleanup();
    }
};

class StepperDRV8834 : public Motor {
private:
    const int stepPin;
    const int dirPin;
    const int enablePin;
    const int microSteps;
    const int stepsPerRev;
    int currentSteps;
    const int totalSteps;

    void step(bool direction) {
        GPIO::output(dirPin, direction);
        GPIO::output(stepPin, GPIO::HIGH);
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        GPIO::output(stepPin, GPIO::LOW);
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

public:
    StepperDRV8834(int stepP, int dirP, int enableP, 
                   int microS = 32, int stepsRev = 200,
                   float minAngle = -180, float maxAngle = 180)
        : Motor(minAngle, maxAngle),
          stepPin(stepP), dirPin(dirP), enablePin(enableP),
          microSteps(microS), stepsPerRev(stepsRev),
          currentSteps(0), totalSteps(stepsRev * microS) {
        
        GPIO::setmode(GPIO::BOARD);
        GPIO::setup(stepPin, GPIO::OUT);
        GPIO::setup(dirPin, GPIO::OUT);
        GPIO::setup(enablePin, GPIO::OUT);
        GPIO::output(enablePin, GPIO::LOW);  // Enable driver
    }

    void moveToPosition(float position) override {
        position = std::clamp(position, minPosition, maxPosition);
        int targetSteps = static_cast<int>((position / 360.0f) * totalSteps);
        int stepsToMove = targetSteps - currentSteps;
        
        bool direction = stepsToMove > 0;
        for (int i = 0; i < std::abs(stepsToMove); ++i) {
            step(direction);
            currentSteps += direction ? 1 : -1;
        }
        
        currentPosition = position;
    }

    void cleanup() override {
        GPIO::output(enablePin, GPIO::HIGH);  // Disable driver
        GPIO::cleanup();
    }
};

class CameraModule {
private:
    std::unique_ptr<Motor> panMotor;
    std::unique_ptr<Motor> tiltMotor;

public:
    CameraModule(std::unique_ptr<Motor> pan, std::unique_ptr<Motor> tilt)
        : panMotor(std::move(pan)), tiltMotor(std::move(tilt)) {}

    void panTo(float angle) { panMotor->moveToPosition(angle); }
    void tiltTo(float angle) { tiltMotor->moveToPosition(angle); }
    
    void panToMinimum() { panMotor->moveToMinimum(); }
    void panToMaximum() { panMotor->moveToMaximum(); }
    void tiltToMinimum() { tiltMotor->moveToMinimum(); }
    void tiltToMaximum() { tiltMotor->moveToMaximum(); }
    
    void increasePan(float amount) { panMotor->increasePosition(amount); }
    void decreasePan(float amount) { panMotor->decreasePosition(amount); }
    void increaseTilt(float amount) { tiltMotor->increasePosition(amount); }
    void decreaseTilt(float amount) { tiltMotor->decreasePosition(amount); }
    
    std::pair<float, float> getPosition() const {
        return {panMotor->getCurrentPosition(), tiltMotor->getCurrentPosition()};
    }
    
    void cleanup() {
        if (panMotor) panMotor->cleanup();
        if (tiltMotor) tiltMotor->cleanup();
    }
};

// main.cpp
#include <iostream>
#include <memory>
#include "motor_control.hpp"

int main() {
    try {
        // Initialize motors
        auto panMotor = std::make_unique<StepperDRV8834>(
            23,     // step pin
            24,     // dir pin
            25,     // enable pin
            32,     // microsteps
            200,    // steps per revolution
            -180,   // min angle
            180     // max angle
        );
        
        auto tiltMotor = std::make_unique<ServoMG92B>(
            18,     // PWM pin
            0,      // min angle
            180     // max angle
        );
        
        // Create camera module
        CameraModule camera(std::move(panMotor), std::move(tiltMotor));
        
        // Example usage
        camera.panToMinimum();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        camera.increasePan(45);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        camera.tiltToMaximum();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        camera.decreaseTilt(30);
        
        auto [pan, tilt] = camera.getPosition();
        std::cout << "Current position: Pan=" << pan 
                  << "°, Tilt=" << tilt << "°\n";
        
        camera.cleanup();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}