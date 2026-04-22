import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("Error: The GPIO library is not installed!")
    print("Please run: pip install rpi-lgpio")
    exit(1)

# BCM Pin 17 (Physical Pin 11)
BUZZER_PIN = 17

def test_buzzer():
    print("Initializing GPIO...")
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Set the buzzer pin as an output pin
    GPIO.setup(BUZZER_PIN, GPIO.OUT)

    print("Beeping 3 times... Listen closely!")
    try:
        for i in range(3):
            print(f"Beep {i+1}: ON")
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.5)  # Wait half a second
            
            print(f"Beep {i+1}: OFF")
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(0.5)  # Wait half a second
            
        print("Buzzer test complete! It works!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        # Always clean up GPIO pins to prevent damage or locked pins
        GPIO.cleanup()
        print("GPIO cleaned up.")

if __name__ == "__main__":
    test_buzzer()
