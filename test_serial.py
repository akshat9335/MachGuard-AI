import serial

arduino = serial.Serial('COM5',9600)

while True:
    line = arduino.readline().decode().strip()
    print(line) 