#include <Wire.h>
#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>

Adafruit_ICM20948 icm;

void setup() {
  Serial.begin(115200);
  while (!Serial)
    delay(10);

  Serial.println("ICM20948 Gyro Yaw Test");

  // Initialize the sensor on the I2C bus
  if (!icm.begin_I2C()) {
    Serial.println("Failed to find ICM20948 chip. Check wiring or I2C address.");
    while (1) {
      delay(10);
    }
  }
  Serial.println("ICM20948 Found!");

  // Configure the gyro range
  // 500 degrees/second is a good balance of sensitivity and range for a paraglider
  icm.setGyroRange(ICM20948_GYRO_RANGE_500_DPS);
  
  // Set the data rate to match your 50Hz/100Hz EKF loop expectations
  icm.setGyroRateDivisor(10); 
}

void loop() {
  // Create sensor event objects
  sensors_event_t accel, gyro, temp, mag;

  // Fetch the latest readings from the sensor
  icm.getEvent(&accel, &gyro, &temp, &mag);

  // The Adafruit library natively outputs gyro data in radians per second (rad/s).
  // This is the exact unit required by the 'gyro_r' input in your EKF predict() step.
  float gyro_yaw_rate = gyro.gyro.z;

  Serial.print("Yaw_Rate_rad_s:");
  Serial.println(gyro_yaw_rate, 4); // Print with 4 decimal places for precision

  delay(20); // ~50Hz loop
}