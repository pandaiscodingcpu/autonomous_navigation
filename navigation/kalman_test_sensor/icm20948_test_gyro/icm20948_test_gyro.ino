#include <Wire.h>
#include <ICM_20948.h> // SparkFun ICM-20948 Library

#define AD0_VAL 1 // 0 if AD0 -> GND (0x68), 1 if AD0 -> 3.3V (0x69)

ICM_20948_I2C myICM;

// ==========================================
// 1D KALMAN FILTER CLASS
// ==========================================
class KalmanFilter1D {
  private:
    float q; // Process noise covariance
    float r; // Measurement noise covariance
    float p; // Estimation error covariance
    float x; // State estimate

  public:
    KalmanFilter1D(float process_noise = 0.01, float measurement_noise = 2.0, float est_error = 1.0, float init_val = 0.0) {
      q = process_noise;
      r = measurement_noise;
      p = est_error;
      x = init_val;
    }

    float update(float measurement) {
      p = p + q;
      float k = p / (p + r);
      x = x + k * (measurement - x);
      p = (1.0f - k) * p;
      return x;
    }

    float getState() { return x; }
};

// Filters tuned for 100 Hz sampling rate
KalmanFilter1D kfGyrX(0.01, 10.0);
KalmanFilter1D kfAccZ(0.01, 100.0);

// Latest raw values stored during 100Hz sampling
float rawGyrX = 0.0;
float rawAccZ = 0.0;

// Timing variables
unsigned long lastSampleTime = 0;
unsigned long lastPrintTime = 0;

const unsigned long SAMPLE_INTERVAL = 10;   // 10 ms = 100 Hz sampling
const unsigned long PRINT_INTERVAL  = 1000;  // 1000 ms = 1 Hz output

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Wire.begin();
  Wire.setClock(400000); // 400kHz fast I2C bus

  bool initialized = false;
  while (!initialized) {
    myICM.begin(Wire, AD0_VAL);

    if (myICM.status != ICM_20948_Stat_Ok) {
      Serial.print(F("ICM-20948 initialization failed! Status: "));
      Serial.println(myICM.statusString(myICM.status));
      delay(1000);
    } else {
      initialized = true;
    }
  }

  Serial.println(F("ICM-20948 Ready: 100Hz Kalman Filtering, 1Hz Serial Stream"));
}

void loop() {
  unsigned long currentMillis = millis();

  // 1. SAMPLE & UPDATE KALMAN FILTER AT 100 Hz (Every 10 ms)
  if (currentMillis - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currentMillis;

    if (myICM.dataReady()) {
      myICM.getAGMT();

      rawGyrX = myICM.gyrX();
      rawAccZ = myICM.accZ();

      // Update Kalman filter state continuously
      kfGyrX.update(rawGyrX);
      kfAccZ.update(rawAccZ);
    }
  }

  // 2. TRANSMIT RAW + FILTERED DATA AT 1 Hz (Every 1 second)
  if (currentMillis - lastPrintTime >= PRINT_INTERVAL) {
    lastPrintTime = currentMillis;

    // Send both RAW and FILTERED pairs for comparison
    Serial.print("RawGyrY:");   Serial.print(rawGyrY);
    Serial.print(",FiltGyrX:"); Serial.print(kfGyrX.getState());
    Serial.print(",RawAccZ:");   Serial.print(rawAccZ);
    Serial.print(",FiltAccZ:"); Serial.println(kfAccZ.getState());
  }
}