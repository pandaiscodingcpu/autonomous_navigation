#include <Wire.h>
#include <SPI.h>
#include <Adafruit_Sensor.h>
#include "Adafruit_BMP3XX.h"

Adafruit_BMP3XX bmp;
#define SEALEVELPRESSURE_HPA (1013.25)

// The initialization vector Xk [position, velocity]
float Xk[2] = {0.0, 0.0}; 

// Covariance matrix P initialized with 500 on the diagonal
float P[2][2] = {{500.0, 0.0}, 
                 {0.0, 500.0}};

// Static process noise Q with 0.01 on the diagonal
float Q[2][2] = {{0.01, 0.0},
                 {0.0, 0.01}};

// Noise level R (more value = less trust on sensor readings)
float R = 10000.0; 

unsigned long lastTime = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!bmp.begin_I2C()) {
    Serial.println("BMP390 not found.");
    while (1);
  }

  bmp.setTemperatureOversampling(BMP3_OVERSAMPLING_8X);
  bmp.setPressureOversampling(BMP3_OVERSAMPLING_4X);
  bmp.setIIRFilterCoeff(BMP3_IIR_FILTER_COEFF_3);
  bmp.setOutputDataRate(BMP3_ODR_50_HZ);

  // Equivalent to init_state(x0) to establish the initial altitude
  if (bmp.performReading()) {
    Xk[0] = bmp.readAltitude(SEALEVELPRESSURE_HPA);
  }
  lastTime = millis();
}

void loop() {
  if (!bmp.performReading()) {
    return;
  }

  // Calculate self.dT (time step)[cite: 1]
  unsigned long currentTime = millis();
  float dT = (currentTime - lastTime) / 1000.0; 
  lastTime = currentTime;

  // Current measurement z[cite: 1]
  float z = bmp.readAltitude(SEALEVELPRESSURE_HPA);

  // --- predict() logic ---
  // self.Xk = self.A @ self.Xk[cite: 1]
  float Xk_pred0 = Xk[0] + dT * Xk[1];
  float Xk_pred1 = Xk[1];
  
  Xk[0] = Xk_pred0;
  Xk[1] = Xk_pred1;

  // self.P = (self.A @ self.P @ self.A.T) + self.Q[cite: 1]
  float P_pred00 = P[0][0] + dT * P[1][0] + P[0][1] * dT + dT * dT * P[1][1] + Q[0][0];
  float P_pred01 = P[0][1] + dT * P[1][1] + Q[0][1];
  float P_pred10 = P[1][0] + dT * P[1][1] + Q[1][0];
  float P_pred11 = P[1][1] + Q[1][1];

  P[0][0] = P_pred00;
  P[0][1] = P_pred01;
  P[1][0] = P_pred10;
  P[1][1] = P_pred11;

  // --- kalman(z) update logic ---
  // y = z - (self.H @ self.Xk)[cite: 1]
  // Because H is [[1, 0]], H @ Xk is simply Xk[0][cite: 1]
  float y = z - Xk[0];

  // S = (self.H @ self.P @ self.H.T) + self.R[cite: 1]
  // H @ P @ H.T isolates P[0][0][cite: 1]
  float S = P[0][0] + R;

  // K = self.P @ self.H.T @ np.linalg.inv(S)[cite: 1]
  // S is a scalar, so inverse is 1/S[cite: 1]
  float K0 = P[0][0] / S;
  float K1 = P[1][0] / S;

  // self.Xk = self.Xk + (K @ y)[cite: 1]
  Xk[0] = Xk[0] + K0 * y;
  Xk[1] = Xk[1] + K1 * y;

  // self.P = (self.I - K @ self.H) @ self.P[cite: 1]
  // I is an identity matrix eye(2,2)[cite: 1]
  float P_upd00 = (1.0 - K0) * P[0][0];
  float P_upd01 = (1.0 - K0) * P[0][1];
  float P_upd10 = -K1 * P[0][0] + P[1][0];
  float P_upd11 = -K1 * P[0][1] + P[1][1];

  P[0][0] = P_upd00;
  P[0][1] = P_upd01;
  P[1][0] = P_upd10;
  P[1][1] = P_upd11;

  Serial.print(z);
  Serial.print(",");
  Serial.print(Xk[0]);
  Serial.print(",");
  Serial.println(Xk[1]);

  delay(20); 
}