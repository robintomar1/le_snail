// RP2040 Arduino core + Adafruit BNO08x library and its declared dependencies.
// Wire the breakout using its voltage/pin documentation. Default Wire pins are board-specific.
#include <Adafruit_BNO08x.h>
#include <Wire.h>

Adafruit_BNO08x imu(-1);
sh2_SensorValue_t event;
float quaternion[4], gyro[3];
uint64_t quaternionTime = 0, gyroTime = 0, lastSent = 0;
uint32_t sequence = 0;
bool haveQuaternion = false, haveGyro = false;

uint64_t micros64() {
  static uint32_t previous = 0;
  static uint64_t upper = 0;
  uint32_t now = micros();
  if (now < previous) upper += (1ULL << 32);
  previous = now;
  return upper + now;
}

void reports() {
  if (!imu.enableReport(SH2_GAME_ROTATION_VECTOR, 10000) ||
      !imu.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000)) {
    while (true) delay(100);  // Host freshness check stops control if reports cannot start.
  }
}

void setup() {
  Serial.begin(460800);
  Wire.begin();
  if (!imu.begin_I2C()) while (true) delay(100);
  reports();
}

void loop() {
  uint64_t now = micros64();
  if (imu.wasReset()) {
    // Let host detect discontinuity and require a fresh controller start.
    sequence = 0;
    haveQuaternion = haveGyro = false;
    reports();
  }
  if (!imu.getSensorEvent(&event)) return;
  if (event.sensorId == SH2_GAME_ROTATION_VECTOR) {
    quaternion[0] = event.un.gameRotationVector.real;
    quaternion[1] = event.un.gameRotationVector.i;
    quaternion[2] = event.un.gameRotationVector.j;
    quaternion[3] = event.un.gameRotationVector.k;
    quaternionTime = now;
    haveQuaternion = true;
  } else if (event.sensorId == SH2_GYROSCOPE_CALIBRATED) {
    gyro[0] = event.un.gyroscope.x;
    gyro[1] = event.un.gyroscope.y;
    gyro[2] = event.un.gyroscope.z;
    gyroTime = now;
    haveGyro = true;
  }
  if (!haveQuaternion || !haveGyro || now - quaternionTime > 20000 || now - gyroTime > 20000 || now - lastSent < 10000) return;
  uint64_t sampleTime = quaternionTime < gyroTime ? quaternionTime : gyroTime;
  char timestamp[24];
  snprintf(timestamp, sizeof(timestamp), "%llu", (unsigned long long)sampleTime);
  Serial.print("{\"seq\":"); Serial.print(sequence++);
  Serial.print(",\"sample_us\":"); Serial.print(timestamp);
  Serial.print(",\"q_wxyz\":[");
  for (int i=0; i<4; ++i) { if (i) Serial.print(','); Serial.print(quaternion[i], 7); }
  Serial.print("],\"gyro_rad_s\":[");
  for (int i=0; i<3; ++i) { if (i) Serial.print(','); Serial.print(gyro[i], 7); }
  Serial.println("]}");
  lastSent = now;
  haveQuaternion = haveGyro = false;
}
