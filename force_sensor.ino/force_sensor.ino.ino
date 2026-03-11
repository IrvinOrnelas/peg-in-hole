const int PIN_SENSOR  = 2;
const int MUESTRAS    = 20;
const int DURACION_MS = 2000;

float valorBase = 0.0;
float valorMax  = 0.0;

float promediarLecturas() {
  long suma = 0;
  for (int i = 0; i < MUESTRAS; i++) {
    suma += analogRead(PIN_SENSOR);
    delay(DURACION_MS / MUESTRAS);
  }
  return suma / (float)MUESTRAS;
}

void setup() {
  Serial.begin(115200);

  Serial.println("Calibrando valor base (sin fuerza)...");
  valorBase = promediarLecturas();
  Serial.print("Valor base: ");
  Serial.println(valorBase);

  Serial.println("Aplica la FUERZA MAXIMA ahora...");
  delay(DURACION_MS);

  Serial.println("Midiendo fuerza máxima...");
  valorMax = promediarLecturas();
  Serial.print("Valor máximo: ");
  Serial.println(valorMax);

  Serial.println("Calibración completa. Reportando fuerza...");
}

void loop() {
  int valorSensor = analogRead(PIN_SENSOR);

  float compresion = (valorBase - valorSensor) / (valorBase - valorMax);
  if (compresion < 0.0) compresion = 0.0;

  // Aplicar la ecuación de fuerza: F = 4.82 * x + 0.734
  float fuerza = 4.82 * compresion + 0.734;

  if (fuerza < 0.0) fuerza = 0.0;

  Serial.print("Fuerza medida: ");
  Serial.print(fuerza);
  Serial.print(" N");

  Serial.print("  ");

  Serial.print("Compresion medida: ");
  Serial.println(compresion);

  delay(100);
}