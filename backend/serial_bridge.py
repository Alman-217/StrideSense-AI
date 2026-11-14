
import time
import serial
import requests

SERIAL_PORT = "COM6"

BAUD_RATE = 115200

API_URL = "http://127.0.0.1:8000/api/imu/batch"

# 32 readings at 64 Hz = approximately 0.5 seconds.
BATCH_SIZE = 32

batch = []

print("Starting StrideSense serial bridge...")
print("Connecting to", SERIAL_PORT)

try:

    with serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=2
    ) as ser:

        # Allow the ESP32 to reset after serial connection.
        time.sleep(2)

        ser.reset_input_buffer()

        print("ESP32 connected.")
        print("Streaming sensor readings to FastAPI...")

        while True:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            # Ignore firmware startup messages and CSV header.
            if (
                line.startswith("STRIDESENSE") or
                line.startswith("timestamp_us") or
                line.startswith("ERROR")
            ):
                print(line)
                continue

            parts = line.split(",")

            if len(parts) != 7:
                continue

            try:

                sample = {
                    "timestamp_us": int(parts[0]),
                    "ax_g": float(parts[1]),
                    "ay_g": float(parts[2]),
                    "az_g": float(parts[3]),
                    "gx_dps": float(parts[4]),
                    "gy_dps": float(parts[5]),
                    "gz_dps": float(parts[6])
                }

            except ValueError:
                continue

            batch.append(sample)

            if len(batch) >= BATCH_SIZE:

                try:

                    response = requests.post(
                        API_URL,
                        json={"samples": batch},
                        timeout=5
                    )

                    response.raise_for_status()

                    result = response.json()

                    score = result.get(
                        "research_model_score"
                    )

                    if score is not None:
                        print(
                            "Sensor batch received | "
                            f"Model score: {score:.3f}"
                        )
                    else:
                        print(
                            "Sensor batch received | "
                            f"Buffer: {result.get('buffer_size')}"
                        )

                except requests.RequestException as error:

                    print("Backend communication error:", error)

                batch.clear()

except serial.SerialException as error:

    print("Serial connection error:", error)
    print("Check the ESP32 port and USB connection.")

except KeyboardInterrupt:

    print("\nStrideSense serial bridge stopped.")