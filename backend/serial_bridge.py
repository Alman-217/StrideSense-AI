"""Read the ESP32 CSV stream and post batches to the local StrideSense API."""
from __future__ import annotations

import argparse
import time
import requests
import serial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM6", help="ESP32 serial port, e.g. COM6")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--api", default="http://127.0.0.1:8000/api/imu/batch")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    batch = []
    print(f"Connecting to ESP32 on {args.port} at {args.baud} baud...")
    try:
        with serial.Serial(args.port, args.baud, timeout=2) as ser:
            time.sleep(2)  # ESP32 boards may reset when serial opens.
            ser.reset_input_buffer()
            print("ESP32 connected. Streaming sensor batches to StrideSense.")
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith(("STRIDESENSE", "timestamp_us", "SENSOR", "ERROR")):
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
                        "gz_dps": float(parts[6]),
                    }
                except ValueError:
                    continue
                batch.append(sample)
                if len(batch) >= args.batch_size:
                    try:
                        response = requests.post(args.api, json={"samples": batch}, timeout=5)
                        response.raise_for_status()
                        result = response.json()
                        if "research_model_score" in result:
                            print(f"Received {len(batch)} samples | uncalibrated research score: {result['research_model_score']:.4f}")
                        else:
                            print(f"Received {len(batch)} samples | buffer: {result.get('buffer_size')}")
                    except requests.RequestException as exc:
                        print(f"API send failed: {exc}")
                    batch.clear()
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
        print("Check the COM port and close Arduino Serial Monitor.")
    except KeyboardInterrupt:
        print("\nSerial bridge stopped.")


if __name__ == "__main__":
    main()
