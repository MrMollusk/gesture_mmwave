import argparse
import serial
from serial.tools import list_ports
import time


def find_port(must_contain: str):
    for p in list_ports.comports():
        desc = (p.description or "").lower()
        if must_contain.lower() in desc:
            return p.device
    return None


def list_all_ports():
    print("Available serial ports:")
    for p in list_ports.comports():
        print(f"  {p.device:10s}  {p.description}")


def main():
    ap = argparse.ArgumentParser(description="TI Gesture Demo - User UART monitor")
    ap.add_argument("--port", default=None, help="User UART COM port, e.g. COM5")
    ap.add_argument("--baud", type=int, default=115200, help="User UART baud")
    ap.add_argument("--auto-enter", action="store_true", help="Send newline after opening")
    ap.add_argument("--list", action="store_true", help="List serial ports and exit")
    args = ap.parse_args()

    if args.list:
        list_all_ports()
        return

    port = args.port or find_port("bridge: enhanced")
    if not port:
        raise SystemExit("Could not find User UART. Use --list and then pass --port COMx")

    print(f"[user-uart] opening {port} @ {args.baud}")
    ser = serial.Serial(port, args.baud, timeout=0.1)

    time.sleep(0.5)

    if args.auto_enter:
        ser.write(b"\r\n")
        ser.flush()
        print("[user-uart] sent startup newline")

    print("[user-uart] reading... Ctrl+C to stop")
    try:
        while True:
            chunk = ser.read(4096)
            if chunk:
                try:
                    text = chunk.decode("utf-8", errors="replace")
                except Exception:
                    text = repr(chunk)
                print(text, end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\n[user-uart] closed")


if __name__ == "__main__":
    main()
