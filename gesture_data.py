import argparse
import math
import struct
import time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import serial
from serial.tools import list_ports

# Optional keyboard control
try:
    from pynput.keyboard import Controller as KeyboardController, Key
    HAVE_PYNPUT = True
except Exception:
    HAVE_PYNPUT = False


MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
FRAME_HDR_LENGTH = 40
TLV_HDR_SZ = 8

# If you later find the exact numeric TLV type IDs from TI source,
# set them here. Leave as None for auto-detect.
FEATURE_TLV_TYPE: Optional[int] = None
PROB_TLV_TYPE: Optional[int] = None

GESTURE_LABELS = [
    "NO_GESTURE",
    "L2R",
    "R2L",
    "U2D",
    "D2U",
    "CW",
    "CCW",
    "OFF",
    "ON",
    "SHINE",
]

GESTURE_TO_ACTION = {
    "L2R": "next_slide",
    "R2L": "prev_slide",
    "ON": "select",
    "OFF": "pause",
    "SHINE": "mode_toggle",
    "U2D": None,
    "D2U": None,
    "CW": None,
    "CCW": None,
    "NO_GESTURE": None,
}


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


def read_packet(data: serial.Serial, buf: bytearray, timeout_s: float = 2.0) -> Optional[bytes]:
    t0 = time.monotonic()

    while True:
        chunk = data.read(4096)
        if chunk:
            buf += chunk

        i = buf.find(MAGIC)

        if i < 0:
            if len(buf) > len(MAGIC):
                del buf[:-len(MAGIC)]
            if time.monotonic() - t0 > timeout_s:
                return None
            continue

        if i > 0:
            del buf[:i]
            continue

        if len(buf) < FRAME_HDR_LENGTH:
            if time.monotonic() - t0 > timeout_s:
                return None
            continue

        packet_len = struct.unpack_from("<I", buf, 12)[0]

        if packet_len < FRAME_HDR_LENGTH or packet_len > 65535:
            del buf[:len(MAGIC)]
            continue

        if len(buf) < packet_len:
            if time.monotonic() - t0 > timeout_s:
                return None
            continue

        packet = bytes(buf[:packet_len])
        del buf[:packet_len]
        return packet


def packet_parser(packet: bytes):
    version, packet_len, platform, frame_num, time_cpu_cycles, num_detect_obj, num_tlvs, subframe = \
        struct.unpack_from("<8I", packet, 8)

    header = {
        "version": version,
        "packet_len": packet_len,
        "platform": platform,
        "frame_num": frame_num,
        "time_cpu_cycles": time_cpu_cycles,
        "num_detect_obj": num_detect_obj,
        "num_tlvs": num_tlvs,
        "subframe": subframe,
    }

    tlvs = []
    off = FRAME_HDR_LENGTH

    for _ in range(num_tlvs):
        if off + TLV_HDR_SZ > len(packet):
            break

        tlv_type, tlv_len = struct.unpack_from("<2I", packet, off)
        off += TLV_HDR_SZ
        remain = len(packet) - off

        # Prefer TI SDK convention: tlv_len = payload length
        if tlv_len <= remain:
            payload_len = tlv_len
        # Fallback if a build uses total TLV length including the 8-byte header
        elif tlv_len >= TLV_HDR_SZ and (tlv_len - TLV_HDR_SZ) <= remain:
            payload_len = tlv_len - TLV_HDR_SZ
        else:
            break

        payload = packet[off:off + payload_len]
        off += payload_len
        tlvs.append((tlv_type, payload))

    return header, tlvs


@dataclass
class GestureFeatures:
    weighted_doppler: float
    weighted_positive_doppler: float
    weighted_negative_doppler: float
    weighted_range: float
    num_points_over_threshold: float
    weighted_azimuth_mean: float
    weighted_elevation_mean: float
    azimuth_doppler_correlation: float
    weighted_azimuth_dispersion: float
    weighted_elevation_dispersion: float


@dataclass
class GestureProbabilities:
    probs: Dict[str, float]

    @property
    def best_label(self) -> str:
        return max(self.probs, key=self.probs.get)

    @property
    def best_prob(self) -> float:
        return self.probs[self.best_label]


def decode_features_tlv(payload: bytes) -> Optional[GestureFeatures]:
    if len(payload) != 40:
        return None
    vals = struct.unpack("<10f", payload)
    return GestureFeatures(*vals)


def decode_probabilities_tlv(payload: bytes) -> Optional[GestureProbabilities]:
    if len(payload) != 40:
        return None

    vals = struct.unpack("<10f", payload)

    if not all(math.isfinite(v) for v in vals):
        return None
    if not all(-1e-3 <= v <= 1.001 for v in vals):
        return None

    s = sum(vals)
    if not (0.70 <= s <= 1.30):
        return None

    probs = {label: float(v) for label, v in zip(GESTURE_LABELS, vals)}
    return GestureProbabilities(probs)


def parse_gesture_tlvs(tlvs: List[Tuple[int, bytes]]) -> Tuple[Optional[GestureFeatures], Optional[GestureProbabilities]]:
    features = None
    probs = None

    # Exact type match first if you later fill them in
    for tlv_type, payload in tlvs:
        if FEATURE_TLV_TYPE is not None and tlv_type == FEATURE_TLV_TYPE:
            maybe = decode_features_tlv(payload)
            if maybe is not None:
                features = maybe

        if PROB_TLV_TYPE is not None and tlv_type == PROB_TLV_TYPE:
            maybe = decode_probabilities_tlv(payload)
            if maybe is not None:
                probs = maybe

    if features is not None or probs is not None:
        return features, probs

    # Auto-detect by payload shape
    for _, payload in tlvs:
        if len(payload) != 40:
            continue

        maybe_probs = decode_probabilities_tlv(payload)
        if maybe_probs is not None and probs is None:
            probs = maybe_probs
            continue

        maybe_feat = decode_features_tlv(payload)
        if maybe_feat is not None and features is None:
            features = maybe_feat

    return features, probs


class PowerPointGestureController:
    def __init__(self, min_prob: float = 0.85, stable_frames: int = 2, cooldown_s: float = 0.9):
        self.min_prob = float(min_prob)
        self.stable_frames = int(stable_frames)
        self.cooldown_s = float(cooldown_s)

        self.kb = KeyboardController() if HAVE_PYNPUT else None
        self.mode = 0
        self.last_fired_t = 0.0
        self.last_candidate = None
        self.candidate_count = 0

    def _press(self, key):
        if self.kb is None:
            return
        self.kb.press(key)
        self.kb.release(key)

    def _do_action(self, action: str):
        if action == "next_slide":
            self._press(Key.right)
            print("[action] next_slide")
        elif action == "prev_slide":
            self._press(Key.left)
            print("[action] prev_slide")
        elif action == "select":
            self._press(Key.enter)
            print("[action] select")
        elif action == "pause":
            self._press("b")
            print("[action] pause/black_screen")
        elif action == "mode_toggle":
            self.mode ^= 1
            print(f"[action] mode_toggle -> mode={self.mode}")

    def update(self, label: str, prob: float):
        now = time.monotonic()

        if label == "NO_GESTURE" or prob < self.min_prob:
            self.last_candidate = None
            self.candidate_count = 0
            return

        if self.last_candidate == label:
            self.candidate_count += 1
        else:
            self.last_candidate = label
            self.candidate_count = 1

        if self.candidate_count < self.stable_frames:
            return

        if (now - self.last_fired_t) < self.cooldown_s:
            return

        action = GESTURE_TO_ACTION.get(label)
        if action:
            self._do_action(action)
            self.last_fired_t = now

        self.last_candidate = None
        self.candidate_count = 0


def main():
    ap = argparse.ArgumentParser(description="TI Gesture Demo - Data UART parser")
    ap.add_argument("--port", default=None, help="Data UART COM port, e.g. COM6")
    ap.add_argument("--baud", type=int, default=921600, help="Data UART baud")
    ap.add_argument("--debug", action="store_true", help="Print frame/TLV diagnostics")
    ap.add_argument("--actions", action="store_true", help="Enable keyboard actions")
    ap.add_argument("--min-prob", type=float, default=0.85)
    ap.add_argument("--stable-frames", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=0.9)
    ap.add_argument("--list", action="store_true", help="List serial ports and exit")
    args = ap.parse_args()

    if args.list:
        list_all_ports()
        return

    port = args.port or find_port("bridge: standard")
    if not port:
        raise SystemExit("Could not find Data UART. Use --list and then pass --port COMx")

    print(f"[data-uart] opening {port} @ {args.baud}")
    ser = serial.Serial(port, args.baud, timeout=0.05)
    buf = bytearray()

    controller = PowerPointGestureController(
        min_prob=args.min_prob,
        stable_frames=args.stable_frames,
        cooldown_s=args.cooldown,
    )

    last_frame_key = None

    try:
        while True:
            packet = read_packet(ser, buf, timeout_s=2.0)
            if packet is None:
                print("[debug] no valid packets on Data UART")
                continue

            header, tlvs = packet_parser(packet)
            frame_key = (header["frame_num"], header["subframe"])

            if frame_key == last_frame_key:
                continue
            last_frame_key = frame_key

            if args.debug:
                print(f"[frame {header['frame_num']}] tlvs={[t for t, _ in tlvs]}")

            _, probs = parse_gesture_tlvs(tlvs)
            if probs is None:
                continue

            best_label = probs.best_label
            best_prob = probs.best_prob

            ordered = ", ".join(f"{k}={v:.3f}" for k, v in probs.probs.items())
            print(f"[frame {header['frame_num']}] best={best_label} ({best_prob:.3f}) | {ordered}")

            if args.actions:
                controller.update(best_label, best_prob)

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\n[data-uart] closed")


if __name__ == "__main__":
    main()
