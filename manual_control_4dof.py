"""
Manual Keyboard Controller — 4-DOF Robotic Arm
===============================================
Controls the arm manually using keyboard keys.
Use this FIRST before running the autonomous code to:
    1. Verify each joint moves correctly
    2. Find safe min/max angles for each joint
    3. Find the best home/resting position
    4. Confirm gripper open/close angles

HARDWARE:
    - Raspberry Pi 4B
    - PCA9685 servo driver (I2C)
    - 4x MG996R servos
    - USB camera (eye-in-hand)

JOINTS:
    Channel 0 → Base      (left/right rotation)
    Channel 1 → Shoulder  (up/down)
    Channel 2 → Elbow     (extend/retract)
    Channel 3 → Gripper   (open/close)

INSTALL DEPENDENCIES:
    pip install adafruit-circuitpython-servokit
    pip install adafruit-circuitpython-pca9685
    pip install opencv-python

WIRING CHECK (before running):
    PCA9685 VCC  → Pi 3.3V (pin 1)
    PCA9685 GND  → Pi GND  (pin 6)
    PCA9685 SDA  → Pi SDA  (pin 3)
    PCA9685 SCL  → Pi SCL  (pin 5)
    Servo power  → External 5V/6V battery (NOT from Pi)

ENABLE I2C ON PI (if not done):
    sudo raspi-config → Interface Options → I2C → Enable
    then reboot

RUN:
    python manual_control.py
"""

import cv2
import time
import sys

# ============================================================================
# JOINT CONFIGURATION
# 4-DOF: Base, Shoulder, Elbow, Gripper
# Channels 0-3 on PCA9685
# ============================================================================

JOINTS = {
    'base':     {'channel': 0, 'min': 30, 'max': 120, 'home': 90, 'step': 3},
    'shoulder': {'channel': 1, 'min': 30, 'max': 120, 'home': 90, 'step': 3},
    'elbow':    {'channel': 2, 'min': 30, 'max': 120, 'home': 90, 'step': 3},
    'gripper':  {'channel': 3, 'min': 0,  'max': 90,  'home': 0,  'step': 5},
    # gripper: 0° = fully open, 90° = fully closed
}

# ============================================================================
# KEYBOARD MAPPING
# Each key maps to (joint_name, direction)
# direction: +1 = increase angle, -1 = decrease angle
# ============================================================================

KEY_MAP = {
    ord('w'): ('base',     +1),   # base rotate one way
    ord('s'): ('base',     -1),   # base rotate other way
    ord('a'): ('shoulder', +1),   # shoulder up
    ord('d'): ('shoulder', -1),   # shoulder down
    ord('q'): ('elbow',    +1),   # elbow extend
    ord('e'): ('elbow',    -1),   # elbow retract
    ord('y'): ('gripper',  -1),   # gripper open
    ord('h'): ('gripper',  +1),   # gripper close
}

# Camera
CAMERA_INDEX  = 0      # change to 1 if wrong camera opens
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

# ============================================================================
# SERVO CONTROLLER — uses PCA9685 via adafruit servokit
# ============================================================================

class ServoController:

    def __init__(self):
        try:
            from adafruit_servokit import ServoKit
            self.kit = ServoKit(channels=16)
            self.current_angles = {}

            # Configure all 4 MG996R servos
            for name, cfg in JOINTS.items():
                self.kit.servo[cfg['channel']].actuation_range = 180
                self.kit.servo[cfg['channel']].set_pulse_width_range(500, 2500)

            # Move all joints to home position
            print("[Servos] Moving to home position...")
            for name, cfg in JOINTS.items():
                self._set(name, cfg['home'])
                time.sleep(0.1)

            print("[Servos] All joints at home\n")
            time.sleep(1.0)

        except ImportError:
            print("[Servos] adafruit_servokit not found — SIMULATION MODE")
            print("         Install: pip install adafruit-circuitpython-servokit\n")
            self.kit = None
            self.current_angles = {n: c['home'] for n, c in JOINTS.items()}

        except Exception as e:
            print(f"[Servos] PCA9685 connection failed: {e}")
            print("         Check I2C wiring and that I2C is enabled on Pi")
            print("         Run: sudo raspi-config → Interface Options → I2C → Enable")
            sys.exit(1)

    def _set(self, joint, angle):
        """Directly set a joint to an angle (no smoothing — for manual control)."""
        cfg = JOINTS[joint]
        angle = max(cfg['min'], min(cfg['max'], angle))  # clamp to safe range
        self.current_angles[joint] = angle
        if self.kit:
            self.kit.servo[cfg['channel']].angle = angle

    def move(self, joint, direction):
        """Move a joint one step in given direction (+1 or -1)."""
        cfg     = JOINTS[joint]
        step    = cfg['step'] * direction
        current = self.current_angles.get(joint, cfg['home'])
        self._set(joint, current + step)
        return self.current_angles[joint]  # return actual angle after clamping

    def go_home(self):
        """Return all joints to home position."""
        print("[Servos] Going to home position")
        for name, cfg in JOINTS.items():
            self._set(name, cfg['home'])
            time.sleep(0.1)

    def open_gripper(self):
        self._set('gripper', JOINTS['gripper']['min'])
        print("[Gripper] Opened")

    def close_gripper(self):
        self._set('gripper', JOINTS['gripper']['max'])
        print("[Gripper] Closed")

    def cleanup(self):
        """Safely release all servos."""
        if self.kit:
            for name, cfg in JOINTS.items():
                try:
                    self.kit.servo[cfg['channel']].angle = None
                except:
                    pass
        print("[Servos] Released")

    def print_angles(self):
        """Print current angles — note these down for arm_controller.py home values."""
        print("\n── Current Joint Angles ──────────────────")
        for name, angle in self.current_angles.items():
            bar_len = int((angle / 180) * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            print(f"  {name:<12} [{bar}] {angle:>5.1f}°  "
                  f"(ch{JOINTS[name]['channel']})")
        print("──────────────────────────────────────────\n")

# ============================================================================
# MANUAL CONTROLLER — main loop
# ============================================================================

class ManualController:

    def __init__(self):
        self.servos = ServoController()

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            print("[ERROR] Cannot open camera. Try changing CAMERA_INDEX to 1")
            self.servos.cleanup()
            return

        self._print_controls()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Camera frame failed")
                    break

                self._draw_overlay(frame)
                cv2.imshow('Manual Control — 4-DOF Arm', frame)

                key = cv2.waitKey(30) & 0xFF

                if key == 255:
                    continue  # no key pressed

                elif key == 27:  # ESC — emergency stop
                    print("\n[STOP] ESC pressed — stopping")
                    break

                elif key == ord(' '):  # SPACE — go home
                    print("\n[HOME] Returning to home position")
                    self.servos.go_home()

                elif key == ord('p'):  # P — print angles
                    self.servos.print_angles()

                elif key == ord('o'):  # O — open gripper
                    self.servos.open_gripper()

                elif key == ord('c'):  # C — close gripper
                    self.servos.close_gripper()

                elif key in KEY_MAP:
                    joint, direction = KEY_MAP[key]
                    actual = self.servos.move(joint, direction)
                    print(f"  {joint:<12} → {actual:.1f}°", end='\r')

        except KeyboardInterrupt:
            print("\n[Manual] Interrupted")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.servos.go_home()
            self.servos.cleanup()
            print("\n[Manual] Session ended")
            print("\nIMPORTANT: Note down the angles printed above.")
            print("Use them to fill in home angles in arm_controller.py\n")

    def _draw_overlay(self, frame):
        """Draw control guide and joint angles on camera frame."""

        # Semi transparent dark bar at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, FRAME_HEIGHT - 150),
                      (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Controls guide
        controls = [
            "W/S: Base    A/D: Shoulder    Q/E: Elbow",
            "Y/H: Gripper open/close",
            "SPACE: Home    P: Print angles    ESC: Stop",
        ]
        y = FRAME_HEIGHT - 138
        for line in controls:
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 20

        # Divider line
        cv2.line(frame, (0, FRAME_HEIGHT - 80),
                 (FRAME_WIDTH, FRAME_HEIGHT - 80), (60, 60, 60), 1)

        # Joint angles — all 4 in one row
        x = 10
        for name, angle in self.servos.current_angles.items():
            cfg = JOINTS[name]
            color = (0, 255, 180)
            if angle <= cfg['min'] + 5 or angle >= cfg['max'] - 5:
                color = (0, 80, 255)  # red if near limit
            cv2.putText(frame, f"{name}: {angle:.0f}deg",
                        (x, FRAME_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            x += 160

    def _print_controls(self):
        print("=" * 50)
        print("  4-DOF ARM — MANUAL CONTROL")
        print("=" * 50)
        print("  W / S      →  Base rotate")
        print("  A / D      →  Shoulder up/down")
        print("  Q / E      →  Elbow extend/retract")
        print("  Y / H      →  Gripper open/close")
        print("  SPACE      →  Go to home (90° all joints)")
        print("  P          →  Print current angles")
        print("  O          →  Open gripper fully")
        print("  C          →  Close gripper fully")
        print("  ESC        →  Emergency stop")
        print("=" * 50)
        print("\nTIP: Press P after positioning arm in a good")
        print("     resting pose — note those angles for")
        print("     arm_controller.py home position!\n")

# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    controller = ManualController()
    controller.run()