#!/usr/bin/env python3
"""Simple GUI to monitor Piper arm status in real time."""

import argparse
import sys
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

from piper_sdk import C_PiperInterface_V2


ARM_STATUS_LABELS = {
    0x00: "normal",
    0x01: "emergency_stop",
    0x02: "no_solution",
    0x03: "singularity",
    0x04: "target_out_of_limit",
    0x05: "joint_comm_error",
    0x06: "joint_brake_locked",
    0x07: "collision_detected",
    0x08: "teach_over_speed",
    0x09: "joint_status_error",
    0x0A: "other_error",
    0x0B: "teach_record",
    0x0C: "teach_exec",
    0x0D: "teach_pause",
    0x0E: "master_ntc_over_temp",
    0x0F: "brake_ntc_over_temp",
}

CTRL_MODE_LABELS = {
    0x00: "standby",
    0x01: "can",
    0x02: "teach",
    0x03: "ethernet",
    0x04: "wifi",
    0x07: "offline_traj",
}

MOVE_MODE_LABELS = {
    0x00: "move_p",
    0x01: "move_j",
    0x02: "move_l",
    0x03: "move_c",
    0x04: "move_m",
    0x05: "move_cpv",
}

MIT_MODE_LABELS = {
    0x00: "pos_speed",
    0xAD: "mit",
    0xFF: "invalid",
}

INSTALL_POS_LABELS = {
    0x00: "invalid",
    0x01: "horizontal_upright",
    0x02: "left_mount",
    0x03: "right_mount",
}

ERR_STATUS_FIELDS = [
    (("joint_1_angle_limit",), "j1_angle_limit"),
    (("joint_2_angle_limit",), "j2_angle_limit"),
    (("joint_3_angle_limit",), "j3_angle_limit"),
    (("joint_4_angle_limit",), "j4_angle_limit"),
    (("joint_5_angle_limit",), "j5_angle_limit"),
    (("joint_6_angle_limit",), "j6_angle_limit"),
    (("communication_status_joint_1",), "j1_comm_error"),
    (("communication_status_joint_2",), "j2_comm_error"),
    (("communication_status_joint_3",), "j3_comm_error"),
    (("communication_status_joint_4",), "j4_comm_error"),
    (("communication_status_joint_5",), "j5_comm_error"),
    (("communication_status_joint_6",), "j6_comm_error"),
]

JOINT_LIMIT_FIELDS = {
    1: ("joint_1_angle_limit",),
    2: ("joint_2_angle_limit",),
    3: ("joint_3_angle_limit",),
    4: ("joint_4_angle_limit",),
    5: ("joint_5_angle_limit",),
    6: ("joint_6_angle_limit",),
}

JOINT_COMM_FIELDS = {
    1: ("communication_status_joint_1",),
    2: ("communication_status_joint_2",),
    3: ("communication_status_joint_3",),
    4: ("communication_status_joint_4",),
    5: ("communication_status_joint_5",),
    6: ("communication_status_joint_6",),
}

def _safe_get(obj, name, default=None):
    return getattr(obj, name, default)


def _decode_foc_status(status_obj) -> Tuple[str, Optional[bool]]:
    """
    Returns (text, ok_flag). ok_flag None means unknown.
    """
    if status_obj is None:
        return "n/a", None

    # Parsed status with boolean fields
    flag_names = [
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "collision_status",
        "driver_error_status",
        "driver_enable_status",
        "stall_status",
        "sensor_status",
        "homing_status",
    ]
    flags = []
    for name in flag_names:
        if hasattr(status_obj, name):
            flags.append((name, bool(getattr(status_obj, name))))

    if flags:
        bad = [name for name, is_bad in flags if is_bad and name != "driver_enable_status"]
        if bad:
            return "bad: " + ",".join(bad), False
        return "ok", True

    # Raw integer status code
    if isinstance(status_obj, int):
        if status_obj == 0:
            return "0", True
        return str(status_obj), False

    return str(status_obj), None


def _decode_err_status(err_status) -> Tuple[str, Optional[bool]]:
    if err_status is None:
        return "n/a", None

    if isinstance(err_status, int):
        return f"0x{err_status:08X}", True if err_status == 0 else False

    active = []
    found = False
    for names, label in ERR_STATUS_FIELDS:
        value = None
        for name in names:
            if isinstance(err_status, dict) and name in err_status:
                value = err_status[name]
                found = True
                break
            if hasattr(err_status, name):
                value = getattr(err_status, name)
                found = True
                break
        if value is not None and bool(value):
            active.append(label)

    if found:
        if not active:
            return "none", True
        return ", ".join(active), False

    return str(err_status), None


def _get_err_flag(err_status, names: Tuple[str, ...]) -> Optional[bool]:
    if err_status is None:
        return None
    if isinstance(err_status, dict):
        for name in names:
            if name in err_status:
                return bool(err_status[name])
        return None
    for name in names:
        if hasattr(err_status, name):
            return bool(getattr(err_status, name))
    return None


class StatusItem:
    def __init__(self, label: str, getter):
        self.label = label
        self.getter = getter
        self.var = tk.StringVar()
        self.value_label: Optional[ttk.Label] = None


class PiperStatusGUI:
    def __init__(self, can_interface: str, refresh_ms: int, enable_arm: bool):
        self.piper = C_PiperInterface_V2(can_interface)
        self.piper.ConnectPort()

        if enable_arm:
            retry = 0
            while not self.piper.EnablePiper() and retry < 100:
                time.sleep(0.01)
                retry += 1
            if retry >= 100:
                raise RuntimeError("Failed to enable Piper arm after 100 retries")

        self.root = tk.Tk()
        self.root.title("Piper Arm Status Monitor")
        self.root.geometry("1200x820")

        style = ttk.Style(self.root)
        style.configure("Status.TLabel", font=("Consolas", 11))
        style.configure("Header.TLabel", font=("Consolas", 12, "bold"))

        self.color_ok = "#1b8a2f"
        self.color_bad = "#b51d1d"
        self.color_unknown = "#6b7280"

        self.refresh_ms = refresh_ms
        self._refresh_token = 0
        self._status_cache_token = -1
        self._status_cache = None
        self._motor_cache_token = -1
        self._motor_cache = None
        self._lowspd_cache_token = -1
        self._lowspd_cache = None
        self._angle_limit_cache_token = -1
        self._angle_limit_cache = None
        self._end_vel_acc_cache_token = -1
        self._end_vel_acc_cache = None
        self._crash_level_cache_token = -1
        self._crash_level_cache = None
        self._gripper_teach_cache_token = -1
        self._gripper_teach_cache = None
        self._motor_max_acc_cache_token = -1
        self._motor_max_acc_cache = None
        self._all_motor_max_acc_cache_token = -1
        self._all_motor_max_acc_cache = None
        self._all_motor_angle_limit_cache_token = -1
        self._all_motor_angle_limit_cache = None
        self._firmware_cache_token = -1
        self._firmware_cache = None
        self._resp_inst_cache_token = -1
        self._resp_inst_cache = None
        self._arm_mode_ctrl_cache_token = -1
        self._arm_mode_ctrl_cache = None
        self._arm_joint_ctrl_cache_token = -1
        self._arm_joint_ctrl_cache = None
        self._arm_gripper_ctrl_cache_token = -1
        self._arm_gripper_ctrl_cache = None

        self.items: List[StatusItem] = []
        self._build_layout()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(main, text="Realtime Status", style="Header.TLabel")
        header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=1)

        left_col = ttk.Frame(main)
        mid_col = ttk.Frame(main)
        right_col = ttk.Frame(main)
        left_col.grid(row=1, column=0, sticky="nw", padx=(0, 6))
        mid_col.grid(row=1, column=1, sticky="nw", padx=(6, 6))
        right_col.grid(row=1, column=2, sticky="nw", padx=(6, 0))

        def add_section(container: ttk.Frame, title: str) -> ttk.LabelFrame:
            frame = ttk.LabelFrame(container, text=title, padding=6)
            frame.pack(fill=tk.X, anchor="n", pady=(0, 6))
            frame.columnconfigure(1, weight=1)
            return frame

        def add_item(frame: ttk.LabelFrame, row: int, label: str, getter) -> int:
            item = StatusItem(label, getter)
            label_widget = ttk.Label(frame, text=label + ":", style="Status.TLabel")
            label_widget.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            value_widget = ttk.Label(frame, textvariable=item.var, style="Status.TLabel")
            value_widget.grid(row=row, column=1, sticky="w", pady=2)
            item.value_label = value_widget
            self.items.append(item)
            return row + 1

        section = add_section(left_col, "Connection & Version")
        row = 0
        row = add_item(section, row, "Connection", self._get_connection_status)
        row = add_item(section, row, "CAN FPS", self._get_can_fps)
        row = add_item(section, row, "Firmware Version", self._get_firmware_version)

        section = add_section(left_col, "Arm Status")
        row = 0
        row = add_item(section, row, "Status Timestamp", self._get_status_timestamp)
        row = add_item(section, row, "Status FPS", self._get_status_hz)
        row = add_item(section, row, "Ctrl Mode", self._get_ctrl_mode)
        row = add_item(section, row, "Move Mode", self._get_move_mode)
        row = add_item(section, row, "Arm Status", self._get_arm_status)
        row = add_item(section, row, "Teach Status", self._get_teach_status)
        row = add_item(section, row, "Motion Status", self._get_motion_status)
        row = add_item(section, row, "Trajectory No", self._get_trajectory_num)
        row = add_item(section, row, "Err Status", self._get_err_status)
        row = add_item(section, row, "Error Code", self._get_error_code)

        section = add_section(left_col, "Joint & Gripper Feedback")
        row = 0
        row = add_item(section, row, "Joints (deg)", self._get_joint_angles)
        row = add_item(section, row, "Gripper (mm)", self._get_gripper_angle)
        row = add_item(section, row, "Gripper Status", self._get_gripper_status)

        section = add_section(left_col, "Joint Faults")
        row = 0
        row = add_item(section, row, "J1 Limit", lambda: self._get_joint_limit_status(1))
        row = add_item(section, row, "J2 Limit", lambda: self._get_joint_limit_status(2))
        row = add_item(section, row, "J3 Limit", lambda: self._get_joint_limit_status(3))
        row = add_item(section, row, "J4 Limit", lambda: self._get_joint_limit_status(4))
        row = add_item(section, row, "J5 Limit", lambda: self._get_joint_limit_status(5))
        row = add_item(section, row, "J6 Limit", lambda: self._get_joint_limit_status(6))
        row = add_item(section, row, "J1 Comm", lambda: self._get_joint_comm_status(1))
        row = add_item(section, row, "J2 Comm", lambda: self._get_joint_comm_status(2))
        row = add_item(section, row, "J3 Comm", lambda: self._get_joint_comm_status(3))
        row = add_item(section, row, "J4 Comm", lambda: self._get_joint_comm_status(4))
        row = add_item(section, row, "J5 Comm", lambda: self._get_joint_comm_status(5))
        row = add_item(section, row, "J6 Comm", lambda: self._get_joint_comm_status(6))

        section = add_section(mid_col, "Motor High-Speed")
        row = 0
        row = add_item(section, row, "Motor Timestamp", self._get_motor_timestamp)
        row = add_item(section, row, "Motor FPS", self._get_motor_hz)
        row = add_item(section, row, "Motor Speed", self._get_motor_speeds)
        row = add_item(section, row, "Motor Current", self._get_motor_currents)
        row = add_item(section, row, "Motor Pos", self._get_motor_positions)
        row = add_item(section, row, "Motor Effort", self._get_motor_efforts)

        section = add_section(mid_col, "Motor Low-Speed")
        row = 0
        row = add_item(section, row, "LowSpd Timestamp", self._get_lowspd_timestamp)
        row = add_item(section, row, "LowSpd FPS", self._get_lowspd_hz)
        row = add_item(section, row, "Driver Voltage (V)", self._get_driver_voltages)
        row = add_item(section, row, "Driver Temp (C)", self._get_driver_temps)
        row = add_item(section, row, "Motor Temp (C)", self._get_motor_temps)
        row = add_item(section, row, "Bus Current (A)", self._get_bus_currents)
        row = add_item(section, row, "Driver Enable", self._get_driver_enable_statuses)
        row = add_item(section, row, "Driver 1", lambda: self._get_driver_status(1))
        row = add_item(section, row, "Driver 2", lambda: self._get_driver_status(2))
        row = add_item(section, row, "Driver 3", lambda: self._get_driver_status(3))
        row = add_item(section, row, "Driver 4", lambda: self._get_driver_status(4))
        row = add_item(section, row, "Driver 5", lambda: self._get_driver_status(5))
        row = add_item(section, row, "Driver 6", lambda: self._get_driver_status(6))

        section = add_section(mid_col, "Commands")
        row = 0
        row = add_item(section, row, "ArmMode Timestamp", self._get_arm_mode_ctrl_timestamp)
        row = add_item(section, row, "ArmMode FPS", self._get_arm_mode_ctrl_hz)
        row = add_item(section, row, "ArmMode Ctrl", self._get_arm_mode_ctrl_mode)
        row = add_item(section, row, "ArmMode Move", self._get_arm_mode_move_mode)
        row = add_item(section, row, "ArmMode Speed %", self._get_arm_mode_speed)
        row = add_item(section, row, "ArmMode MIT", self._get_arm_mode_mit_mode)
        row = add_item(section, row, "ArmMode Residence", self._get_arm_mode_residence_time)
        row = add_item(section, row, "ArmMode Install", self._get_arm_mode_install_pos)
        row = add_item(section, row, "JointCtrl Timestamp", self._get_arm_joint_ctrl_timestamp)
        row = add_item(section, row, "JointCtrl FPS", self._get_arm_joint_ctrl_hz)
        row = add_item(section, row, "JointCtrl Angles (deg)", self._get_arm_joint_ctrl_angles)
        row = add_item(section, row, "GripperCtrl Timestamp", self._get_gripper_ctrl_timestamp)
        row = add_item(section, row, "GripperCtrl FPS", self._get_gripper_ctrl_hz)
        row = add_item(section, row, "GripperCtrl Angle (mm)", self._get_gripper_ctrl_angle)
        row = add_item(section, row, "GripperCtrl Effort", self._get_gripper_ctrl_effort)
        row = add_item(section, row, "GripperCtrl Status", self._get_gripper_ctrl_status)
        row = add_item(section, row, "GripperCtrl SetZero", self._get_gripper_ctrl_set_zero)

        section = add_section(right_col, "Limits & Params")
        row = 0
        row = add_item(section, row, "Limit Timestamp", self._get_angle_limit_timestamp)
        row = add_item(section, row, "Angle Limits (deg)", self._get_angle_limits)
        row = add_item(section, row, "Max Joint Speed", self._get_max_joint_speed)
        row = add_item(section, row, "Max Acc Timestamp", self._get_motor_max_acc_timestamp)
        row = add_item(section, row, "Max Joint Acc", self._get_motor_max_joint_acc)
        row = add_item(section, row, "All Max Acc Timestamp", self._get_all_motor_max_acc_timestamp)
        row = add_item(section, row, "All Max Joint Acc", self._get_all_motor_max_joint_acc)
        row = add_item(section, row, "All Angle/Spd Timestamp", self._get_all_motor_angle_limit_timestamp)
        row = add_item(section, row, "All Angle Limits", self._get_all_motor_angle_limits)
        row = add_item(section, row, "All Max Joint Spd", self._get_all_motor_max_joint_spd)
        row = add_item(section, row, "End Vel/Acc Timestamp", self._get_end_vel_acc_timestamp)
        row = add_item(section, row, "End Max Linear Vel", self._get_end_max_linear_vel)
        row = add_item(section, row, "End Max Angular Vel", self._get_end_max_angular_vel)
        row = add_item(section, row, "End Max Linear Acc", self._get_end_max_linear_acc)
        row = add_item(section, row, "End Max Angular Acc", self._get_end_max_angular_acc)
        row = add_item(section, row, "Crash Level Timestamp", self._get_crash_level_timestamp)
        row = add_item(section, row, "Crash Levels", self._get_crash_levels)
        row = add_item(section, row, "Gripper Teach Timestamp", self._get_gripper_teach_timestamp)
        row = add_item(section, row, "Teach Range %", self._get_teach_range_percent)
        row = add_item(section, row, "Max Teach Range", self._get_max_teach_range)
        row = add_item(section, row, "Teach Friction", self._get_teach_friction)

        section = add_section(right_col, "Responses")
        row = 0
        row = add_item(section, row, "Resp Timestamp", self._get_resp_instruction_timestamp)
        row = add_item(section, row, "Resp Instruction", self._get_resp_instruction_index)
        row = add_item(section, row, "Zero Config OK", self._get_zero_config_flag)

    def _set_item_state(self, item: StatusItem, text: str, state: str):
        item.var.set(text)
        color = self.color_unknown
        if state == "ok":
            color = self.color_ok
        elif state == "bad":
            color = self.color_bad
        if item.value_label is not None:
            item.value_label.configure(foreground=color)

    def _get_connection_status(self):
        try:
            ok = bool(self.piper.isOk())
            return ("ok" if ok else "no data"), "ok" if ok else "bad"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_can_fps(self):
        try:
            fps = self.piper.GetCanFps()
            return f"{fps:.2f}", "ok" if fps > 0 else "bad"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_status_packet(self):
        if self._status_cache_token != self._refresh_token:
            self._status_cache_token = self._refresh_token
            self._status_cache = self.piper.GetArmStatus()
        return self._status_cache

    def _get_motor_packet(self):
        if self._motor_cache_token != self._refresh_token:
            self._motor_cache_token = self._refresh_token
            self._motor_cache = self.piper.GetArmHighSpdInfoMsgs()
        return self._motor_cache

    def _get_lowspd_packet(self):
        if self._lowspd_cache_token != self._refresh_token:
            self._lowspd_cache_token = self._refresh_token
            self._lowspd_cache = self.piper.GetArmLowSpdInfoMsgs()
        return self._lowspd_cache

    def _get_angle_limit_packet(self):
        if self._angle_limit_cache_token != self._refresh_token:
            self._angle_limit_cache_token = self._refresh_token
            self._angle_limit_cache = self.piper.GetCurrentMotorAngleLimitMaxVel()
        return self._angle_limit_cache

    def _get_angle_limit_payload(self):
        packet = self._get_angle_limit_packet()
        if packet is None:
            return None
        return _safe_get(packet, "current_motor_angle_limit_max_vel", packet)

    def _get_end_vel_acc_packet(self):
        if self._end_vel_acc_cache_token != self._refresh_token:
            self._end_vel_acc_cache_token = self._refresh_token
            self._end_vel_acc_cache = self.piper.GetCurrentEndVelAndAccParam()
        return self._end_vel_acc_cache

    def _get_end_vel_acc_payload(self):
        packet = self._get_end_vel_acc_packet()
        if packet is None:
            return None
        return _safe_get(packet, "current_end_vel_acc_param", packet)

    def _get_crash_level_packet(self):
        if self._crash_level_cache_token != self._refresh_token:
            self._crash_level_cache_token = self._refresh_token
            self._crash_level_cache = self.piper.GetCrashProtectionLevelFeedback()
        return self._crash_level_cache

    def _get_crash_level_payload(self):
        packet = self._get_crash_level_packet()
        if packet is None:
            return None
        return _safe_get(packet, "crash_protection_level_feedback", packet)

    def _get_gripper_teach_packet(self):
        if self._gripper_teach_cache_token != self._refresh_token:
            self._gripper_teach_cache_token = self._refresh_token
            self._gripper_teach_cache = self.piper.GetGripperTeachingPendantParamFeedback()
        return self._gripper_teach_cache

    def _get_gripper_teach_payload(self):
        packet = self._get_gripper_teach_packet()
        if packet is None:
            return None
        return _safe_get(packet, "arm_gripper_teaching_param_feedback", packet)

    def _get_motor_max_acc_packet(self):
        if self._motor_max_acc_cache_token != self._refresh_token:
            self._motor_max_acc_cache_token = self._refresh_token
            self._motor_max_acc_cache = self.piper.GetCurrentMotorMaxAccLimit()
        return self._motor_max_acc_cache

    def _get_motor_max_acc_payload(self):
        packet = self._get_motor_max_acc_packet()
        if packet is None:
            return None
        return _safe_get(packet, "current_motor_max_acc_limit", packet)

    def _get_all_motor_max_acc_packet(self):
        if self._all_motor_max_acc_cache_token != self._refresh_token:
            self._all_motor_max_acc_cache_token = self._refresh_token
            self._all_motor_max_acc_cache = self.piper.GetAllMotorMaxAccLimit()
        return self._all_motor_max_acc_cache

    def _get_all_motor_max_acc_payload(self):
        packet = self._get_all_motor_max_acc_packet()
        if packet is None:
            return None
        return _safe_get(packet, "all_motor_max_acc_limit", packet)

    def _get_all_motor_angle_limit_packet(self):
        if self._all_motor_angle_limit_cache_token != self._refresh_token:
            self._all_motor_angle_limit_cache_token = self._refresh_token
            self._all_motor_angle_limit_cache = self.piper.GetAllMotorAngleLimitMaxSpd()
        return self._all_motor_angle_limit_cache

    def _get_all_motor_angle_limit_payload(self):
        packet = self._get_all_motor_angle_limit_packet()
        if packet is None:
            return None
        return _safe_get(packet, "all_motor_angle_limit_max_spd", packet)

    def _get_firmware_version_packet(self):
        if self._firmware_cache_token != self._refresh_token:
            self._firmware_cache_token = self._refresh_token
            self._firmware_cache = self.piper.GetPiperFirmwareVersion()
        return self._firmware_cache

    def _get_resp_instruction_packet(self):
        if self._resp_inst_cache_token != self._refresh_token:
            self._resp_inst_cache_token = self._refresh_token
            self._resp_inst_cache = self.piper.GetRespInstruction()
        return self._resp_inst_cache

    def _get_resp_instruction_payload(self):
        packet = self._get_resp_instruction_packet()
        if packet is None:
            return None
        return packet

    def _get_arm_mode_ctrl_packet(self):
        if self._arm_mode_ctrl_cache_token != self._refresh_token:
            self._arm_mode_ctrl_cache_token = self._refresh_token
            self._arm_mode_ctrl_cache = self.piper.GetArmModeCtrl()
        return self._arm_mode_ctrl_cache

    def _get_arm_mode_ctrl_payload(self):
        packet = self._get_arm_mode_ctrl_packet()
        if packet is None:
            return None
        return _safe_get(packet, "ctrl_151", packet)

    def _get_arm_joint_ctrl_packet(self):
        if self._arm_joint_ctrl_cache_token != self._refresh_token:
            self._arm_joint_ctrl_cache_token = self._refresh_token
            self._arm_joint_ctrl_cache = self.piper.GetArmJointCtrl()
        return self._arm_joint_ctrl_cache

    def _get_arm_joint_ctrl_payload(self):
        packet = self._get_arm_joint_ctrl_packet()
        if packet is None:
            return None
        return _safe_get(packet, "joint_ctrl", packet)

    def _get_gripper_ctrl_packet(self):
        if self._arm_gripper_ctrl_cache_token != self._refresh_token:
            self._arm_gripper_ctrl_cache_token = self._refresh_token
            self._arm_gripper_ctrl_cache = self.piper.GetArmGripperCtrl()
        return self._arm_gripper_ctrl_cache

    def _get_gripper_ctrl_payload(self):
        packet = self._get_gripper_ctrl_packet()
        if packet is None:
            return None
        return _safe_get(packet, "gripper_ctrl", packet)

    def _get_motor_list(self):
        packet = self._get_motor_packet()
        if packet is None:
            return []
        motors = []
        for idx in range(1, 7):
            motor = _safe_get(packet, f"motor_{idx}", None)
            if motor is not None:
                motors.append((idx, motor))
        return motors

    def _get_lowspd_list(self):
        packet = self._get_lowspd_packet()
        if packet is None:
            return []
        motors = []
        for idx in range(1, 7):
            motor = _safe_get(packet, f"motor_{idx}", None)
            if motor is not None:
                motors.append((idx, motor))
        return motors

    def _get_arm_status_struct(self):
        status_packet = self._get_status_packet()
        return _safe_get(status_packet, "arm_status", None)

    def _get_status_timestamp(self):
        try:
            status_packet = self._get_status_packet()
            time_stamp = _safe_get(status_packet, "time_stamp", _safe_get(status_packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_status_hz(self):
        try:
            status_packet = self._get_status_packet()
            hz = _safe_get(status_packet, "Hz", _safe_get(status_packet, "hz", None))
            if hz is None:
                return "n/a", "unknown"
            try:
                hz_value = float(hz)
            except (TypeError, ValueError):
                return str(hz), "unknown"
            return f"{hz_value:.2f}", "ok" if hz_value > 0 else "bad"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_motor_timestamp(self):
        try:
            packet = self._get_motor_packet()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_motor_hz(self):
        try:
            packet = self._get_motor_packet()
            hz = _safe_get(packet, "Hz", _safe_get(packet, "hz", None))
            if hz is None:
                return "n/a", "unknown"
            try:
                hz_value = float(hz)
            except (TypeError, ValueError):
                return str(hz), "unknown"
            return f"{hz_value:.2f}", "ok" if hz_value > 0 else "bad"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_lowspd_timestamp(self):
        try:
            packet = self._get_lowspd_packet()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_lowspd_hz(self):
        try:
            packet = self._get_lowspd_packet()
            hz = _safe_get(packet, "Hz", _safe_get(packet, "hz", None))
            if hz is None:
                return "n/a", "unknown"
            try:
                hz_value = float(hz)
            except (TypeError, ValueError):
                return str(hz), "unknown"
            return f"{hz_value:.2f}", "ok" if hz_value > 0 else "bad"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_angle_limit_timestamp(self):
        try:
            packet = self._get_angle_limit_packet()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_angle_limits(self):
        try:
            payload = self._get_angle_limit_payload()
            if payload is None:
                return "n/a", "unknown"
            motor_num = _safe_get(payload, "motor_num", None)
            max_limit = _safe_get(payload, "max_angle_limit", None)
            min_limit = _safe_get(payload, "min_angle_limit", None)
            if max_limit is None and min_limit is None:
                return "n/a", "unknown"
            max_deg = None
            min_deg = None
            try:
                if max_limit is not None:
                    max_deg = float(max_limit) / 10.0
                if min_limit is not None:
                    min_deg = float(min_limit) / 10.0
            except (TypeError, ValueError):
                max_deg = max_limit
                min_deg = min_limit
            prefix = f"m{motor_num}: " if motor_num is not None else ""
            return f"{prefix}min {min_deg}, max {max_deg}", "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_max_joint_speed(self):
        try:
            payload = self._get_angle_limit_payload()
            if payload is None:
                return "n/a", "unknown"
            motor_num = _safe_get(payload, "motor_num", None)
            max_spd = _safe_get(payload, "max_joint_spd", None)
            if max_spd is None:
                return "n/a", "unknown"
            try:
                spd = float(max_spd) * 0.001
            except (TypeError, ValueError):
                spd = max_spd
            prefix = f"m{motor_num}: " if motor_num is not None else ""
            return f"{prefix}{spd}", "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_end_vel_acc_timestamp(self):
        try:
            packet = self._get_end_vel_acc_packet()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _format_end_vel_acc_field(self, field: str, scale: float) -> Tuple[str, str]:
        payload = self._get_end_vel_acc_payload()
        if payload is None:
            return "n/a", "unknown"
        raw = _safe_get(payload, field, None)
        if raw is None:
            return "n/a", "unknown"
        try:
            value = float(raw) * scale
        except (TypeError, ValueError):
            value = raw
        return str(value), "ok"

    def _get_end_max_linear_vel(self):
        try:
            return self._format_end_vel_acc_field("end_max_linear_vel", 0.001)
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_end_max_angular_vel(self):
        try:
            return self._format_end_vel_acc_field("end_max_angular_vel", 0.001)
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_end_max_linear_acc(self):
        try:
            return self._format_end_vel_acc_field("end_max_linear_acc", 0.001)
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_end_max_angular_acc(self):
        try:
            return self._format_end_vel_acc_field("end_max_angular_acc", 0.001)
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_crash_level_timestamp(self):
        try:
            packet = self._get_crash_level_packet()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_crash_levels(self):
        try:
            payload = self._get_crash_level_payload()
            if payload is None:
                return "n/a", "unknown"
            values = []
            for idx in range(1, 7):
                level = _safe_get(payload, f"joint_{idx}_protection_level", None)
                if level is None:
                    values.append(f"j{idx}:n/a")
                else:
                    values.append(f"j{idx}:{level}")
            return "[" + ", ".join(values) + "]", "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_gripper_teach_timestamp(self):
        try:
            packet = self._get_gripper_teach_packet()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_teach_range_percent(self):
        try:
            payload = self._get_gripper_teach_payload()
            if payload is None:
                return "n/a", "unknown"
            value = _safe_get(payload, "teaching_range_per", None)
            if value is None:
                return "n/a", "unknown"
            return str(value), "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_max_teach_range(self):
        try:
            payload = self._get_gripper_teach_payload()
            if payload is None:
                return "n/a", "unknown"
            value = _safe_get(payload, "max_range_config", None)
            if value is None:
                return "n/a", "unknown"
            return str(value), "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_teach_friction(self):
        try:
            payload = self._get_gripper_teach_payload()
            if payload is None:
                return "n/a", "unknown"
            value = _safe_get(payload, "teaching_friction", None)
            if value is None:
                return "n/a", "unknown"
            return str(value), "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_motor_max_acc_timestamp(self):
        try:
            packet = self._get_motor_max_acc_packet()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_motor_max_joint_acc(self):
        try:
            payload = self._get_motor_max_acc_payload()
            if payload is None:
                return "n/a", "unknown"
            motor_num = _safe_get(payload, "joint_motor_num", None)
            max_acc = _safe_get(payload, "max_joint_acc", None)
            if max_acc is None:
                return "n/a", "unknown"
            try:
                acc = float(max_acc) * 0.001
            except (TypeError, ValueError):
                acc = max_acc
            prefix = f"m{motor_num}: " if motor_num is not None else ""
            return f"{prefix}{acc}", "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_all_motor_max_acc_timestamp(self):
        try:
            packet = self._get_all_motor_max_acc_packet()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_all_motor_max_joint_acc(self):
        try:
            payload = self._get_all_motor_max_acc_payload()
            if payload is None:
                return "n/a", "unknown"
            motors = []
            for idx in range(1, 7):
                motor = _safe_get(payload, f"motor_{idx}", None)
                if motor is None:
                    continue
                motor_num = _safe_get(motor, "joint_motor_num", idx)
                max_acc = _safe_get(motor, "max_joint_acc", None)
                if max_acc is None:
                    motors.append(f"m{motor_num}:n/a")
                else:
                    try:
                        acc = float(max_acc) * 0.001
                    except (TypeError, ValueError):
                        acc = max_acc
                    motors.append(f"m{motor_num}:{acc}")
            if not motors:
                return "n/a", "unknown"
            return "[" + ", ".join(motors) + "]", "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_all_motor_angle_limit_timestamp(self):
        try:
            packet = self._get_all_motor_angle_limit_packet()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_all_motor_angle_limits(self):
        try:
            payload = self._get_all_motor_angle_limit_payload()
            if payload is None:
                return "n/a", "unknown"
            motors = []
            for idx in range(1, 7):
                motor = _safe_get(payload, f"motor_{idx}", None)
                if motor is None:
                    continue
                motor_num = _safe_get(motor, "motor_num", idx)
                max_limit = _safe_get(motor, "max_angle_limit", None)
                min_limit = _safe_get(motor, "min_angle_limit", None)
                if max_limit is None and min_limit is None:
                    motors.append(f"m{motor_num}:n/a")
                    continue
                try:
                    max_deg = float(max_limit) / 10.0 if max_limit is not None else "n/a"
                    min_deg = float(min_limit) / 10.0 if min_limit is not None else "n/a"
                except (TypeError, ValueError):
                    max_deg = max_limit
                    min_deg = min_limit
                motors.append(f"m{motor_num}:min {min_deg}, max {max_deg}")
            if not motors:
                return "n/a", "unknown"
            return "[" + ", ".join(motors) + "]", "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_all_motor_max_joint_spd(self):
        try:
            payload = self._get_all_motor_angle_limit_payload()
            if payload is None:
                return "n/a", "unknown"
            motors = []
            for idx in range(1, 7):
                motor = _safe_get(payload, f"motor_{idx}", None)
                if motor is None:
                    continue
                motor_num = _safe_get(motor, "motor_num", idx)
                max_spd = _safe_get(motor, "max_joint_spd", None)
                if max_spd is None:
                    motors.append(f"m{motor_num}:n/a")
                else:
                    try:
                        spd = float(max_spd) * 0.001
                    except (TypeError, ValueError):
                        spd = max_spd
                    motors.append(f"m{motor_num}:{spd}")
            if not motors:
                return "n/a", "unknown"
            return "[" + ", ".join(motors) + "]", "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_firmware_version(self):
        try:
            version = self._get_firmware_version_packet()
            if version is None:
                return "n/a", "unknown"
            if isinstance(version, int) and version < 0:
                return f"{version}", "bad"
            return str(version), "ok"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_resp_instruction_timestamp(self):
        try:
            packet = self._get_resp_instruction_payload()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_resp_instruction_index(self):
        try:
            packet = self._get_resp_instruction_payload()
            value = _safe_get(packet, "instruction_index", None)
            if value is None:
                return "n/a", "unknown"
            try:
                return f"0x{int(value):02X}", "ok"
            except (TypeError, ValueError):
                return str(value), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_zero_config_flag(self):
        try:
            packet = self._get_resp_instruction_payload()
            value = _safe_get(packet, "zero_config_success_flag", None)
            if value is None:
                return "n/a", "unknown"
            if value in (0, 1):
                return "ok" if value == 1 else "fail", "ok" if value == 1 else "bad"
            return str(value), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_arm_mode_ctrl_timestamp(self):
        try:
            packet = self._get_arm_mode_ctrl_packet()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_arm_mode_ctrl_hz(self):
        try:
            packet = self._get_arm_mode_ctrl_packet()
            hz = _safe_get(packet, "Hz", _safe_get(packet, "hz", None))
            if hz is None:
                return "n/a", "unknown"
            try:
                hz_value = float(hz)
            except (TypeError, ValueError):
                return str(hz), "unknown"
            return f"{hz_value:.2f}", "ok" if hz_value > 0 else "bad"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_arm_mode_ctrl_mode(self):
        try:
            payload = self._get_arm_mode_ctrl_payload()
            if payload is None:
                return "n/a", "unknown"
            ctrl_mode = _safe_get(payload, "ctrl_mode", None)
            label = CTRL_MODE_LABELS.get(ctrl_mode, str(ctrl_mode))
            return label, "ok" if ctrl_mode in CTRL_MODE_LABELS else "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_arm_mode_move_mode(self):
        try:
            payload = self._get_arm_mode_ctrl_payload()
            if payload is None:
                return "n/a", "unknown"
            move_mode = _safe_get(payload, "move_mode", None)
            label = MOVE_MODE_LABELS.get(move_mode, str(move_mode))
            return label, "ok" if move_mode in MOVE_MODE_LABELS else "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_arm_mode_speed(self):
        try:
            payload = self._get_arm_mode_ctrl_payload()
            if payload is None:
                return "n/a", "unknown"
            value = _safe_get(payload, "move_spd_rate_ctrl", None)
            if value is None:
                return "n/a", "unknown"
            return str(value), "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_arm_mode_mit_mode(self):
        try:
            payload = self._get_arm_mode_ctrl_payload()
            if payload is None:
                return "n/a", "unknown"
            value = _safe_get(payload, "mit_mode", None)
            label = MIT_MODE_LABELS.get(value, str(value))
            return label, "ok" if value in MIT_MODE_LABELS else "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_arm_mode_residence_time(self):
        try:
            payload = self._get_arm_mode_ctrl_payload()
            if payload is None:
                return "n/a", "unknown"
            value = _safe_get(payload, "residence_time", None)
            if value is None:
                return "n/a", "unknown"
            return str(value), "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_arm_mode_install_pos(self):
        try:
            payload = self._get_arm_mode_ctrl_payload()
            if payload is None:
                return "n/a", "unknown"
            value = _safe_get(payload, "installation_pos", None)
            label = INSTALL_POS_LABELS.get(value, str(value))
            return label, "ok" if value in INSTALL_POS_LABELS else "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_arm_joint_ctrl_timestamp(self):
        try:
            packet = self._get_arm_joint_ctrl_packet()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_arm_joint_ctrl_hz(self):
        try:
            packet = self._get_arm_joint_ctrl_packet()
            hz = _safe_get(packet, "Hz", _safe_get(packet, "hz", None))
            if hz is None:
                return "n/a", "unknown"
            try:
                hz_value = float(hz)
            except (TypeError, ValueError):
                return str(hz), "unknown"
            return f"{hz_value:.2f}", "ok" if hz_value > 0 else "bad"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_arm_joint_ctrl_angles(self):
        try:
            payload = self._get_arm_joint_ctrl_payload()
            if payload is None:
                return "n/a", "unknown"
            joints = []
            for idx in range(1, 7):
                value = _safe_get(payload, f"joint_{idx}", None)
                if value is None:
                    joints.append("n/a")
                else:
                    try:
                        deg = float(value) / 1000.0
                    except (TypeError, ValueError):
                        deg = value
                    joints.append(f"{deg:.2f}" if isinstance(deg, float) else str(deg))
            return "[" + ", ".join(joints) + "]", "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_gripper_ctrl_timestamp(self):
        try:
            packet = self._get_gripper_ctrl_packet()
            time_stamp = _safe_get(packet, "time_stamp", _safe_get(packet, "timestamp", None))
            if time_stamp is None:
                return "n/a", "unknown"
            try:
                return f"{float(time_stamp):.3f}", "ok"
            except (TypeError, ValueError):
                return str(time_stamp), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_gripper_ctrl_hz(self):
        try:
            packet = self._get_gripper_ctrl_packet()
            hz = _safe_get(packet, "Hz", _safe_get(packet, "hz", None))
            if hz is None:
                return "n/a", "unknown"
            try:
                hz_value = float(hz)
            except (TypeError, ValueError):
                return str(hz), "unknown"
            return f"{hz_value:.2f}", "ok" if hz_value > 0 else "bad"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_gripper_ctrl_angle(self):
        try:
            payload = self._get_gripper_ctrl_payload()
            if payload is None:
                return "n/a", "unknown"
            value = _safe_get(payload, "grippers_angle", None)
            if value is None:
                return "n/a", "unknown"
            try:
                mm = float(value) / 1000.0
            except (TypeError, ValueError):
                mm = value
            return f"{mm:.2f}" if isinstance(mm, float) else str(mm), "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_gripper_ctrl_effort(self):
        try:
            payload = self._get_gripper_ctrl_payload()
            if payload is None:
                return "n/a", "unknown"
            value = _safe_get(payload, "grippers_effort", None)
            if value is None:
                return "n/a", "unknown"
            try:
                effort = float(value) / 1000.0
            except (TypeError, ValueError):
                effort = value
            return f"{effort:.3f}" if isinstance(effort, float) else str(effort), "ok"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_gripper_ctrl_status(self):
        try:
            payload = self._get_gripper_ctrl_payload()
            if payload is None:
                return "n/a", "unknown"
            value = _safe_get(payload, "status_code", None)
            if value is None:
                return "n/a", "unknown"
            return f"0x{int(value):02X}", "ok"
        except (TypeError, ValueError):
            return "n/a", "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_gripper_ctrl_set_zero(self):
        try:
            payload = self._get_gripper_ctrl_payload()
            if payload is None:
                return "n/a", "unknown"
            value = _safe_get(payload, "set_zero", None)
            if value is None:
                return "n/a", "unknown"
            try:
                return f"0x{int(value):02X}", "ok"
            except (TypeError, ValueError):
                return str(value), "unknown"
        except AttributeError:
            return "n/a", "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _format_motor_field(self, field: str, scale: float = 1.0) -> Tuple[str, str]:
        motors = self._get_motor_list()
        if not motors:
            return "n/a", "unknown"
        values = []
        for idx, motor in motors:
            raw = _safe_get(motor, field, None)
            if raw is None:
                values.append(f"m{idx}:n/a")
            else:
                try:
                    val = float(raw) * scale
                    values.append(f"m{idx}:{val:.3f}")
                except (TypeError, ValueError):
                    values.append(f"m{idx}:{raw}")
        return "[" + ", ".join(values) + "]", "ok"

    def _get_motor_speeds(self):
        try:
            return self._format_motor_field("motor_speed", scale=0.001)
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_motor_currents(self):
        try:
            return self._format_motor_field("current", scale=0.001)
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_motor_positions(self):
        try:
            return self._format_motor_field("pos", scale=1.0)
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_motor_efforts(self):
        try:
            return self._format_motor_field("effort", scale=0.001)
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _format_lowspd_field(self, field: str, scale: float = 1.0) -> Tuple[str, str]:
        motors = self._get_lowspd_list()
        if not motors:
            return "n/a", "unknown"
        values = []
        for idx, motor in motors:
            raw = _safe_get(motor, field, None)
            if raw is None:
                values.append(f"m{idx}:n/a")
            else:
                try:
                    val = float(raw) * scale
                    values.append(f"m{idx}:{val:.3f}")
                except (TypeError, ValueError):
                    values.append(f"m{idx}:{raw}")
        return "[" + ", ".join(values) + "]", "ok"

    def _get_driver_voltages(self):
        try:
            return self._format_lowspd_field("vol", scale=0.1)
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_driver_temps(self):
        try:
            return self._format_lowspd_field("foc_temp", scale=1.0)
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_motor_temps(self):
        try:
            return self._format_lowspd_field("motor_temp", scale=1.0)
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_bus_currents(self):
        try:
            return self._format_lowspd_field("bus_current", scale=0.001)
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_driver_enable_statuses(self):
        try:
            motors = self._get_lowspd_list()
            if not motors:
                return "n/a", "unknown"
            values = []
            for idx, motor in motors:
                foc_status = _safe_get(motor, "foc_status", None)
                enabled = _safe_get(foc_status, "driver_enable_status", None)
                if enabled is None:
                    values.append(f"m{idx}:n/a")
                else:
                    values.append(f"m{idx}:{'on' if enabled else 'off'}")
            text = "[" + ", ".join(values) + "]"
            state = "ok" if all("off" not in v for v in values if "n/a" not in v) else "bad"
            return text, state
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_ctrl_mode(self):
        try:
            status = self._get_arm_status_struct()
            ctrl_mode = _safe_get(status, "ctrl_mode", None)
            label = CTRL_MODE_LABELS.get(ctrl_mode, str(ctrl_mode))
            return label, "ok" if ctrl_mode in CTRL_MODE_LABELS else "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_move_mode(self):
        try:
            status = self._get_arm_status_struct()
            move_mode = _safe_get(status, "mode_feed", None)
            label = MOVE_MODE_LABELS.get(move_mode, str(move_mode))
            return label, "ok" if move_mode in MOVE_MODE_LABELS else "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_arm_status(self):
        try:
            status = self._get_arm_status_struct()
            code = _safe_get(status, "arm_status", None)
            if code is None:
                return "n/a", "unknown"
            try:
                int_code = int(code)
            except (TypeError, ValueError):
                return str(code), "unknown"
            label = ARM_STATUS_LABELS.get(int_code, "unknown")
            text = f"0x{int_code:02X} ({label})"
            return text, "ok" if int_code == 0 else "bad"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_teach_status(self):
        try:
            status = self._get_arm_status_struct()
            teach_status = _safe_get(status, "teach_status", None)
            if teach_status is None:
                return "n/a", "unknown"
            return str(teach_status), "ok"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_motion_status(self):
        try:
            status = self._get_arm_status_struct()
            motion = _safe_get(status, "motion_status", None)
            if motion is None:
                return "n/a", "unknown"
            label = "arrived" if motion == 0 else "moving"
            return f"{motion} ({label})", "ok" if motion == 0 else "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_trajectory_num(self):
        try:
            status = self._get_arm_status_struct()
            trajectory_num = _safe_get(status, "trajectory_num", None)
            if trajectory_num is None:
                return "n/a", "unknown"
            return str(trajectory_num), "ok"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_err_status(self):
        try:
            status = self._get_arm_status_struct()
            err_status = _safe_get(status, "err_status", None)
            text, ok_flag = _decode_err_status(err_status)
            if ok_flag is None:
                return text, "unknown"
            return text, "ok" if ok_flag else "bad"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_joint_limit_status(self, joint_id: int):
        try:
            status = self._get_arm_status_struct()
            err_status = _safe_get(status, "err_status", None)
            names = JOINT_LIMIT_FIELDS.get(joint_id, ())
            value = _get_err_flag(err_status, names)
            if value is None:
                return "n/a", "unknown"
            return "limit" if value else "ok", "bad" if value else "ok"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_joint_comm_status(self, joint_id: int):
        try:
            status = self._get_arm_status_struct()
            err_status = _safe_get(status, "err_status", None)
            names = JOINT_COMM_FIELDS.get(joint_id, ())
            value = _get_err_flag(err_status, names)
            if value is None:
                return "n/a", "unknown"
            return "error" if value else "ok", "bad" if value else "ok"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_error_code(self):
        try:
            status = self._get_arm_status_struct()
            err_code = _safe_get(status, "err_code", _safe_get(status, "err_status", None))
            if err_code is None:
                return "n/a", "unknown"
            if isinstance(err_code, int):
                return f"0x{err_code:08X}", "ok" if err_code == 0 else "bad"
            return str(err_code), "unknown"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_joint_angles(self):
        try:
            joint_feedback = self.piper.GetArmJointMsgs().joint_state
            joints = [
                joint_feedback.joint_1,
                joint_feedback.joint_2,
                joint_feedback.joint_3,
                joint_feedback.joint_4,
                joint_feedback.joint_5,
                joint_feedback.joint_6,
            ]
            deg = [f"{j / 1000.0:.2f}" for j in joints]
            return "[" + ", ".join(deg) + "]", "ok"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_gripper_angle(self):
        try:
            gripper_feedback = self.piper.GetArmGripperMsgs().gripper_state
            angle = _safe_get(gripper_feedback, "grippers_angle", None)
            if angle is None:
                return "n/a", "unknown"
            return f"{angle / 1000.0:.2f}", "ok"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_gripper_status(self):
        try:
            gripper_feedback = self.piper.GetArmGripperMsgs().gripper_state
            status_obj = _safe_get(gripper_feedback, "foc_status", None)
            text, ok_flag = _decode_foc_status(status_obj)
            if ok_flag is None:
                return text, "unknown"
            return text, "ok" if ok_flag else "bad"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _get_driver_status(self, motor_id: int):
        try:
            motors = self.piper.GetArmLowSpdInfoMsgs()
            motor = _safe_get(motors, f"motor_{motor_id}", None)
            if motor is None:
                return "n/a", "unknown"
            status_obj = _safe_get(motor, "foc_status", _safe_get(motor, "foc_status_code", None))
            text, ok_flag = _decode_foc_status(status_obj)
            if ok_flag is None:
                return text, "unknown"
            return text, "ok" if ok_flag else "bad"
        except Exception as exc:
            return f"error: {exc}", "bad"

    def _on_close(self):
        try:
            self.piper.DisconnectPort()
        except Exception:
            pass
        self.root.destroy()

    def refresh(self):
        self._refresh_token += 1
        for item in self.items:
            try:
                text, state = item.getter()
            except Exception as exc:
                text = f"error: {exc}"
                state = "bad"
            self._set_item_state(item, text, state)
        self.root.after(self.refresh_ms, self.refresh)

    def run(self):
        self.refresh()
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Piper arm status GUI monitor")
    parser.add_argument("--can-interface", type=str, default="can0", help="CAN interface name")
    parser.add_argument("--refresh-ms", type=int, default=200, help="Refresh interval in ms")
    parser.add_argument("--enable-arm", action="store_true", help="Enable the arm at startup")
    args = parser.parse_args()

    try:
        app = PiperStatusGUI(args.can_interface, args.refresh_ms, args.enable_arm)
        app.run()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
