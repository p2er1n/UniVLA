import time
import argparse
import h5py
import sys
from piper_sdk import *

def get_first_endpose(file_path, dataset_path="arm/endPose/piper_end"):
    """
    从 HDF5 文件中读取第一帧 endpose 数据:
    期望 endpose 为 [x, y, z, rx, ry, rz, gripper(optional)]
    """
    try:
        with h5py.File(file_path, "r") as f:
            if dataset_path not in f:
                print(f"错误：找不到数据集路径 {dataset_path}")
                print("文件内可用的 key（部分）:")
                # 打印一下顶层 key，方便你排查路径
                print(list(f.keys()))
                sys.exit(1)

            data = f[dataset_path][:]
            if len(data) == 0:
                print(f"错误：数据集 {dataset_path} 为空")
                sys.exit(1)

            # 如果是 (7,) 这种一维，就直接返回；如果是 (frames, 7)，取第一帧
            if data.ndim == 1:
                return data
            return data[0]

    except Exception as e:
        print(f"读取文件时出错: {e}")
        sys.exit(1)

def endpose_to_piper_units(endpose):
    """
    endpose: [x,y,z,rx,ry,rz,(gripper)]
    转换到 Piper SDK 需要的整数单位：
      - x,y,z: * 1e6
      - rx,ry,rz: * 57295.7795
      - gripper: * 1e6 (如果存在)
    """
    factor_xyz = 1_000_000
    factor_r = 57295.7795

    if len(endpose) < 6:
        raise ValueError(f"endpose 长度不足 6，实际为 {len(endpose)}：{endpose}")

    x, y, z, rx, ry, rz = endpose[:6]
    gripper = endpose[6] if len(endpose) > 6 else 0

    X = round(float(x) * factor_xyz)
    Y = round(float(y) * factor_xyz)
    Z = round(float(z) * factor_xyz)
    RX = round(float(rx) * factor_r)
    RY = round(float(ry) * factor_r)
    RZ = round(float(rz) * factor_r)
    G = round(float(gripper) * factor_xyz)

    return X, Y, Z, RX, RY, RZ, G

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piper 机械臂 HDF5 endPose 重现脚本")
    parser.add_argument("-f", "--file", required=True, help="HDF5 文件路径")
    parser.add_argument(
        "-d", "--dataset",
        default="arm/endPose/piper_end",
        help="endPose 数据集路径（默认: arm/endPose/piper_end）"
    )
    parser.add_argument(
        "--speed", type=int, default=100,
        help="运动速度（示例里 MotionCtrl_2 的第三个参数，默认 100）"
    )
    parser.add_argument(
        "--hold", type=float, default=3.0,
        help="到位后等待秒数（默认 3s）"
    )
    args = parser.parse_args()

    # 1) 读取 endpose（第一帧）
    endpose = get_first_endpose(args.file, args.dataset)
    print(f"成功读取第一帧 endpose: {endpose}")

    # 2) 初始化机械臂
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()

    while not piper.EnablePiper():
        print("正在等待机械臂使能...")
        time.sleep(0.01)

    # 初始化夹爪
    piper.GripperCtrl(0, 1000, 0x01, 0)

    # 3) 单位转换
    try:
        X, Y, Z, RX, RY, RZ, G = endpose_to_piper_units(endpose)
    except Exception as e:
        print(f"endpose 转换失败: {e}")
        sys.exit(1)

    print(f"转换后控制量: X={X}, Y={Y}, Z={Z}, RX={RX}, RY={RY}, RZ={RZ}, Gripper={G}")

    # 4) 执行动作：切换到末端位姿控制 + 下发 EndPoseCtrl
    # 你提供的示例：piper.MotionCtrl_2(0x01, 0x00, 100, 0x00, installation_pos=0x00)
    piper.MotionCtrl_2(0x01, 0x00, args.speed, 0x00, installation_pos=0x00)

    # 下发末端位姿
    piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

    # 夹爪（如果数据里有第7维，否则默认 0）
    piper.GripperCtrl(abs(G), 1000, 0x01, 0)

    # 5) 打印状态 & 等待
    print("机械臂当前末端状态:", piper.GetArmEndPoseMsgs())
    print("机械臂当前状态:", piper.GetArmStatus())

    time.sleep(args.hold)
