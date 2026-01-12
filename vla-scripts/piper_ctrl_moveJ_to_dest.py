import time
import argparse
import h5py
import sys
from piper_sdk import *

def get_first_qpos(file_path):
    """从HDF5文件中读取第一帧qpos数据"""
    try:
        with h5py.File(file_path, 'r') as f:
            # 获取 /observations/qpos 数据
            # 假设其形状为 (frames, joint_dims)
            qpos_data = f['observations/qpos'][:]
            if len(qpos_data) > 0:
                return qpos_data[0]
            else:
                print("错误：HDF5文件中的qpos数据为空")
                sys.exit(1)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 1. 配置命令行参数解析
    parser = argparse.ArgumentParser(description="Piper 机械臂 HDF5 轨迹重现脚本")
    parser.add_argument("-f", "--file", required=True, help="HDF5 文件的路径")
    args = parser.parse_args()

    # 2. 从文件获取 position
    # 得到的 position 通常是弧度值数组
    position = get_first_qpos(args.file)
    print(f"成功读取第一帧数据: {position}")

    # 3. 初始化机械臂
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    
    while(not piper.EnablePiper()):
        print("正在等待机械臂使能...")
        time.sleep(0.01)
    
    # 初始化抓取器
    piper.GripperCtrl(0, 1000, 0x01, 0)

    # 4. 数据转换逻辑
    # 注意：根据您的原始脚本，前6位是关节角，第7位通过 factor 转换，第8位是夹爪
    factor = 57295.7795
    joint_0 = round(position[0] * factor)
    joint_1 = round(position[1] * factor)
    joint_2 = round(position[2] * factor)
    joint_3 = round(position[3] * factor)
    joint_4 = round(position[4] * factor)
    joint_5 = round(position[5] * factor)
    
    # 原始脚本中 position 拼接了一个小量，这里假设 qpos 的最后一位是夹爪行程
    # 如果 qpos 只有7位，请根据实际数组长度调整索引
    gripper_val = round(position[6] * 1000000) if len(position) > 6 else 0

    # 5. 执行动作
    piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    piper.GripperCtrl(abs(gripper_val), 1000, 0x01, 0)

    # 6. 状态打印
    print("机械臂当前状态:", piper.GetArmStatus())