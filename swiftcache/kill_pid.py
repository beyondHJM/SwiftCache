import os
import sys
sys.path.insert(0, '/home/admin/workspace/aop_lab/app_source/hujianmin/nano-vllm/')
import json
import signal
from swiftcache.global_config import global_config

# 假设 global_config 已经定义好
pid_dir = global_config['os_pid_dir']

if not os.path.isdir(pid_dir):
    raise ValueError(f"PID 目录不存在: {pid_dir}")

killed_pids = []
removed_files = []

for filename in os.listdir(pid_dir):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(pid_dir, filename)
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        pid = data.get("pid")
        if isinstance(pid, int):
            try:
                # 检查进程是否存在
                os.kill(pid, 0)
            except ProcessLookupError:
                print(f"[!] 进程 {pid} 不存在（可能已退出）")
            else:
                try:
                    # ✅ 强制杀掉进程
                    os.kill(pid, signal.SIGKILL)
                    killed_pids.append(pid)
                    print(f"[√] 已发送 SIGKILL 给进程 {pid}（来自 {filename}）")
                except Exception as e:
                    print(f"[!] 强制结束进程 {pid} 失败: {e}")
        else:
            print(f"[!] {filename} 中没找到有效的 PID: {pid}")

    except json.JSONDecodeError:
        print(f"[!] JSON 解析失败: {file_path}")
    except Exception as e:
        print(f"[!] 处理 {file_path} 出错: {e}")
    finally:
        # ✅ 强制删除文件
        try:
            os.remove(file_path)
            removed_files.append(filename)
            print(f"[√] 已删除: {file_path}")
        except Exception as e:
            print(f"[!] 删除 {file_path} 失败: {e}")

print("========== 总结 ==========")
print(f"已强制杀掉的进程 PID: {killed_pids if killed_pids else '无'}")
print(f"已删除的 JSON 文件: {removed_files if removed_files else '无'}")
