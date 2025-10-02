#!/usr/bin/env python3
import sys
import subprocess
import importlib

# 如需使用国内镜像，加上如下地址（不需要就设为 None）
# 例如清华镜像: "https://pypi.tuna.tsinghua.edu.cn/simple"
INDEX_URL = None  # 比如改为 "https://pypi.tuna.tsinghua.edu.cn/simple"

# pip 包名 -> 导入模块名（用于验证与打印版本）
PKGS = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "tqdm": "tqdm",
}

def run(cmd):
    print("> " + " ".join(cmd))
    subprocess.check_call(cmd)

def pip_install(args):
    cmd = [sys.executable, "-m", "pip"] + args
    if INDEX_URL:
        cmd += ["-i", INDEX_URL]
    run(cmd)

def main():
    print("Upgrading pip/setuptools/wheel ...")
    pip_install(["install", "-U", "pip", "setuptools", "wheel"])

    print("\nInstalling required packages ...")
    pip_install(["install", "-U"] + list(PKGS.keys()))

    print("\nVerifying installations and versions:")
    for pkg_name, module_name in PKGS.items():
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  {pkg_name} (module '{module_name}') version: {ver}")
        except Exception as e:
            print(f"  [ERROR] {pkg_name} not importable as '{module_name}': {e}")

    print("\nAll done!")

if __name__ == "__main__":
    main()