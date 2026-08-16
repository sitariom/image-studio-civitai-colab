# -*- coding: utf-8 -*-
import subprocess, sys, time, re

def authenticate_with_code(code_str):
    code_str = code_str.strip()
    cmd = "/home/simoesfsa/.local/bin/colab sessions"
    p = subprocess.Popen(["wsl", "bash", "-c", cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Send code immediately when prompt appears
    out, err = p.communicate(input=code_str + "\n", timeout=30)
    print("STDOUT:", out)
    if err:
        print("STDERR:", err)
    return p.returncode

if __name__ == "__main__":
    if len(sys.argv) > 1:
        code = sys.argv[1]
        authenticate_with_code(code)
