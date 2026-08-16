# -*- coding: utf-8 -*-
"""Supervisor persistente e detached para universal_app.py (modo Colab CLI).

Fica rodando como processo setsid independente (sobrevive ao exec/kernel).
Loop: inicia o app, reinicia com backoff se cair, para com .stop_supervisor.
Log proprio em /content/sup.log
"""
import os, sys, subprocess, time

LOG = "/content/universal_app.log"
SUP_LOG = "/content/sup.log"
STOP = "/content/.stop_supervisor"
APP = "/content/universal_app.py"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    with open(SUP_LOG, "a", encoding="utf-8") as f:
        f.write(line)

def start_app():
    if os.path.exists(LOG):
        try:
            os.rename(LOG, LOG + ".prev")
        except Exception:
            pass
    logf = open(LOG, "w", encoding="utf-8")
    return subprocess.Popen([sys.executable, "-u", APP], cwd="/content",
                            stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)

def main():
    log("Supervisor persistente iniciado (setsid).")
    subprocess.run(["pkill", "-9", "-f", "universal_app.py"], capture_output=True)
    time.sleep(2)
    quick_fails = 0
    delay = 5
    p = None
    while True:
        if os.path.exists(STOP):
            try:
                os.remove(STOP)
            except Exception:
                pass
            log("Parada solicitada (.stop_supervisor).")
            if p is not None:
                p.terminate()
            break
        p = start_app()
        t0 = time.time()
        rc = p.wait()
        alive = time.time() - t0
        if alive >= 20:
            quick_fails = 0
            delay = 5
        else:
            quick_fails += 1
        log(f"app saiu rc={rc} viveu={int(alive)}s quick_fails={quick_fails} delay={delay}s")
        if quick_fails >= 3:
            log("CRASH LOOP (3x <20s). Tail do log do app:")
            try:
                tail = open(LOG, encoding="utf-8", errors="replace").read().splitlines()[-30:]
                for l in tail:
                    log("  " + l[:180])
            except Exception:
                pass
            break
        time.sleep(delay)
        delay = min(delay * 2, 60)
    log("Supervisor encerrado.")

if __name__ == "__main__":
    main()