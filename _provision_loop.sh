#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
for i in $(seq 1 40); do
  echo "[$(date +%H:%M:%S)] tentativa $i/40"
  colab new -s image_studio --gpu T4 >/dev/null 2>&1
  L=$(colab ls 2>&1 | grep -c "image_studio")
  if [ "$L" -gt 0 ]; then
    echo "SESSAO_CRIADA"
    colab ls 2>&1 | head -8
    exit 0
  fi
  d=$(( i * 15 ))
  [ $d -gt 300 ] && d=300
  sleep $d
done
echo "FIM_SEM_SESSAO"
