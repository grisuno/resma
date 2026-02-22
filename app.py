#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: 22/11/2025
Licencia: GPL v3

Descripción:  RESMA 4.3.6
"""
import os

cmd = """
systemd-run --scope \
  -p MemoryMax=4G \
  -p MemoryHigh=3.5G \
  -p CPUWeight=50 \
  -p KillMode=mixed \
  -p TimeoutStopSec=30 \
  nice -n 15 python3 resma4.11.py

"""
os.system(cmd)
