#!/bin/bash

if [ ! -d "./.venv" ]; then
	python3.12 -m venv .venv && .venv/bin/python3 -m pip install -r requirements.txt
fi

.venv/bin/python3 simulation.py
