import sys
from cx_Freeze import setup, Executable


setup(name="Object Detection Software",
      version="1",
      description="Realtime Object Detection Software",
      executables=[Executable("main.py")]
      )

