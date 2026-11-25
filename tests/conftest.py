"""試験の読み込みの道筋: 今の置き場と、前の試しの置き場（past/）の両方から部品を読めるようにする。"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for parts in (("tests", "past"), ("tests",), ("tools", "past"), ("intervention", "past"), ("tools",), ("intervention",)):
    p = os.path.abspath(os.path.join(ROOT, *parts))
    if p not in sys.path:
        sys.path.insert(0, p)
