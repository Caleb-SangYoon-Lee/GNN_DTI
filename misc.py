# -*- coding: utf-8 -*-
import traceback

# 2020-03-23: added by caleb
def whoami():
    stack = traceback.extract_stack()
    file_name, codeline, func_name, text = stack[-2]
    return func_name
