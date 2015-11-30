#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import math
import time

MAX_LEN = 30                    
def get_progressbar_str(progress):
    BAR_LEN = int(MAX_LEN * progress)
    return ('[' + '=' * BAR_LEN +
            ('>' if BAR_LEN < MAX_LEN else '') +
            ' ' * (MAX_LEN - BAR_LEN) +
            '] %.1f%%' % (progress * 100.))
            
def show_progress(now, end):
    sys.stderr.write('\r\033[K' + get_progressbar_str(1.0 * now / end))
    sys.stderr.flush()

def end_progress():
    sys.stderr.write('\n')
    sys.stderr.flush()

def show_progressxxx(now, end):
    sys.stderr.write('\r\033[K' + get_progressbar_str(1.0 * now / end) + ':::' + str(now) + '/' + str(end))
    sys.stderr.flush()
