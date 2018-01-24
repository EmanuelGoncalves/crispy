#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves


def bin_cnv(value, thresold):
    return str(int(value)) if value < thresold else '%s+' % thresold
