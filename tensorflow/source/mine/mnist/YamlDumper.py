#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper 

    
class YamlDumper():

    def __init__(self):
        self.raw_data = None
    
    def load_yaml(self, fname='default.yaml'):
        with open(fname) as yaml_file:
            self.raw_data = yaml.load(yaml_file.read())

        return self.raw_data

    def dump_yaml(self):
        return yaml.dump(self.raw_data)

    def get_parent(self, element_name=''):
        if element_name in self.raw_data:
            return self.raw_data[element_name]
        else:
            return None
        
    def get_child(self, raw_yaml, element_name=''):
        if element_name in raw_yaml:
            return raw_yaml[element_name]
        else:
            return None

    