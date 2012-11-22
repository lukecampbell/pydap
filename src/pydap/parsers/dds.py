import re
from urllib import unquote

import numpy as np

from pydap.parsers import SimpleParser
from pydap.model import *


typemap = {
    'byte'    : np.dtype('B'),
    'int'     : np.dtype('>i'),
    'uint'    : np.dtype('>I'),
    'int16'   : np.dtype('>i'),
    'uint16'  : np.dtype('>I'),
    'int32'   : np.dtype('>i'),
    'uint32'  : np.dtype('>I'),
    'float32' : np.dtype('>f'),
    'float64' : np.dtype('>d'),
    'string'  : np.dtype('S'),
    'url'     : np.dtype('S'),
    }
constructors = ('grid', 'sequence', 'structure')
name_regexp = '[\w%!~"\'\*-]+'


class DDSParser(SimpleParser):
    def __init__(self, dds):
        super(DDSParser, self).__init__(dds, re.IGNORECASE)
        self.dds = dds

    def consume(self, regexp):
        token = super(DDSParser, self).consume(regexp)
        self.buffer = self.buffer.lstrip()
        return token

    def parse(self):
        dataset = DatasetType('nameless')

        self.consume('dataset')
        self.consume('{')
        while not self.peek('}'):
            var = self.declaration()
            dataset[var.name] = var
        self.consume('}')

        dataset.name = unquote(self.consume('[^;]+'))
        dataset._set_id(dataset.name)
        self.consume(';')

        return dataset

    def declaration(self):
        token = self.peek('\w+').lower()

        map = {
               'grid'      : self.grid,
               'sequence'  : self.sequence,
               'structure' : self.structure,
               }
        method = map.get(token, self.base)
        return method()

    def base(self):
        type = self.consume('\w+')

        dtype = typemap[type.lower()]
        name = unquote(self.consume('[^;\[]+'))
        shape, dimensions = self.dimensions()
        data = Container(dtype=dtype, shape=shape)

        self.consume(';')

        return BaseType(name, data, dimensions)

    def dimensions(self):
        shape = []
        names = []
        while not self.peek(';'):
            self.consume('\[')
            token = self.consume(name_regexp)
            if self.peek('='):
                names.append(token)
                self.consume('=')
                token = self.consume('\d+')
            shape.append(int(token))
            self.consume('\]')
        return tuple(shape), tuple(names)

    def sequence(self):
        sequence = SequenceType('nameless')
        self.consume('sequence')
        self.consume('{')

        while not self.peek('}'):
            var = self.declaration()
            sequence[var.name] = var
        self.consume('}')

        sequence.name = unquote(self.consume('[^;]+'))
        self.consume(';')

        # build dtype from the str attribute if available, else
        # use the whole dtype since it's a list
        sequence.dtype = [(c.name, to_descr(c.dtype)) for c in sequence.children()]

        return sequence

    def structure(self):
        structure = StructureType('nameless')
        self.consume('structure')
        self.consume('{')

        while not self.peek('}'):
            var = self.declaration()
            structure[var.name] = var
        self.consume('}')

        structure.name = unquote(self.consume('[^;]+'))
        self.consume(';')

        structure.dtype = [(c.name, to_descr(c.dtype)) for c in structure.children()]

        return structure

    def grid(self):
        grid = GridType('nameless')
        self.consume('grid')
        self.consume('{')

        self.consume('array')
        self.consume(':')
        array = self.base()
        grid[array.name] = array

        self.consume('maps')
        self.consume(':')
        while not self.peek('}'):
            var = self.base()
            grid[var.name] = var
        self.consume('}')

        grid.name = unquote(self.consume('[^;]+'))
        self.consume(';')
        return grid


def to_descr(dtype):
    if isinstance(dtype, list):
        return dtype
    else:
        return dtype.str.replace('|S0', 'S')


class Container(object):
    """
    Placeholder container for data.

    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_dataset(dds):
    return DDSParser(dds).parse()
