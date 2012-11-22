from urlparse import urlsplit, urlunsplit
import operator

import numpy as np
import requests

from pydap.model import *
from pydap.lib import encode, combine_slices, fix_slice, hyperslab, START_OF_SEQUENCE
from pydap.handlers.lib import ConstraintExpression, BaseHandler, walk
from pydap.parsers.dds import build_dataset
from pydap.parsers.das import parse_das


class DAPHandler(BaseHandler):
    def __init__(self, url):
        # download DDS/DAS
        scheme, netloc, path, query, fragment = urlsplit(url)
        ddsurl = urlunsplit((scheme, netloc, path + '.dds', query, fragment))
        dds = requests.get(ddsurl).text.encode('utf-8')
        dasurl = urlunsplit((scheme, netloc, path + '.das', query, fragment))
        das = requests.get(dasurl).text.encode('utf-8')

        # build the dataset from the DDS
        self.dataset = build_dataset(dds)

        # and add attributes from the DAS
        attributes = parse_das(das)
        for var in walk(self.dataset):
            # attributes can be flat, eg, "foo.bar" : {...}
            if var.id in attributes:
                var.attributes.update(attributes[var.id])
            # or nested, eg, "foo" : { "bar" : {...} }
            try:
                var.attributes.update(
                    reduce(operator.getitem, [attributes] + var.id.split('.')))
            except KeyError:
                pass

        # now add data proxies
        for var in walk(self.dataset, BaseType):
            var.data = BaseProxy(url, var.id, var.dtype, var.shape)
        for var in walk(self.dataset, SequenceType):
            var.data = SequenceProxy(url, var.id, var.dtype)


class BaseProxy(object):
    def __init__(self, baseurl, id, dtype, shape, slice_=None):
        self.baseurl = baseurl
        self.id = id
        self.dtype = dtype
        self.shape = shape
        self.slice = slice_ or (slice(None),)

    def __getitem__(self, index):
        # build download url
        index = combine_slices(self.slice, fix_slice(index, self.shape))
        scheme, netloc, path, query, fragment = urlsplit(self.baseurl)
        url = urlunsplit((
                scheme, netloc, path + '.dods',
                self.id + hyperslab(index) + '&' + query,
                fragment)).rstrip('&')

        # download and unpack data
        r = requests.get(url)
        dds, data = r.content.split('\nData:\n', 1)
        
        if self.shape:
            # skip size packing
            if self.dtype.char == 'S':
                data = data[4:]
            else:
                data = data[8:]

        # calculate array size
        shape = tuple((s.stop-s.start)/s.step for s in index)
        size = np.prod(shape)

        if self.dtype == np.byte:
            return np.fromstring(data[:size], 'B')
        elif self.dtype.char == 'S':
            out = []
            for word in range(size):
                n = np.fromstring(data[:4], '>I')  # read length
                data = data[4:]
                out.append(data[:n])
                data = data[n + (-n % 4):]
            return np.array(out, 'S')
        else:
            return np.fromstring(data, self.dtype).reshape(shape)

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        return iter(self[:])

    # Comparisons return a boolean array
    def __eq__(self, other): return self[:] == other
    def __ne__(self, other): return self[:] != other
    def __ge__(self, other): return self[:] >= other
    def __le__(self, other): return self[:] <= other
    def __gt__(self, other): return self[:] > other
    def __lt__(self, other): return self[:] < other


class SequenceProxy(object):

    shape = ()

    def __init__(self, baseurl, id, descr, selection=None, slice_=None):
        self.baseurl = baseurl
        self.id = id
        self.descr = descr
        self.selection = selection or []
        self.slice = slice_ or (slice(None),)

    def __iter__(self):
        scheme, netloc, path, query, fragment = urlsplit(self.baseurl)
        url = urlunsplit((
                scheme, netloc, path + '.dods',
                self.id + hyperslab(self.slice) + '&' + '&'.join(self.selection),
                fragment)).rstrip('&')

        # download and unpack data
        r = requests.get(url, prefetch=False)

        # strip dds response
        marker = '\nData:\n'
        buf = []
        while ''.join(buf) != marker:
            chunk = r.raw.read(1)
            if not chunk:
                break
            buf.append(chunk)
            buf = buf[-len(marker):]

        return unpack_sequence(r, self.descr)

    def __getitem__(self, key):
        out = self.clone()

        # return the data for a children
        if isinstance(key, basestring):
            out.id = '{id}.{child}'.format(id=self.id, child=key)
            out.descr = apply_to_list(
                    (lambda l, key=key: (key, dict(l)[key])),
                    out.descr)

        # return a new object with requested columns
        elif isinstance(key, list):
            out.descr = apply_to_list(
                    (lambda l, key=key: [(k, v) for k, v in l if k in key]),
                    out.descr)

        # return a copy with the added constraints
        elif isinstance(key, ConstraintExpression):
            out.selection.extend( str(key).split('&') )

        # slice data
        else:
            if isinstance(key, int):
                key = slice(key, key+1)
            out.slice = combine_slices(self.slice, (key,))

        return out

    def clone(self):
        return self.__class__(self.baseurl, self.id, self.descr,
                self.selection[:], self.slice[:])

    def __eq__(self, other): return ConstraintExpression('%s=%s' % (self.id, encode(other)))
    def __ne__(self, other): return ConstraintExpression('%s!=%s' % (self.id, encode(other)))
    def __ge__(self, other): return ConstraintExpression('%s>=%s' % (self.id, encode(other)))
    def __le__(self, other): return ConstraintExpression('%s<=%s' % (self.id, encode(other)))
    def __gt__(self, other): return ConstraintExpression('%s>%s' % (self.id, encode(other)))
    def __lt__(self, other): return ConstraintExpression('%s<%s' % (self.id, encode(other)))


def apply_to_list(func, descr):
    """
    Apply a function to a list inside a dtype descriptor.

    """
    if not isinstance(descr, tuple):
        descr = ('', descr)
        wrap = True
    else:
        wrap = False

    name, dtype = descr
    if isinstance(dtype, list):
        dtype = func(dtype)
    else:
        dtype = apply_to_list(func, dtype)

    if wrap:
        return dtype
    else:
        return name, dtype


def unpack_sequence(r, descr):
    """
    Unpack data from a sequence.

    """
    # numpy dtypes must be list of tuples, but we use single tuples to 
    # differentiate children from sequences with only one child, ie,
    # sequence['foo'] from sequence[['foo']]; here we convert descr to
    # a proper dtype
    sequence = isinstance(descr, list)
    def fix(descr):
        if isinstance(descr, tuple):
            return [ (descr[0], fix(descr[1])) ]
        else:
            return descr
    dtype = fix(descr)

    # if there are no strings and no nested sequences we can
    # unpack record by record easily
    simple = all(not isinstance(v, list) and 'S' not in v
            for k, v in dtype)
    if simple:
        dtype = np.dtype(dtype)
        marker = r.raw.read(4)
        while marker == START_OF_SEQUENCE:
            rec = np.fromstring(r.raw.read(dtype.itemsize), dtype=dtype)[0]
            if not sequence:
                rec = rec[0]
            yield rec
            marker = r.raw.read(4)
    else:
        marker = r.raw.read(4)
        while marker == START_OF_SEQUENCE:
            rec = []
            for name, d in dtype:
                d = np.dtype(d)
                if d.char == 'S':
                    n = np.fromstring(r.raw.read(4), '>I')[0]
                    rec.append(r.raw.read(n))
                    r.raw.read(-n % 4)
                elif d.char == 'V':
                    if isinstance(descr, list):
                        sub = [v for k, v in descr if k == name][0]
                    else:
                        sub = descr[1]
                    rec.append(tuple(unpack_sequence(r, sub)))
                elif d.char == 'B':
                    data = np.fromstring(r.raw.read(d.itemsize), d)[0]
                    r.raw.read(3)
                    rec.append(data)
                else:
                    data = np.fromstring(r.raw.read(d.itemsize), d)[0]
                    rec.append(data)
            if not sequence:
                rec = rec[0]
            yield tuple(rec)
            marker = r.raw.read(4)


if __name__ == '__main__':
    seq = SequenceProxy('http://test.opendap.org:8080/dods/dts/test.07', 'types', 
            [('b', 'B'), ('i32', '>i'), ('ui32', '>I'), ('i16', '>i'), ('ui16', '>I'), ('f32', '>f'), ('f64', '>d'), ('s', 'S'), ('u', 'S')])
    for rec in seq:
        print rec

    seq = SequenceProxy('http://test.opendap.org:8080/dods/dts/NestedSeq', 'person1',
            [('age', '>i'), ('stuff', [('foo', '>i')])])
    for rec in seq:
        print rec

