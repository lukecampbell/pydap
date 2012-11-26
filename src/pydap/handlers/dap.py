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
            var.data = BaseProxy(url, var.id, var.descr)
        for var in walk(self.dataset, SequenceType):
            var.data = SequenceProxy(url, var.id, var.descr)


class BaseProxy(object):
    def __init__(self, baseurl, id, descr, slice_=None):
        self.baseurl = baseurl
        self.id = id
        self.dtype = np.dtype(descr[1])
        self.shape = descr[2]
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
        if isinstance(self.descr[1], list):
            id = ','.join('%s.%s' % (self.id, d[0]) for d in self.descr[1])
        else:
            id = self.id
        url = urlunsplit((
                scheme, netloc, path + '.dods',
                id + hyperslab(self.slice) + '&' + '&'.join(self.selection),
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

        return unpack_sequence(BufferedReader(r.raw), self.descr)

    def __getitem__(self, key):
        out = self.clone()

        # return the data for a children
        if isinstance(key, basestring):
            out.id = '{id}.{child}'.format(id=self.id, child=key)
            def get_child(descr):
                mapping = dict((d[0], d) for d in descr)
                return mapping[key]
            out.descr = apply_to_list(get_child, out.descr)

        # return a new object with requested columns
        elif isinstance(key, list):
            def get_children(descr):
                mapping = dict((d[0], d) for d in descr)
                return [mapping[k] for k in key]
            out.descr = apply_to_list(get_children, out.descr)

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


class BufferedReader(object):
    """
    A simple buffered reader for urllib3.HTTPResponse.

    """
    def __init__(self, raw):
        self.raw = raw
        self.buf = ''

    def read(self, n):
        buffered = len(self.buf)
        if n <= buffered:
            out, self.buf = self.buf[:n], self.buf[n:]
        else:
            out, self.buf = self.buf + self.raw.read(n-buffered), ''
        return out

    def peek(self, n):
        buffered = len(self.buf)
        if n > buffered:
            self.buf += self.raw.read(n-buffered)
        return self.buf[:n]


def apply_to_list(func, descr):
    """
    Apply a function to a list inside a dtype descriptor.

    """
    name, dtype, shape = descr
    if isinstance(dtype, list):
        dtype = func(dtype)
    else:
        dtype = apply_to_list(func, dtype)
    return name, dtype, shape


def unpack_sequence(buf, descr):
    """
    Unpack data from a sequence.

    """
    name, dtype, shape = descr

    # is this a sequence or a sequence child?
    sequence = isinstance(dtype, list)

    # if there are no strings and no nested sequences we can
    # unpack record by record easily
    simple = all(isinstance(d, basestring) and 'S' not in d for d in dtype)

    if simple:
        dtype = np.dtype(fix(dtype))
        marker = buf.read(4)
        while marker == START_OF_SEQUENCE:
            rec = np.fromstring(buf.read(dtype.itemsize), dtype=dtype)[0]
            if not sequence:
                rec = rec[0]
            yield rec
            marker = buf.read(4)
    else:
        marker = buf.read(4)
        while marker == START_OF_SEQUENCE:
            rec = unpack_children(buf, descr)
            if not sequence:
                rec = rec[0]
            else:
                rec = tuple(rec)
            yield rec
            marker = buf.read(4)


def unpack_children(buf, descr):
    """
    Unpack sequence children.

    """
    name, dtype, shape = descr

    out = []
    if not isinstance(dtype, list):
        dtype = [dtype]

    for descr in dtype:
        name, d, shape = descr
        d = np.dtype(fix(d))
        if d.char == 'S':
            n = np.fromstring(buf.read(4), '>I')[0]
            out.append(buf.read(n))
            buf.read(-n % 4)
        elif d.char == 'V':
            if buf.peek(4) == START_OF_SEQUENCE:
                out.append(tuple(unpack_sequence(buf, descr)))
            else:
                out.append(tuple(unpack_children(buf, descr)))
        elif d.char == 'B':
            data = np.fromstring(buf.read(1), d)[0]
            buf.read(3)
            out.append(data)
        else:
            if shape:
                n = np.fromstring(buf.read(4), '>I')[0]
                buf.read(4)
                data = np.fromstring(buf.read(d.itemsize*n), d).reshape(shape)
            else:
                data = np.fromstring(buf.read(d.itemsize), d)[0]
            out.append(data)
    return out


def fix(descr):
    """
    Numpy dtypes must be list of tuples, but we use single tuples to 
    differentiate children from sequences with only one child, ie,
    sequence['foo'] from sequence[['foo']]. Here we convert descr to
    a proper dtype.

    """
    if isinstance(descr, tuple):
        name, dtype, shape = descr
        return [ (name, fix(dtype), shape) ]
    else:
        return descr


if __name__ == '__main__':
    seq = SequenceProxy('http://sfbeams.sfsu.edu:8080/opendap/sfbeams/data_met/real-time/sfb_MET_PUF.dat', 'MET-REALTIME_CSV',
            ('MET-REALTIME_CSV', [('Month', '>i', ()), ('Day', '>i', ()), ('Year', '>i', ()), ('Hour', '>i', ()), ('Min', '>i', ()), ('Sec', '>i', ()), ('Air_Temp', '>f', ()), ('RH', '>f', ()), ('Pres', '>f', ()), ('Sol', '>f', ()), ('PAR', '>f', ()), ('Rain', '>f', ()), ('Wspd_S', '>f', ()), ('Wspd_U', '>f', ()), ('Wdir_DU', '>f', ()), ('Wdir_SDU', '>f', ()), ('Wspd_MAX', '>f', ()), ('InstSN', '>i', ())], ()))
    for i, rec in enumerate(seq[['Day', 'Month', 'Year']]):
        print rec
        if i>5: break
    for i, rec in enumerate(seq['Day']):
        print rec
        if i>5: break

    seq = SequenceProxy('http://test.opendap.org:8080/dods/dts/test.07', 'types', 
            ('types', [('b', 'B', ()), ('i32', '>i', ()), ('ui32', '>I', ()), ('i16', '>i', ()), ('ui16', '>I', ()), ('f32', '>f', ()), ('f64', '>d', ()), ('s', 'S', ()), ('u', 'S', ())], ()))
    for rec in seq:
        print rec
    for rec in seq['s']:
        print rec

    seq = SequenceProxy('http://test.opendap.org:8080/dods/dts/NestedSeq', 'person1',
            ('person1', [('age', '>i', ()), ('stuff', [('foo', '>i', ())], ())], ()))
    for rec in seq:
        print rec
    for rec in seq['age']:
        print rec
    for rec in seq['stuff']:
        print rec
    for rec in seq['stuff']['foo']:
        print rec

    seq = SequenceProxy('http://www.ocdb.csdb.cn:9091/dods/Chemistry/200509KFHC_nutrient.cdp', 'location',
            ('location', [('lon', '>f', ()), ('time', '>d', ()), ('lat', '>f', ()), ('_id', '>i', ()), ('profile', [('NH3-N', '>f', ()), ('SiO3-', '>f', ()), ('PO4-P', '>f', ()), ('NO2-N', '>f', ()), ('NO3-N', '>f', ()), ('depth', '>f', ())], ()), ('attributes', [('CAST', '|S1', ()), ('COORD_SYSTEM', '|S1', ()), ('NODC-COUNTRYCODE', '|S1', ()), ('Conventions', '|S1', ()), ('INST_TYPE', '|S1', ()), ('DATA_CMNT', '|S1', ()), ('DATA_ORIGIN', '|S1', ()), ('CREATION_DATE', '|S1', ()), ('DATA_SUBTYPE', '|S1', ()), ('DATA_TYPE', '|S1', ()), ('OCL-STATION-NUM', '|S1', ()), ('BOTTLE', '|S1', ())], ()), ('variable_attributes', [('depth', [('valid_range', '>f', (2,))], ())], ())], ()))
    for rec in seq:
        print rec

