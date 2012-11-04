import sys
import re
import operator
import ast

from webob import Request
from webob.exc import HTTPException
from pkg_resources import iter_entry_points

from pydap.responses.lib import load_responses
from pydap.responses.error import ErrorResponse
from pydap.parsers import parse_ce
from pydap.exceptions import ConstraintExpressionError, ExtensionNotSupportedError
from pydap.lib import walk, fix_shorthand, get_var
from pydap.model import *


def load_handlers():
    return [ep.load() for ep in iter_entry_points("pydap.handler")]


def get_handler(filepath, handlers=None):
    # Check each handler to see which one handles this file.
    for handler in handlers or load_handlers():
        p = re.compile(handler.extensions)
        if p.match(filepath):
            return handler(filepath)

    raise ExtensionNotSupportedError(
            'No handler available for file {filepath}.'.format(filepath=filepath))


class BaseHandler(object):
    """
    Base class for Pydap handlers.

    Handlers are WSGI applications that parse the client request and build the
    corresponding dataset. The dataset is passed to proper Response (DDS, DAS,
    etc.)

    """

    # load all available responses
    responses = load_responses()

    def __init__(self, dataset=None):
        self.dataset = dataset
        self.additional_headers = []

    def __call__(self, environ, start_response):
        req = Request(environ)
        path, response = req.path.rsplit('.', 1)
        if response == 'das':
            req.query_string = ''
        projection, selection = parse_ce(req.query_string)

        try:
            # build the dataset and pass it to the proper response, returning a 
            # WSGI app
            dataset = self.parse(projection, selection)
            app = self.responses[response](dataset)
            app.close = self.close

            # now build a Response and set additional headers
            res = req.get_response(app)
            for key, value in self.additional_headers:
                res.headers.add(key, value)

            return res(environ, start_response)
        except HTTPException, exc:
            # HTTP exceptions are used to redirect the user
            return exc(environ, start_response)
        except:
            # should the exception be catched?
            # http://wsgi.readthedocs.org/en/latest/specifications/throw_errors.html
            if environ.get('x-wsgiorg.throw_errors'):
                raise
            else:
                res = ErrorResponse(info=sys.exc_info())
                return res(environ, start_response)

    def parse(self, projection, selection):
        """
        Parse the constraint expression.

        """
        if self.dataset is None:
            raise NotImplementedError(
                "Subclasses must define a dataset attribute pointing to a DatasetType.")

        # make a copy of the dataset, so we can filter sequences inplace
        dataset = self.dataset.clone()

        # apply the selection to the dataset, inplace
        apply_selection(selection, dataset)

        # fix projection
        if projection:
            projection = fix_shorthand(projection, dataset)
        else:
            projection = [[(key, ())] for key in dataset.keys()]
        out = apply_projection(projection, dataset)

        return out

    def close(self):
        pass


def apply_selection(selection, dataset):
    """
    Apply a given selection to a dataset, modifying it inplace.

    """
    for seq in walk(dataset, SequenceType):
        # apply only relevant selections
        conditions = [condition for condition in selection
                if re.match('%s\.[^\.]+(<=|<|>=|>|=|!=)' % re.escape(seq.id), condition)]
        for condition in conditions:
            id1, op, id2 = parse_selection(condition, dataset)
            seq.data = seq[ op(id1, id2) ].data
    return dataset


def apply_projection(projection, dataset):
    """
    Apply a given projection to a dataset.

    The function returns a new dataset object, after applying the projection to
    the original dataset.

    """
    out = DatasetType(name=dataset.name, attributes=dataset.attributes)

    for var in projection:
        target, template = out, dataset
        while var:
            name, slice_ = var.pop(0)
            candidate = template[name]
            
            # apply slice
            if slice_:
                if isinstance(candidate, BaseType):
                    candidate.data = candidate[slice_]
                elif isinstance(candidate, SequenceType):
                    candidate = candidate[slice_[0]]
                elif isinstance(candidate, GridType):
                    candidate = candidate[slice_]

            # handle structures
            if isinstance(candidate, StructureType):
                # add variable to target
                if name not in target.keys():
                    if var:
                        # if there are more children to add we need to clear the
                        # candidate so it has only explicitly added children; 
                        # also, Grids are degenerated into Structures
                        if isinstance(candidate, GridType):
                            candidate = StructureType(candidate.name, candidate.attributes)
                        candidate._keys = []
                    target[name] = candidate
                target, template = target[name], template[name]
            else:
                target[name] = candidate

    # fix sequence data, including only variables that are in the sequence
    for seq in walk(out, SequenceType):
        seq.data = get_var(dataset, seq.id)[tuple(seq.keys())].data

    return out


def parse_selection(expression, dataset):
    """
    Parse a selection expression into its elements.

    This function will parse a selection expression into three tokens: two
    variables or values and a comparison operator. Variables are returned as 
    Pydap objects from a given dataset, while values are parsed using
    `ast.literal_eval`.

    """
    id1, op, id2 = re.split('(<=|>=|!=|=~|>|<|=)', expression, 1)

    op = {
        '<=': operator.le,
        '>=': operator.ge,
        '!=': operator.ne,
        '=': operator.eq,
        '>': operator.gt, 
        '<': operator.lt,
    }[op]

    try:
        id1 = get_var(dataset, id1)
    except:
        id1 = ast.literal_eval(id1)

    try:
        id2 = get_var(dataset, id2)
    except:
        id2 = ast.literal_eval(id2)

    return id1, op, id2


class ConstraintExpression(object):
    """
    An object representing a selection on a constraint expression.
    
    These can be accumulated and evaluated only once.
    
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)
        
    def __unicode__(self):
        return unicode(self.value)

    def __and__(self, other):
        """Join two CEs together."""
        return self.__class__(self.value + '&' + str(other))

    def __or__(self, other):
        raise ConstraintExpressionError('OR constraints not allowed in the Opendap specification.')
