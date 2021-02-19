import copy


class ReferencePoint(list):
    """A reference point exists in objective space an has a set of individuals
    associated to it."""
    def __init__(self, *args):
        list.__init__(self, *args)
        self.associations_count = 0
        self.associations = []


def generate_reference_points(num_objs, num_divisions_per_obj=4):
    """Generates reference points for NSGA-III selection. This code is based on
    `jMetal NSGA-III implementation <https://github.com/jMetal/jMetal>`_.
    """
    def gen_refs_recursive(work_point, num_objs, left, total, depth):
        if depth == num_objs - 1:
            work_point[depth] = left/total
            ref = ReferencePoint(copy.deepcopy(work_point))
            return [ref]
        else:
            res = []
            for i in range(left):
                work_point[depth] = i/total
                res = res + gen_refs_recursive(work_point, num_objs, left-i, total, depth+1)
            return res
    return gen_refs_recursive([0]*num_objs, num_objs, num_objs*num_divisions_per_obj,
                              num_objs*num_divisions_per_obj, 0)
