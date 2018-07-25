import tensorflow as tf

def uniqueScope(name):
    vars = tf.global_variables()
    scope = [v for v in vars if v.name.startswith(name+'/')]

    if not scope:
        return name

    count = 0

    if name[-1] != '_':
        name += '_'
    name += str(count)

    while scope:
        count += 1

        name = list(name)
        name[-1] = str(count)
        name = "".join(name)

        scope = [v for v in vars if v.name.startswith(name+'/')]

    return name


def assignRNNTuple(sess,t1,t2):
    sess.run(t1.c.assign(t2.c))
    sess.run(t1.h.assign(t2.h))

class tempAssign:

    def __init__(self, session, tensors, values):

        self.session = session
        self.tensors = tensors
        self.values = values

        self.prev_c = session.run(tensors.c)
        self.prev_h = session.run(tensors.h)

    def __enter__(self):
        self.session.run([self.tensors.c.assign(self.values.c)])
        self.session.run([self.tensors.h.assign(self.values.h)])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.run([self.tensors.c.assign(self.prev_c)])
        self.session.run([self.tensors.h.assign(self.prev_h)])
