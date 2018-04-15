import tensorflow as tf

def uniqeScope(name):
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