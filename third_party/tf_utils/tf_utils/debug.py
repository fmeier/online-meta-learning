
import sys
import traceback
import tensorflow as tf


def ipdb_raise():
    # Import here otherwise we have test issues.
    import ipdb
    ipdb.set_trace()


def ipdb_exception():
    # Import here otherwise we have test issues.
    import ipdb
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback, file=sys.stdout)
    traceback.print_exc(file=sys.stdout)
    traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
    _, _, tb = sys.exc_info()
    ipdb.post_mortem(tb)


def tf_debug_init():
    """Start queue runners and init all variables, this makes debugging easier.

    Example usage, would be in the function of choice call
    defines.debug_init()
    # Now you can call for any tensor you want tensor.eval()
    # and import ipdb;ipdb_set_trace() to get an interactive shell.
    """
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess=sess)


def create_debug_op(*args):
    def fn(*args):
        import ipdb
        ipdb.set_trace()
        return args[0]
    return tf.py_func(fn,
                      inp=[arg for arg in args],
                      Tout=args[0].dtype)
