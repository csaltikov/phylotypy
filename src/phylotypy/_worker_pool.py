import atexit
import multiprocessing as mp
import multiprocessing.pool

_pool = None


def get_pool(num_processes: int | None = None) -> mp.pool.Pool:
    """Returns a shared, lazily-created worker pool, reused for the life of the process.

    num_processes only takes effect on the call that first creates the pool;
    later calls reuse whatever pool already exists, regardless of the value passed.
    """
    global _pool
    if _pool is None:
        ctx = mp.get_context("spawn")
        _pool = ctx.Pool(num_processes or mp.cpu_count())
    return _pool


@atexit.register
def _shutdown():
    global _pool
    if _pool is not None:
        _pool.close()
        _pool.join()
        _pool = None
