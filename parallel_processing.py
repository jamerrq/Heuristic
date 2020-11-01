import local_search
import multiprocessing as mp
import os


# Init pool
pool = mp.Pool(mp.cpu_count())


try:
    os.system('rm ./Solutions/*')
except Exception:
    pass


[pool.map(local_search.print_results, [i + 1 for i in range(20)])]


pool.close()
print('Done!')
