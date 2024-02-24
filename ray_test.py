import ray
from ray.util.actor_pool import ActorPool
import numpy as np
import cProfile

@ray.remote
class Actortst:
    def run(self, idx):
        print("idxstart", idx)
        count = 0
        for i in range(100000):
            count+=1
        print("idxend", idx)
        return count, idx

def main():
    actorpool = ActorPool([Actortst.remote() for i in range(7)])

    for i in range(10):
        actor_generator = actorpool.map_unordered(
                    lambda a, v: a.run.remote(v),
                    np.arange(10),
                )

        num_finished = 0
        for count, idx in actor_generator:
            num_finished+=1
            print(idx, "returned", count)
            if num_finished == 7:
                break
        print("done!")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("example.prof")