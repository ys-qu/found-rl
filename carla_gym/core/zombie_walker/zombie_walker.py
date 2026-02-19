import numpy as np


class ZombieWalker(object):
    def __init__(self, walker_id, controller_id, world):

        self._walker = world.get_actor(walker_id)
        self._controller = world.get_actor(controller_id)

        self._controller.start()
        self._controller.go_to_location(world.get_random_location_from_navigation())
        self._controller.set_max_speed(1 + np.random.random())


    def clean(self):
        try:
            if self._controller is not None and self._controller.is_alive:
                self._controller.stop()
                self._controller.destroy()
        except RuntimeError:
            pass  # controller already dead or inaccessible

        try:
            if self._walker is not None and self._walker.is_alive:
                self._walker.destroy()
        except RuntimeError:
            pass  # walker already dead
