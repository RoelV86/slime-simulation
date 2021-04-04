# References
# https://sagejenson.com/physarum
# https://www.youtube.com/watch?v=X-iSQQgOd1A&ab_channel=SebastianLague

import numpy as np
import imageio
from datetime import datetime
from tqdm import tqdm

SIZE = (600, 600)
N_SLIMES = 16000
SPAWN_RADIUS = 200

LAT_SPEED = 2
ROT_SPEED = 0.5  # (5/360) * 2 * np.pi
ROT_RANDOM = (20/360) * 2 * np.pi
SENSE_ANGLE = np.deg2rad(20)
SENSE_DIST = 10

DECAY_RATE = 0.9
DECAY_LIMIT = 0.05

N_STEPS = 200
UPDATE_TRAIL_FRAMES = 1
SKIP_FRAMES = 5


class Slime:
    def __init__(self, id, trail_map, lat_speed, rot_speed, rot_random, sense_angle, sense_dist, spawn_radius):
        self.id = id
        self.grid_shape = trail_map.values.shape

        self.x = 0
        self.y = 0
        self.spawn_radius = spawn_radius
        self.respawn()

        self.phi = np.random.rand()*2*np.pi

        self.lat_speed = lat_speed
        self.rot_speed = rot_speed
        self.rot_random = rot_random

        self.sensors = [0, -sense_angle, sense_angle]
        self.sense_dist = sense_dist

        self.t1 = 0
        self.t2 = 0
        self.t3 = 0
        self.t4 = 0

    def respawn(self):
        angle = np.random.rand() * 2 * np.pi
        radius = np.random.rand()*self.spawn_radius
        self.x = int(np.sin(angle)*radius + self.grid_shape[0]/2)
        self.y = int(np.cos(angle)*radius + self.grid_shape[1]/2)

    def update(self, trail, verbose=False):
        sensor_values = self.sense(trail, verbose)
        self.turn(sensor_values)
        self.move()
        self.deposit(trail)

    def sense(self, trail_map, verbose=False):
        sensor_values = np.zeros((3, ))
        for i, angle in enumerate(self.sensors):
            x = int(self.x + self.sense_dist * np.sin(self.phi + angle))
            y = int(self.y + self.sense_dist * np.cos(self.phi + angle))

            if 0 < x < self.grid_shape[0] and 0 < y < self.grid_shape[1]:
                sensor_values[i] = trail_map.values[(x, y)]

        if verbose:
            print(f'Slime {self.id} sensor values: {sensor_values}')

        return sensor_values

    def turn(self, sensor_values):
        direction = [0, -1, 1][np.argmax(sensor_values)]
        self.phi += (direction * self.rot_speed + self.rot_random * 2 * (np.random.rand() - 0.5))

    def move(self):
        self.x += int(np.sin(self.phi) * self.lat_speed)
        self.y += int(np.cos(self.phi) * self.lat_speed)

        if self.x < 0:
            self.x = 0
            self.phi = np.arctan2(-np.sin(self.phi), np.cos(self.phi))
        elif self.x >= self.grid_shape[0]:
            self.x = self.grid_shape[0]-1
            self.phi = np.arctan2(-np.sin(self.phi), np.cos(self.phi))

        if self.y < 0:
            self.y = 0
            self.phi = np.arctan2(np.sin(self.phi), -np.cos(self.phi))
        elif self.y >= self.grid_shape[1]:
            self.y = self.grid_shape[1]-1
            self.phi = np.arctan2(np.sin(self.phi), -np.cos(self.phi))

    def deposit(self, trail_map):
        trail_map.deposit(self.x, self.y)


class Trail:
    def __init__(self, shape, decay_rate, decay_limit):
        self.values = np.zeros(shape)
        self.decay_rate = decay_rate
        self.decay_limit = decay_limit

    def update(self):
        self.diffuse_matrix()
        self.decay()

    def deposit(self, x, y):
        self.values[(x, y)] += 0.3
        self.values[(x, y)] = max([self.values[(x, y)], 1])

    def diffuse_matrix(self):
        new_values = self.values.copy()

        new_values[:-1, :-1] += self.values[:-1, :-1]   # Top left
        new_values[:-1, :] += self.values[:-1, :]       # Top mid
        new_values[:-1, 1:] += self.values[:-1, 1:]     # Top right

        new_values[:, :-1] += self.values[:, :-1]       # Mid left
        new_values[:, 1:] += self.values[:, 1:]         # Mid right

        new_values[1:, :-1] += self.values[1:, :-1]     # Bottom left
        new_values[1:, :] += self.values[1:, :]         # Bottom mid
        new_values[1:, 1:] += self.values[1:, 1:]       # Bottom right

        self.values = new_values/9

    def decay(self):
        self.values *= self.decay_rate
        self.values[self.values < self.decay_limit] = 0


# Initialization
trail = Trail(SIZE, DECAY_RATE, DECAY_LIMIT)
slimes = [Slime(i, trail, LAT_SPEED, ROT_SPEED, ROT_RANDOM, SENSE_ANGLE, SENSE_DIST, SPAWN_RADIUS)
          for i in range(N_SLIMES)]
trail_save = np.zeros((N_STEPS, SIZE[0], SIZE[1]))

# Run simulation
t_slime = 0
t_trail = 0
t_save = 0

t_start = datetime.now()
for i in tqdm(range(N_STEPS)):
    t0 = datetime.now()

    for slime in slimes:
        slime.update(trail, verbose=False)
    t1 = datetime.now()

    if i % UPDATE_TRAIL_FRAMES == 0:
        trail.update()
    t2 = datetime.now()

    trail_save[i, :, :] = (255*trail.values)
    t3 = datetime.now()

    t_slime += (t1 - t0).total_seconds()
    t_trail += (t2 - t1).total_seconds()
    t_save += (t3 - t2).total_seconds()

dt_simulation = (datetime.now()-t_start).total_seconds()
print(f'Time taken: {dt_simulation/N_STEPS}s per step')

print(f'Slime update: {t_slime:.1f}s')
print(f'Trail update: {t_trail:.1f}s')
print(f'Trail save: {t_save:.1f}s')

# Visualize
trail_save = trail_save.clip(0, 255)
trail_save = trail_save.astype('uint8')

imageio.mimwrite('output.gif', trail_save[::SKIP_FRAMES, :, :])
