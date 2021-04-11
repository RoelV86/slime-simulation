# References
# https://sagejenson.com/physarum
# https://www.youtube.com/watch?v=X-iSQQgOd1A&ab_channel=SebastianLague

import numpy as np
import imageio
from datetime import datetime
from tqdm import tqdm
from pygifsicle import gifsicle

SIZE = (500, 500)
N_SLIMES = 2**10  # 2**18 for size 1080, 1920
SPAWN_RADIUS = 50

LAT_SPEED = 3
ROT_SPEED = 0.4
ROT_RANDOM = 0.5
SENSE_ANGLE = np.deg2rad(20)
SENSE_DIST = 10

DECAY_RATE = 0.9
DECAY_LIMIT = 0.05

N_STEPS = 8000
UPDATE_TRAIL_FRAMES = 1
SKIP_FRAMES = 40

# Pre-calculations
ACCURACY = 3
SD_SIN = np.array([int(SENSE_DIST * np.sin(phi)) for phi in np.linspace(0, 2*np.pi, 1 + int(10**ACCURACY * 2 * np.pi))])
SD_COS = np.array([int(SENSE_DIST * np.cos(phi)) for phi in np.linspace(0, 2*np.pi, 1 + int(10**ACCURACY * 2 * np.pi))])
SIN = np.array([np.sin(phi) for phi in np.linspace(0, 2*np.pi, 1 + int(10**ACCURACY * 2 * np.pi))])
COS = np.array([np.cos(phi) for phi in np.linspace(0, 2*np.pi, 1 + int(10**ACCURACY * 2 * np.pi))])


class Colony:
    def __init__(self, n_slimes, trail_map, lat_speed, rot_speed, rot_random, sense_angle, sense_dist, spawn_radius):
        self.grid_shape = trail_map.values.shape

        self.x = None
        self.y = None
        self.phi = None

        self.spawn_radius = spawn_radius
        self.count = n_slimes
        self.spawn(n_slimes)

        self.lat_speed = lat_speed
        self.rot_speed = rot_speed
        self.rot_random = rot_random

        self.sensors = [-sense_angle, 0, sense_angle]
        self.sense_dist = sense_dist

        self.t1 = 0
        self.t2 = 0
        self.t3 = 0
        self.t4 = 0

    def spawn(self, n_slimes):
        x = (2 * np.random.rand(n_slimes)) - 1
        y = (2 * np.random.rand(n_slimes)) - 1
        valid_spawn = False

        while not valid_spawn:
            valid_spawn = True
            for i, _ in enumerate(x):
                r = np.sqrt(x[i]**2 + y[i]**2)

                if r > 1:
                    x[i] = (2 * np.random.rand()) - 1
                    y[i] = (2 * np.random.rand()) - 1
                    valid_spawn = False

        self.x = np.expand_dims((x * self.spawn_radius + self.grid_shape[0] / 2).astype(int), -1)
        self.y = np.expand_dims((y * self.spawn_radius + self.grid_shape[1] / 2).astype(int), -1)
        self.phi = np.expand_dims(np.random.rand(n_slimes) * 2 * np.pi, -1)

    def update(self, trail):
        # t0 = datetime.now()
        sensor_values = self.sense(trail)
        # t1 = datetime.now()
        self.turn(sensor_values)
        # t2 = datetime.now()
        self.move()
        # t3 = datetime.now()
        self.deposit(trail)
        # t4 = datetime.now()

        # self.t1 += (t1 - t0).total_seconds()
        # self.t2 += (t2 - t1).total_seconds()
        # self.t3 += (t3 - t2).total_seconds()
        # self.t4 += (t4 - t3).total_seconds()

    def sense(self, trail_map):
        phi = np.repeat(self.phi, 3, axis=1)

        phi[:, 1] += self.sensors[1]
        phi[:, 2] += self.sensors[2]

        phi = (10 ** ACCURACY*np.mod(phi, 2 * np.pi)).astype(int)
        x = (SD_SIN[phi] + self.x).clip(0, self.grid_shape[0]-1)
        y = (SD_COS[phi] + self.y).clip(0, self.grid_shape[1]-1)

        sensor_values = trail_map.values[x, y]

        return sensor_values

    def turn(self, sensor_values):
        direction = np.expand_dims(np.argmax(sensor_values, axis=1) - 1, -1)
        self.phi += (direction * self.rot_speed +
                     self.rot_random * 2 * np.expand_dims(np.random.rand(self.count) - 0.5, -1))

    def move(self):
        phi = (10 ** ACCURACY*np.mod(self.phi, 2 * np.pi)).astype(int)

        self.x += (SIN[phi] * self.lat_speed).astype(int)
        self.y += (COS[phi] * self.lat_speed).astype(int)

        # Teleport to other side if exiting the map
        self.x[self.x < 0] += self.grid_shape[0]
        self.x[self.x >= self.grid_shape[0]] -= self.grid_shape[0]

        self.y[self.y < 0] += self.grid_shape[1]
        self.y[self.y >= self.grid_shape[1]] -= self.grid_shape[1]

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
        self.values[x, y] = 1.0

    def diffuse_matrix(self):
        new_values = self.values.copy()

        new_values[:-1, :-1] += self.values[1:, 1:]   # Top left
        new_values[:-1, :] += self.values[1:, :]      # Top mid
        new_values[:-1, 1:] += self.values[1:, :-1]   # Top right

        new_values[:, :-1] += self.values[:, 1:]      # Mid left
        new_values[:, 1:] += self.values[:, :-1]      # Mid right

        new_values[1:, :-1] += self.values[:-1, 1:]   # Bottom left
        new_values[1:, :] += self.values[:-1, :]      # Bottom mid
        new_values[1:, 1:] += self.values[:-1, :-1]   # Bottom right

        self.values = new_values/9

    def decay(self):
        self.values *= self.decay_rate
        self.values[self.values < self.decay_limit] = 0


# Initialization
trail = Trail(SIZE, DECAY_RATE, DECAY_LIMIT)
colony = Colony(N_SLIMES, trail, LAT_SPEED, ROT_SPEED, ROT_RANDOM, SENSE_ANGLE, SENSE_DIST, SPAWN_RADIUS)
trail_save = np.zeros((int(N_STEPS/SKIP_FRAMES)+1, SIZE[0], SIZE[1]))
i_frame = 0

# Run simulation
t_colony = 0
t_trail = 0
t_save = 0

t_start = datetime.now()
for i in tqdm(range(N_STEPS)):
    colony.update(trail)

    if i % UPDATE_TRAIL_FRAMES == 0:
        trail.update()

    if i % SKIP_FRAMES == 0:
        trail_save[i_frame, :, :] = (255*trail.values)
        i_frame += 1

dt_simulation = (datetime.now()-t_start).total_seconds()
print(f'Time taken: {dt_simulation/N_STEPS}s per step')

# Visualize
trail_save = trail_save.clip(0, 255)
trail_save = trail_save.astype('uint8')

print('Saving static image')
imageio.imwrite('output.png', trail_save[int(0.75*len(trail_save)), :, :])

print('Saving gif image')
imageio.mimwrite('output.gif', trail_save)
gifsicle(sources='output.gif', optimize=True, colors=32)
