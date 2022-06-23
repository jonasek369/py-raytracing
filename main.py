import math
import random

import pygame
from numba import njit

vec2 = pygame.math.Vector2
vec3 = pygame.math.Vector3

resolution = width, height = (1280, 720)
clock = pygame.time.Clock()

screen = pygame.display.set_mode(resolution)
vision_surf = pygame.Surface(resolution)

DEFAULT_GREY = (24, 24, 24)
DRAW_COLOR = (255, 255, 255)
DRAW_COLOR2 = (0, 255, 0)

RAY_LEN = width


def angle_to_vec(angle):
    return vec2(math.cos(angle), math.sin(angle))


class Ray:
    def __init__(self, pos: vec2, angle: float or vec2):
        self.position = pos
        if isinstance(angle, vec2):
            self.direction = angle
        else:
            self.direction = angle_to_vec(angle)

    def get_point(self, distance: int = 1) -> vec2:
        x = self.position.x + self.direction.x * distance
        y = self.position.y + self.direction.y * distance
        return vec2(x, y)

    def lookat(self, vec):
        self.direction.x = vec.x - self.position.x
        self.direction.y = vec.y - self.position.y
        self.direction.normalize()

    def cast(self, _wall):
        x1 = _wall.a.x
        y1 = _wall.a.y
        x2 = _wall.b.x
        y2 = _wall.b.y

        x3 = self.position.x
        y3 = self.position.y
        x4 = self.position.x + self.direction.x
        y4 = self.position.y + self.direction.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if den == 0:
            return

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if 0 < t < 1 and u > 0:
            point = vec2()
            point.x = x1 + t * (x2 - x1)
            point.y = y1 + t * (y2 - y1)
            return point
        else:
            return

    def draw(self):
        pygame.draw.line(screen, DRAW_COLOR, self.position, self.get_point(250))


@njit(fastmath=True)
def cast(x1, y1, x2, y2, x3, y3, dirx, diry):
    x4 = x3 + dirx
    y4 = y3 + diry

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if den == 0:
        return

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    if 0 < t < 1 and u > 0:
        return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]
    else:
        return


class Boundary:
    def __init__(self, a: vec2, b: vec2):
        self.a = a
        self.b = b
        self.normal = vec2(self.a - self.b).normalize()

    def draw(self):
        pygame.draw.line(screen, DRAW_COLOR, self.a, self.b)


class RayCluster:
    def __init__(self):
        self.pos = vec2(width / 2, height / 2)
        self.rays = []
        for i in range(0, 360):
            self.rays.append(Ray(self.pos, math.radians(i)))

    def draw(self):
        pygame.draw.circle(screen, DRAW_COLOR, self.pos, 6)

    def change_pos(self, vec):
        self.pos = vec
        for i in self.rays:
            i.position = self.pos

    def look(self, walls, draw=False):
        point_list = []
        for ray in self.rays:
            closest = None
            record = width + height
            for wall in walls:
                pt = cast(wall.a.x, wall.a.y, wall.b.x, wall.b.y, ray.position.x, ray.position.y, ray.direction.x,
                          ray.direction.y)
                if pt:
                    d = self.pos.distance_to(pt)
                    if d < record:
                        record = d
                        closest = vec2(pt)
            if closest:
                if not record > RAY_LEN:
                    if draw:
                        pygame.draw.line(screen, DRAW_COLOR, self.pos, closest)
                    point_list.append(closest)
                else:
                    if draw:
                        pygame.draw.line(screen, DRAW_COLOR, self.pos, ray.get_point(RAY_LEN))
                    point_list.append(ray.get_point(RAY_LEN))
            else:
                if draw:
                    pygame.draw.line(screen, DRAW_COLOR, self.pos, ray.get_point(RAY_LEN))
                point_list.append(ray.get_point(RAY_LEN))
        return point_list


class WallRect:
    def __init__(self, pos: vec2, sizes: vec2):
        self.pos = pos
        self.size = sizes

    def get_boundaries(self):
        return [
            Boundary(vec2(self.pos.x, self.pos.y), vec2(self.pos.x + self.size.x, self.pos.y)),
            Boundary(vec2(self.pos.x, self.pos.y), vec2(self.pos.x, self.pos.y + self.size.y)),
            Boundary(vec2(self.pos.x, self.pos.y + self.size.y),
                     vec2(self.pos.x + self.size.x, self.pos.y + self.size.y)),
            Boundary(vec2(self.pos.x + self.size.x, self.pos.y + self.size.y),
                     vec2(self.pos.x + self.size.x, self.pos.y))
        ]


class LinkedRayList:
    def __init__(self, ray: Ray = None):
        self.next: Ray or None = None
        self.ray = ray

    def get_len(self):
        n = 0
        ray = self
        while True:
            ray = ray.next
            if ray is None:
                break
            n += 1
        return n
    
    def lookat(self, vec):
        self.ray.lookat(vec)

    def bounce(self, walls, depth):
        current_node = self
        while len(self) < depth:
            closest = None
            bounced_wall = None
            record = width + height

            ray = current_node.ray

            for wall in walls:
                pt = cast(wall.a.x, wall.a.y, wall.b.x, wall.b.y, ray.position.x, ray.position.y,
                          ray.direction.x,
                          ray.direction.y)
                if pt:
                    d = self.ray.position.distance_to(pt)
                    if d < record:
                        record = d
                        closest = vec2(pt)
                        bounced_wall = wall
            if closest and bounced_wall:
                current_node.next = LinkedRayList(Ray(closest, closest.reflect(bounced_wall.normal)))
                current_node = current_node.next
                pygame.draw.line(screen, DRAW_COLOR, ray.position, closest)
            else:
                pygame.draw.line(screen, DRAW_COLOR, ray.position, ray.get_point(RAY_LEN))
                current_node.next = None
                break
    
    def draw(self):
        n = 0
        ray = self
        pygame.draw.line(screen, DRAW_COLOR, ray.ray.position, ray.ray.get_point(RAY_LEN))
        while True:
            ray = ray.next
            if ray is None:
                break
            pygame.draw.line(screen, DRAW_COLOR, ray.ray.position, ray.ray.get_point(RAY_LEN))
            n += 1
        return n

    def __len__(self):
        return self.get_len()


walls = []

for i in range(10):
    ax = random.randint(0, width)
    ay = random.randint(0, height)
    bx = random.randint(0, width)
    by = random.randint(0, height)
    walls.append(Boundary(vec2(ax, ay), vec2(bx, by)))

wr = LinkedRayList(Ray(vec2(width / 2, height / 2), 70))

while True:
    dt = clock.tick(0)
    dt = dt / 1000
    screen.fill(DEFAULT_GREY)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit(0)

    for wall in walls:
        wall.draw()

    wr.bounce(walls, 12)
    wr.lookat(vec2(pygame.mouse.get_pos()))
    # wr.draw()

    pygame.display.flip()
    pygame.display.set_caption(str(clock.get_fps()))
