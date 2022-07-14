import math

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
    def __init__(self, pos: vec2, angle: float or vec2, bounced_from=None):
        self.position = pos
        if isinstance(angle, vec2):
            self.direction = angle
        else:
            self.direction = angle_to_vec(angle)
        self.bounced_from = bounced_from

    def get_point(self, distance: int = 1) -> vec2:
        x = self.position.x + self.direction.x * distance
        y = self.position.y + self.direction.y * distance
        return vec2(x, y)

    def lookat(self, vec):
        x = vec.x - self.position.x
        y = vec.y - self.position.y
        if x == 0 and y == 0:
            self.direction = vec2(0, 0)
        else:
            self.direction = vec2(x, y).normalize()

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

    def draw(self, length):
        pygame.draw.line(screen, DRAW_COLOR, self.position, self.get_point(length))


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
        line = b - a
        self.normal = line.rotate(90).normalize()

    def draw(self):
        pygame.draw.line(screen, DRAW_COLOR2, self.a, self.b)


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
        self.next: LinkedRayList or None = None
        self.ray = ray

        self.__LOOP_LIMIT = 9999

    def get_len(self):
        node = self
        i = 1
        for i in range(1, self.__LOOP_LIMIT):
            node = node.next
            if node is None:
                return i
        return i

    def get_rays(self):
        node = self
        rays: [Ray] = [node.ray]
        for i in range(self.__LOOP_LIMIT):
            node = node.next
            if node is None:
                break
            rays.append(node.ray)
        return rays

    def look_at(self, vec):
        self.ray.lookat(vec)

    def bounce(self, walls, depth, draw=False):
        current_node = self
        for passes in range(depth):
            ray = current_node.ray

            if not ray:
                break

            closest = None
            record = width + height
            bounced_wall = None
            for wall in walls:
                if wall == ray.bounced_from:
                    continue

                pt = cast(wall.a.x, wall.a.y, wall.b.x, wall.b.y, ray.position.x, ray.position.y, ray.direction.x,
                          ray.direction.y)
                if pt:
                    d = ray.position.distance_to(pt)
                    if d < record:
                        record = d
                        closest = vec2(pt)
                        bounced_wall = wall
            if closest:
                current_node.next = LinkedRayList(
                    Ray(closest, ray.direction.reflect(bounced_wall.normal), bounced_wall))
                current_node = current_node.next
                if draw:
                    pygame.draw.line(screen, DRAW_COLOR, ray.position, closest)
            else:
                current_node.next = None
                if draw:
                    pygame.draw.line(screen, DRAW_COLOR, ray.position, ray.get_point(RAY_LEN))
                break

    def __len__(self):
        return self.get_len()


walls = []

walls.append(Boundary(vec2(100, 100), vec2(100, 900)))
walls.append(Boundary(vec2(600, 100), vec2(600, 300)))
walls.append(Boundary(vec2(100, 100), vec2(600, 100)))

wr = LinkedRayList(Ray(vec2(width / 2, height / 2), vec2(0, 0)))

MAX_RAYS = 64

last_press = 0

while True:
    dt = clock.tick(0)
    dt = dt / 1000
    screen.fill(DEFAULT_GREY)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit(0)

    for wall in walls:
        wall.draw()

    wr.look_at(vec2(pygame.mouse.get_pos()))
    wr.bounce(walls, MAX_RAYS, True)

    pygame.display.flip()
    pygame.display.set_caption(str(clock.get_fps()))
