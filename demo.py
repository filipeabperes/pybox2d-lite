import torch
import pygame

from arbiter import Arbiter
from body import Body, DTYPE
from joint import Joint
from world import World


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
PPM = 10.0
SCREEN_OFFSET_X = int(SCREEN_WIDTH / 2 / PPM)
# SCREEN_OFFSET_Y = int(SCREEN_HEIGHT / 2 / PPM)
# SCREEN_OFFSET_X = 0.0
SCREEN_OFFSET_Y = 0.0

TARGET_FPS = 30
TIMESTEP = 1.0 / TARGET_FPS
IMPULSE_ITERATIONS = 10

GRAVITY = torch.tensor([0.0, -10.0], dtype=DTYPE)


def to_screen(vertices):
    return [(int((SCREEN_OFFSET_X + v[0]) * PPM),
             SCREEN_HEIGHT - int((SCREEN_OFFSET_Y + v[1]) * PPM))
            for v in vertices]


# def to_world(vertices):
#     return [(float(v[0]) / PPM - SCREEN_OFFSET_X,
#              (SCREEN_OFFSET_Y - float(v[1]) / PPM))
#             for v in vertices]


class Bomb(Body):
    pass


def draw_test(x: int, y: int, text: str) -> None:
    pass  # TODO


def draw_body(screen: pygame.Surface, body: Body) -> None:
    R = body.rotation_matrix()
    x = body.position
    h = 0.5 * body.width

    v1 = x + R @ -h
    v2 = x + R @ torch.tensor([h[0], -h[1]])
    v3 = x + R @ h
    v4 = x + R @ torch.tensor([-h[0], h[1]])
    vertices = to_screen((v1, v2, v3, v4))
    # print(sum((v1, v2, v3, v4)) / 4, '->', sum(torch.tensor(vertices)) / 4)

    if isinstance(body, Bomb):
        color = (102, 230, 102)
    else:
        color = (204, 204, 230)
    fill_color = (102, 102, 115)

    pygame.draw.polygon(screen, fill_color, vertices, 0)
    pygame.draw.polygon(screen, color, vertices, 1)


def draw_joint(screen: pygame.Surface, joint: Joint) -> None:
    b1 = joint.body_1
    b2 = joint.body_2

    R1 = b1.rotation_matrix()
    R2 = b2.rotation_matrix()

    x1 = b1.position
    p1 = x1 + R1 @ joint.local_anchor_1
    x1, p1 = to_screen((x1, p1))

    x2 = b2.position
    p2 = x2 + R2 @ joint.local_anchor_2
    x2, p2 = to_screen((x2, p2))

    color = (128, 128, 204)
    pygame.draw.line(screen, color, x1, p1)
    pygame.draw.line(screen, color, p1, x2)
    pygame.draw.line(screen, color, x2, p2)


def draw_arbiter(screen: pygame.Surface, arbiter: Arbiter) -> None:
    color = (255, 0, 0)
    for c in arbiter.contacts:
        p = to_screen([c.position])[0]
        pygame.draw.circle(screen, color, p, 3.0, 0)


def draw_world(screen: pygame.Surface, world: World) -> None:
    for body in world.bodies:
        draw_body(screen, body)

    for joint in world.joints:
        draw_joint(screen, joint)

    for arbiter in world.arbiters.values():
        draw_arbiter(screen, arbiter)


def launch_bomb(world: World) -> None:
    # remove previous bomb
    for i, b in enumerate(world.bodies):
        if isinstance(b, Bomb):
            del world.bodies[i]

    world.arbiters = {key: arb for key, arb in world.arbiters.items()
                      if not isinstance(arb.body_1, Bomb) and not isinstance(arb.body_2, Bomb)}

    bomb = Bomb([3.0, 3.0], 50.0)
    bomb.friction = 0.2
    bomb.position = torch.tensor([torch.rand(1) * 100 - 50, 50.0])
    bomb.rotation = torch.rand(1) * 3 - 1.5
    bomb.velocity = -1.5 * bomb.position
    bomb.angular_velocity = torch.rand(1) * 40 - 20

    world.add_body(bomb)


def main() -> None:
    """
    Main loop.

    Updates the world and then the screen.
    """

    world = demo_1()
    cur_demo = 1

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.Font(None, 16)

    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and
                                             event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    launch_bomb(world)
                elif event.key == pygame.K_1:
                    world = demo_1()
                    cur_demo = 1
                elif event.key == pygame.K_2:
                    world = demo_2()
                    cur_demo = 2
                elif event.key == pygame.K_3:
                    world = demo_3()
                    cur_demo = 3
                elif event.key == pygame.K_4:
                    world = demo_4()
                    cur_demo = 4
                # elif event.key == pygame.K_5:
                #     world = demo_5()
                #     cur_demo = 5
                # elif event.key == pygame.K_6:
                #     world = demo_6()
                # elif event.key == pygame.K_7:
                #     world = demo_7()
                # elif event.key == pygame.K_8:
                #     world = demo_8()
                # elif event.key == pygame.K_9:
                #     world = demo_9()

        screen.fill((0, 0, 0))
        text_line = 10

        # display fps
        fps = clock.get_fps()
        fps_str = f'{fps:.1f} FPS'
        fps_render = font.render(fps_str, True, pygame.Color('coral'))
        screen.blit(fps_render, (10, text_line))
        text_line += 15
        # print(fps_str, end='\r')

        # display information
        info_str = f'Press keys 1-4 for demos. Current demo: {cur_demo}.'
        info_render = font.render(info_str, True, pygame.Color('coral'))
        screen.blit(info_render, (10, text_line))
        text_line += 15
        info_str = f'Press SPACE to launch the bomb.'
        info_render = font.render(info_str, True, pygame.Color('coral'))
        screen.blit(info_render, (10, text_line))

        # step the world
        world.step(TIMESTEP)
        draw_world(screen, world)

        pygame.display.flip()
        clock.tick(TARGET_FPS)


def demo_1() -> World:
    world = World(gravity=GRAVITY, iterations=IMPULSE_ITERATIONS)

    body_1 = Body([100.0, 20.0])
    body_1.position = torch.tensor([0.0, 0.5 * body_1.width[1]])
    world.add_body(body_1)

    body_2 = Body([5.0, 5.0], 200.0)
    body_2.position = torch.tensor([0.0, 40.0])
    world.add_body(body_2)

    return world


def demo_2() -> World:
    world = World(gravity=GRAVITY, iterations=IMPULSE_ITERATIONS)

    body_1 = Body([100.0, 10.0])
    body_1.position = torch.tensor([0.0, 0.5 * body_1.width[1]])
    body_1.friction = 0.2
    body_1.rotation = 0.0
    world.add_body(body_1)

    body_2 = Body([5.0, 5.0], 100.0)
    body_2.position = torch.tensor([45.0, 60.0])
    body_2.friction = 0.2
    body_2.rotation = 0.0
    world.add_body(body_2)

    joint = Joint(body_1, body_2, body_2.position * torch.tensor([0.0, 1.0]))
    world.add_joint(joint)

    return world


def demo_3() -> World:
    world = World(gravity=GRAVITY, iterations=IMPULSE_ITERATIONS)

    body_1 = Body([100.0, 1.0])
    body_1.position = torch.tensor([0.0, 20.0])
    body_1.rotation = -0.25
    world.add_body(body_1)

    frictions = [0.75, 0.5, 0.35, 0.1, 0.0]
    for i, fric in enumerate(frictions):
        body = Body([3.0, 3.0], 25.0)
        body.friction = fric
        body.position = torch.tensor([-45.0 + 10.0 * i, 40.0])
        world.add_body(body)

    return world


def demo_4() -> World:
    world = World(gravity=GRAVITY, iterations=IMPULSE_ITERATIONS)

    body_1 = Body([100.0, 10.0])
    body_1.position = torch.tensor([0.0, 0.5 * body_1.width[1]])
    world.add_body(body_1)

    for i in range(8):
        body = Body([3.0, 3.0], 1.0)
        body.friction = 0.2
        x = torch.rand(1) - 0.5
        body.position = torch.tensor([x, 12.0 + 3.5 * i])
        world.add_body(body)

    return world


def demo_5() -> World:
    pass


def demo_6() -> World:
    pass


def demo_7() -> World:
    pass


def demo_8() -> World:
    pass


def demo_9() -> World:
    pass


if __name__ == '__main__':
    with torch.no_grad():
        main()
