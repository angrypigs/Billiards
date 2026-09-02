from math import atan2, degrees, sqrt
from dataclasses import dataclass, field

import pygame
from pygame.math import Vector2

from src.ball import Ball
from src.utils import (
    ERROR_THRESHOLD,
    HEIGHT,
    HOLES,
    MAX_POWER,
    POCKET_RADIUS,
    RADIUS,
    WIDTH,
    estimate_normal,
)

@dataclass
class ShotEvents:
    pocketed_balls: list[int] = field(default_factory=list)

class PhysicsEngine:
    """Move balls and resolve collisions with balls, cushions and pockets."""

    def __init__(self, table_mask: pygame.mask.Mask) -> None:
        self.table_mask = table_mask
        self.shot_events = ShotEvents()

    def update(self, balls: list[Ball]) -> None:
        for ball in balls:
            if not ball.active or not ball.moving:
                continue

            ball.last_valid_coords = ball.coords.copy()
            ball.coords += ball.velocity
            self._resolve_table_collision(ball)

            ball.velocity *= 0.99
            if ball.velocity.magnitude() < 0.1:
                ball.moving = False
                continue

            for other_ball in balls:
                if ball != other_ball and other_ball.active:
                    self._resolve_ball_collision(ball, other_ball)

            self._recover_out_of_bounds_ball(ball)

    def _resolve_table_collision(self, ball: Ball) -> None:
        offset = (int(ball.coords.x - RADIUS), int(ball.coords.y - RADIUS))
        overlap = self.table_mask.overlap(ball.mask, offset)

        if overlap:
            nx, ny = estimate_normal(self.table_mask, overlap[0], overlap[1])
            vx, vy = ball.velocity
            dot = vx * nx + vy * ny
            ball.velocity.x = vx - 2 * dot * nx
            ball.velocity.y = vy - 2 * dot * ny

            px, py = ball.coords
            while self.table_mask.overlap(
                ball.mask,
                (int(px - RADIUS), int(py - RADIUS)),
            ):
                px -= nx * 0.5
                py -= ny * 0.5
            ball.coords.update(px, py)

        for hole_x, hole_y in HOLES:
            distance = sqrt(
                (hole_x - ball.coords.x) ** 2
                + (hole_y - ball.coords.y) ** 2
            )
            if distance >= POCKET_RADIUS:
                continue

            ball.velocity.update(0, 0)
            if ball.index == 0:
                ball.coords.update(WIDTH // 2, (HEIGHT - 100) // 2)
            else:
                self.shot_events.pocketed_balls.append(ball.index)
                ball.active = False
                ball.coords.update(-1000, -1000)
            break

    def _resolve_ball_collision(
        self,
        ball: Ball,
        other_ball: Ball,
    ) -> None:
        distance = ball.coords.distance_to(other_ball.coords)
        if distance > 2 * RADIUS:
            return

        collision_vector = other_ball.coords - ball.coords
        collision_angle = atan2(collision_vector.y, collision_vector.x)
        velocity = ball.velocity.rotate(-degrees(collision_angle))
        other_velocity = other_ball.velocity.rotate(-degrees(collision_angle))

        ball.velocity = Vector2(other_velocity.x, velocity.y).rotate(
            degrees(collision_angle)
        )
        other_ball.velocity = Vector2(velocity.x, other_velocity.y).rotate(
            degrees(collision_angle)
        )

        overlap = 2 * RADIUS - distance
        correction = collision_vector.normalize() * (overlap / 2)
        ball.coords -= correction
        other_ball.coords += correction
        other_ball.moving = True

    def _recover_out_of_bounds_ball(self, ball: Ball) -> None:
        if (
            10 <= ball.coords.x <= WIDTH - 10
            and 10 <= ball.coords.y <= HEIGHT - 10
        ):
            return

        print(f"Physics error: ball {ball} on {ball.coords}")
        ball.coords = ball.last_valid_coords.copy()
        if ball.velocity.magnitude() > MAX_POWER:
            ball.velocity.scale_to_length(MAX_POWER * 0.8)

        if (
            ball.coords.x < ERROR_THRESHOLD
            or ball.coords.x > WIDTH - ERROR_THRESHOLD
            or ball.coords.y < ERROR_THRESHOLD
            or ball.coords.y > HEIGHT - ERROR_THRESHOLD
        ):
            print(f"Critical physics error: ball {ball} reseted to table middle")
            ball.coords = Vector2(WIDTH // 2, (HEIGHT - 100) // 2)
            ball.velocity = Vector2(0, 0)
