import pygame
import sys
import random


WIDTH, HEIGHT = 800, 600
AGENT_RADIUS = 15
AGENT_COLOR = (255, 0, 0)  # Red
WALL_COLOR = (0, 0, 0)  # Black
FPS = 10

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, walls):
       
        directions = [(0, 5), (0, -5), (5, 0), (-5, 0),
            (5, 5), (5, -5), (-5, 5), (-5, -5)  # (Moore neighborhood)
        ]
        dx, dy = random.choice(directions)

        # Calculate new position
        new_x = self.x + dx
        new_y = self.y + dy

        # Check for collision with walls
        for wall in walls:
            x1, y1, x2, y2 = wall
            if x1 <= new_x <= x2 and y1 <= new_y <= y2:
                return  # Don't move if there's a wall in the way

        # Ensure the agent stays within the screen boundaries
        new_x = max(AGENT_RADIUS, min(WIDTH - AGENT_RADIUS, new_x))
        new_y = max(AGENT_RADIUS, min(HEIGHT - AGENT_RADIUS, new_y))

        self.x = new_x
        self.y = new_y

    def draw(self, screen):
        pygame.draw.circle(screen, AGENT_COLOR, (self.x, self.y), AGENT_RADIUS)

class Environment:
    def __init__(self):
        self.walls = [] 

    def add_wall(self, x1, y1, x2, y2):
        self.walls.append((x1, y1, x2, y2))

    def draw_walls(self, screen):
        for wall in self.walls:
            pygame.draw.line(screen, WALL_COLOR, (wall[0], wall[1]), (wall[2], wall[3]), 5)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Agent Movement in Environment")
    clock = pygame.time.Clock()

    agent = Agent(WIDTH // 2+30, HEIGHT // 2)
    environment = Environment()

    # walls
    environment.add_wall(100, 100, 300, 100)
    environment.add_wall(400, 200, 400, 400)
    environment.add_wall(200, 400, 600, 400)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        agent.move(environment.walls)

        screen.fill((255, 255, 255))  # Clear the screen
        agent.draw(screen)
        environment.draw_walls(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
