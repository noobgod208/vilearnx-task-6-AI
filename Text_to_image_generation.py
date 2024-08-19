import pygame
import random
import numpy as np

# Initialize pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)
BLACK = (0, 0, 0)

# Game parameters
SNAKE_BLOCK = 10
SNAKE_SPEED = 45  # Increased speed

# Set the display to windowed mode with a fixed size
WIDTH, HEIGHT = 800, 600  # Moderately larger window
display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Snake Game with Q-learning')

# Clock
clock = pygame.time.Clock()

# Q-learning parameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# Initialize Q-table
q_table = {}

# Initialize generation counter and high score
generation_counter = 0
high_score = 0

# Cooldown settings
cooldown_time = 67  # Reduced cooldown time for increased speed
last_move_time = 0

# Exploration settings
visited_positions = set()  # Track visited positions


# Initialize the game state
def initialize_game():
    snake = [(WIDTH // 2, HEIGHT // 2)]
    direction = random.choice(ACTIONS)
    food = get_random_food_position(snake)
    visited_positions.clear()
    visited_positions.add(snake[0])
    return snake, direction, food


# Get random position for food
def get_random_food_position(snake):
    while True:
        x = random.randint(0, (WIDTH - SNAKE_BLOCK) // SNAKE_BLOCK) * SNAKE_BLOCK
        y = random.randint(0, (HEIGHT - SNAKE_BLOCK) // SNAKE_BLOCK) * SNAKE_BLOCK
        if (x, y) not in snake:
            return (x, y)


# Calculate new position based on action
def get_new_position(position, action):
    x, y = position
    if action == "UP":
        y -= SNAKE_BLOCK
    elif action == "DOWN":
        y += SNAKE_BLOCK
    elif action == "LEFT":
        x -= SNAKE_BLOCK
    elif action == "RIGHT":
        x += SNAKE_BLOCK
    return x, y


# Check if the snake has collided with the walls or itself
def is_collision(snake):
    head = snake[0]
    if head[0] >= WIDTH or head[0] < 0 or head[1] >= HEIGHT or head[1] < 0:
        return True
    if head in snake[1:]:
        return True
    return False


# Get state representation
def get_state(snake, food):
    head = snake[0]
    food_direction = (
        1 if food[0] > head[0] else -1 if food[0] < head[0] else 0,
        1 if food[1] > head[1] else -1 if food[1] < head[1] else 0,
    )

    if len(snake) > 1:
        body_direction = (
            1 if snake[1][0] > head[0] else -1 if snake[1][0] < head[0] else 0,
            1 if snake[1][1] > head[1] else -1 if snake[1][1] < head[1] else 0,
        )
    else:
        body_direction = (0, 0)

    return (food_direction, body_direction)


# Get action from Q-table or move toward food
def get_action(state, epsilon):
    if state not in q_table:
        q_table[state] = {a: 0 for a in ACTIONS}  # Initialize the state in the Q-table

    if np.random.rand() < epsilon:
        food_direction = state[0]
        if food_direction == (1, 0):
            return "RIGHT"
        elif food_direction == (-1, 0):
            return "LEFT"
        elif food_direction == (0, 1):
            return "DOWN"
        elif food_direction == (0, -1):
            return "UP"
    return max(q_table[state], key=q_table[state].get)


# Update Q-table
def update_q_table(state, action, reward, next_state):
    if state not in q_table:
        q_table[state] = {a: 0 for a in ACTIONS}
    if next_state not in q_table:
        q_table[next_state] = {a: 0 for a in ACTIONS}
    old_value = q_table[state][action]
    next_max = max(q_table[next_state].values())
    q_table[state][action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)


# Main game loop
def main():
    global EPSILON, generation_counter, high_score, last_move_time
    while True:
        snake, direction, food = initialize_game()
        score = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # Cooldown mechanism
            current_time = pygame.time.get_ticks()
            if current_time - last_move_time >= cooldown_time:
                # Get current state and action
                state = get_state(snake, food)
                action = get_action(state, EPSILON)
                last_move_time = current_time

                # Update snake position
                new_head = get_new_position(snake[0], action)
                snake = [new_head] + snake[:-1]

                # Check for collision
                if is_collision(snake):
                    reward = -100  # Higher penalty for collisions
                    update_q_table(state, action, reward, None)
                    generation_counter += 1
                    break

                # Check if food is eaten
                if snake[0] == food:
                    snake.append(snake[-1])
                    food = get_random_food_position(snake)
                    reward = 100  # Higher reward for eating food
                    score += 1
                else:
                    # Penalty for moving to a previously visited position
                    if snake[0] not in visited_positions:
                        reward = 0.1
                        visited_positions.add(snake[0])
                    else:
                        reward = -0.1  # Small penalty to encourage exploration

                # Update Q-table
                next_state = get_state(snake, food)
                update_q_table(state, action, reward, next_state)

                # Decay epsilon
                EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

            # Draw everything
            display.fill(BLACK)
            for block in snake:
                pygame.draw.rect(display, GREEN, [block[0], block[1], SNAKE_BLOCK, SNAKE_BLOCK])
            pygame.draw.rect(display, RED, [food[0], food[1], SNAKE_BLOCK, SNAKE_BLOCK])

            # Display generation counter, high score, and current score
            font = pygame.font.SysFont(None, 35)
            gen_text = font.render(f"Generation: {generation_counter}", True, WHITE)
            high_score_text = font.render(f"High Score: {high_score}", True, WHITE)
            current_score_text = font.render(f"Current Score: {score}", True, WHITE)
            display.blit(gen_text, [WIDTH - 240, 10])
            display.blit(high_score_text, [10, 10])
            display.blit(current_score_text, [WIDTH - 240, HEIGHT - 30])

            pygame.display.update()
            clock.tick(SNAKE_SPEED)

        # Update high score
        if score > high_score:
            high_score = score


if __name__ == "__main__":
    main()
