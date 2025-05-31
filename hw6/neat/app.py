import game.wrapped_flappy_bird as game
import os
import sys
import pygame
from pygame.locals import *
import sys
from game.train import train
from game.run_best import run_best
import yaml

with open('params.yaml') as f:
    params = yaml.safe_load(f)

def show_game_over_menu(final_score):
    pygame.font.init()
    SCREEN = pygame.display.set_mode((288, 512))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 26)
    button_font = pygame.font.SysFont("Arial", 22)

    def draw_button(text, rect, color, text_color):
        pygame.draw.rect(SCREEN, color, rect, border_radius=8)
        label = button_font.render(text, True, text_color)
        label_rect = label.get_rect(center=rect.center)
        SCREEN.blit(label, label_rect)

    while True:
        SCREEN.fill((0, 0, 0))

        # Заголовок
        title = font.render(f"Game Over! Score: {final_score}", True, (255, 255, 255))
        SCREEN.blit(title, (20, 100))

        # Кнопки
        button_retry = pygame.Rect(44, 200, 200, 50)
        button_human = pygame.Rect(44, 270, 200, 50)
        button_quit  = pygame.Rect(44, 340, 200, 50)

        draw_button("Сыграть снова", button_retry, (70, 130, 180), (255, 255, 255))
        draw_button("Режим человека", button_human, (60, 160, 100), (255, 255, 255))
        draw_button("Выход", button_quit, (180, 50, 50), (255, 255, 255))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_retry.collidepoint(event.pos):
                    return 'AI'
                elif button_human.collidepoint(event.pos):
                    return 'human'
                elif button_quit.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()

        clock.tick(30)

def main():
    mode = params['game']['default_mode']
    config_path = params['neat']['config_path']
    while True:
        config_path = os.path.join(os.path.dirname(__file__), 'neat-config.txt')
        if mode == 'train':
            train(config_path)
            break
        elif mode == 'AI':
            score = run_best(config_path)
            mode = show_game_over_menu(score)
        else:
            game_state = game.GameState()
            while True:
                input_actions = [1, 0]
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                        input_actions = [0, 1]
                    else:
                        input_actions = [1, 0]

                _, _, done = game_state.frame_step(input_actions)
                if done:
                    mode = show_game_over_menu(0)
                    break


if __name__ == '__main__':
    main()
