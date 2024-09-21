import pygame
import sys

from snake.spirits.consts import BLACK, RED, GREEN, BLUE, WHITE
from snake.spirits.food import Food
from snake.spirits.obstacle import Obstacle
from snake.spirits.snake import Snake

# 初始化 pygame
pygame.init()

# 定义屏幕大小
screen_width = 640
screen_height = 480

class Game:
    def __init__(self, enable_ai_snakes=False, screen_width=640, screen_height=480):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.enable_ai_snakes = enable_ai_snakes
        pygame.display.set_caption('贪吃蛇游戏')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # 初始化玩家蛇
        self.player_snake = Snake(position=(100, 50), direction='RIGHT', color=GREEN)
        self.snakes = [self.player_snake]
        # 初始化 AI 蛇（如果需要多个蛇，可以添加更多 Snake 实例）
        if self.enable_ai_snakes:
            self.ai_snake = Snake(position=(500, 50), direction='LEFT', color=BLUE)
            # 所有的蛇
            self.snakes = [self.player_snake, self.ai_snake]
        # 初始化食物列表
        self.foods = [Food(self.screen_width, self.screen_height)]
        # 初始化障碍物列表
        self.obstacles = [Obstacle(position=(300, 200))]
        # 记录分数
        self.score = 0
        # 存储所有子弹
        self.all_bullets = []

    def handle_events(self):
        # 处理用户输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # 检测按键
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == ord('w'):
                    self.player_snake.change_direction('UP')
                if event.key == pygame.K_DOWN or event.key == ord('s'):
                    self.player_snake.change_direction('DOWN')
                if event.key == pygame.K_LEFT or event.key == ord('a'):
                    self.player_snake.change_direction('LEFT')
                if event.key == pygame.K_RIGHT or event.key == ord('d'):
                    self.player_snake.change_direction('RIGHT')
                if event.key == pygame.K_SPACE:
                    self.player_snake.shoot()

    def update(self):
        # 更新所有蛇
        for snake in self.snakes:
            snake.move()

            obstacle_positions = [obs.position for obs in self.obstacles]
            for s in self.snakes:
                if s != snake:
                    obstacle_positions.extend(s.positions)
                else:
                    obstacle_positions.extend(s.positions[1:])
            # 检查蛇是否吃到食物
            for food in self.foods:
                if snake.get_head_position() == food.position:
                    snake.grow()
                    if snake == self.player_snake:
                        self.score += 10
                    # 重新生成食物，避免与蛇和障碍物重叠
                    food.respawn(self.screen_width, self.screen_height, obstacle_positions+[food.position])

            # 是否与障碍物碰撞
            if snake.check_collision(obstacle_positions):
                if snake == self.player_snake:
                    self.game_over()
                else:
                    self.snakes.remove(snake)
                    self.score += 10
                    continue
            
            # 检查是否撞到边界
            if snake.is_off_screen(self.screen_width, self.screen_height):
                if snake == self.player_snake:
                    self.game_over()
                else:
                    self.snakes.remove(snake)
                    self.score += 10
                    continue

            self.update_bullets(snake)
        self.check_bullet_collisions()

    def update_bullets(self, snake):
        # 更新蛇的每一颗子弹
        for bullet in snake.bullets[:]:
            bullet.move()
            # 如果子弹移出屏幕，移除它
            if bullet.is_off_screen(self.screen_width, self.screen_height):
                snake.bullets.remove(bullet)
            else:
                # 将子弹添加到所有子弹列表中
                if bullet not in self.all_bullets:
                    self.all_bullets.append(bullet)

    def check_bullet_collisions(self):
        # 检查所有子弹与蛇的碰撞
        for bullet in self.all_bullets[:]:
            # 检查子弹是否击中障碍物
            for obstacle in self.obstacles:
                if pygame.Rect(bullet.position[0], bullet.position[1], bullet.size, bullet.size).collidepoint(obstacle.position):
                    # 子弹击中障碍物，移除子弹
                    self.all_bullets.remove(bullet)
                    if snake := bullet.owner():
                        snake.bullets.remove(bullet)
                    break  # 跳出障碍物循环

            # 检查子弹是否击中其他蛇
            for snake in self.snakes:
                if bullet.position in snake.positions:
                    # 子弹击中蛇，移除子弹，可以考虑减少蛇的长度或其他效果
                    self.all_bullets.remove(bullet)
                    if snake := bullet.owner():
                        snake.bullets.remove(bullet)
                    if snake == self.player_snake:
                        self.game_over()
                    self.snakes.remove(snake)  # 例如，移除被击中的蛇
                    break  # 跳出蛇循环

    def is_collision(self, position):
        # 检查特定位置是否有障碍
        for snake in self.snakes:
            if position in snake.positions:
                return True
        for obstacle in self.obstacles:
            if position == obstacle.position:
                return True
        for bullet in self.all_bullets[:]:
            if position == bullet.position:
                return True
        if position[0] < 0 or position[0] > self.screen_width or position[1] < 0 or position[1] > self.screen_height:
            return True
        return False

    def draw(self):
        self.screen.fill(WHITE)
        # 绘制蛇
        for snake in self.snakes:
            snake.draw(self.screen)
            # 绘制蛇的子弹
            for bullet in snake.bullets:
                bullet.draw(self.screen)
        # 绘制食物
        for food in self.foods:
            food.draw(self.screen)
        # 绘制障碍物
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        # 显示分数
        self.show_score()
        pygame.display.flip()

    def show_score(self):
        font = pygame.font.SysFont('arial', 24)
        score_surface = font.render(f'Score: {self.score}', True, BLACK)
        self.screen.blit(score_surface, (10, 10))

    def game_over(self):
        font = pygame.font.SysFont('arial', 54)
        game_over_surface = font.render('游戏结束', True, RED)
        game_over_rect = game_over_surface.get_rect()
        game_over_rect.midtop = (self.screen_width / 2, self.screen_height / 4)
        self.screen.blit(game_over_surface, game_over_rect)
        self.show_score()
        pygame.display.flip()
        pygame.time.wait(3000)
        self.reset()

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(15)
