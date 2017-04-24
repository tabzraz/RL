import numpy as np
import gym
from gym import spaces


class GridWorld(gym.Env):

    # --2d GridWorld--
    # 0 = Nothing
    # 1 = Wall
    # 2 = Goal
    # 3 = Player

    # Our wrapper handles the drawing
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self):
        self._reset()
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        # Time-limit on the environment, 5 is arbitrary
        self.limit = self.grid.size * 5
        self.positive_reward = +1
        self.negative_reward = -(self.goals) / self.limit

        # Gym Stuff
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.grid.shape)
        self.reward_range = (-1, +1)

    def _step(self, a):
        info_dict = {}
        self.steps += 1
        new_player_pos = (self.player_pos[0] + self.actions[a][0], self.player_pos[1] + self.actions[a][1])
        # Clip
        if new_player_pos[0] < 0 or new_player_pos[0] >= self.grid.shape[0]\
        or new_player_pos[1] < 0 or new_player_pos[1] >= self.grid.shape[1]:
            new_player_pos = self.player_pos

        r = self.negative_reward

        finished = False

        # Into a wall
        if self.grid[new_player_pos] == 1:
            new_player_pos = self.player_pos
        # Into a goal
        elif self.grid[new_player_pos] == 2:
            r += self.positive_reward
            self.goals -= 1
            if self.goals == 0:
                finished = True

        self.grid[self.player_pos] = 0
        self.grid[new_player_pos] = 3
        self.player_pos = new_player_pos

        if self.steps >= self.limit:
            finished = True
            info_dict["Steps_Termination"] = True

        return self.grid[:, :, np.newaxis] / 3, r, finished, info_dict

    def _reset(self):
        self.steps = 0
        self.create_grid()
        player_pos_np = np.argwhere(self.grid == 3)[0]
        self.player_pos = (player_pos_np[0], player_pos_np[1])
        self.goals = (self.grid == 2).sum()
        self.num_goals = self.goals
        self.goals_order = np.argwhere(self.grid == 2)
        # print(self.goals_order)
        return self.grid[:, :, np.newaxis] / 3

    def _render(self, mode="rgb_array", close=False):
        if mode == "rgb_array":
            grid = self.grid
            image = np.zeros(shape=(grid.shape[0], grid.shape[1], 3))
            for x in range(grid.shape[0]):
                for y in range(grid.shape[1]):
                    if grid[x, y] != 0:
                        image[x, y] = (255 * grid[x, y] / 3, 255 * grid[x, y] / 3, 255 * grid[x, y] / 3)
            return image
        else:
            pass
            # raise Exception("Cannot do human rendering")

    def create_grid(self):
        self.grid = np.array([[3, 0],
                              [1, 2]])

    def log_player_pos(self):
        goals_list = [self.grid[g[0], g[1]] == 2 for g in self.goals_order]
        # print(goals_list)
        player_pos = list(self.player_pos)
        joint = player_pos + goals_list
        # print(tuple(joint))
        return tuple(joint)

    def player_visits(self, player_visits, args):
        # Log the visitations
        with open("{}/logs/Player_Positions.txt".format(args.log_path), "w") as file:
            file.write('\n'.join(" ".join(str(x) for x in t) for t in player_visits))

        interval = int(args.t_max / args.interval_size)
        scaling = 2

        if self.num_goals > 3:
            raise Exception("Cant do state visitations for >3 goals atm")

        # We want to show visualisations for the agent depending on which goals they've visited as well
        # Keep it seperate from the other one
        colour_images = []
        for i in range(0, args.t_max, interval // 10):
            # Works for num_goals <= 3
            # print(self.grid.shape)
            canvas = np.zeros((self.grid.shape[0] * self.num_goals, self.grid.shape[1] * self.num_goals, 3))
            grid_x = self.grid.shape[0]
            grid_y = self.grid.shape[1]

            for visit in player_visits[i: i + interval]:
                px = visit[0]
                py = visit[1]

                np_goals = np.array(visit[2:])
                goal_colours = [ig for ig, c in enumerate(visit[2:]) if c == True]
                if goal_colours == []:
                    goal_colours = [0, 1, 2]
                x_place = px + grid_x * (np_goals == False).sum()
                yy = 0
                if (np_goals == False).sum() == 1:
                    yy = np.argwhere(np_goals == False)[0]
                elif (np_goals == False).sum() == 2:
                    yy = np.argwhere(np_goals == True)[0]
                y_place = py + grid_y * yy

                # print(x_place, y_place, goal_colours, canvas.shape)
                canvas[x_place, y_place, goal_colours] += 1

            if np.max(canvas) == 0:
                break
            canvas = canvas / (np.max(canvas) / scaling)

            # TODO: Colour the unvisited goals
            for goal in self.goals_order:
                canvas[goal[0], goal[1], :] = 2 / 3
            if self.num_goals >= 2:
                for g in range(self.num_goals):
                    for go_i, goal in enumerate(self.goals_order):
                        if go_i != g:
                            canvas[goal[0] + grid_x, goal[1] + g * grid_y, :] = 2 / 3
            if self.num_goals >= 3:
                for g in range(self.num_goals):
                    for go_i, goal in enumerate(self.goals_order):
                        if go_i == g:
                            canvas[goal[0] + 2 * grid_x, goal[1] + g * grid_y, :] = 2 / 3

            # The walls
            for x in range(grid_x):
                for y in range(grid_y):
                    if self.grid[x, y] == 1:
                        canvas[x, y, :] = 1 / 3
                        for zx in range(1, self.num_goals):
                            for zy in range(self.num_goals):
                                canvas[zx * grid_x + x, zy * grid_y + y, :] = 1 / 3

            # Seperate the mazes
            canvas = np.insert(canvas, [(z + 1) * grid_x for z in range(self.num_goals - 1)], 1, axis=0)
            canvas = np.insert(canvas, [(z + 1) * grid_x for z in range(self.num_goals - 1)], 1, axis=1)
            canvas[0:grid_x, grid_y + 1:, :] = 0
            colour_maze = canvas

            colour_maze = np.clip(colour_maze, 0, 1) * 255
            # colour_maze = np.swapaxes(colour_maze, 0, 1)
            colour_images.append(colour_maze.astype(np.uint8))
        return colour_images
        # save_video("{}/visitations/Goal_Visits__Interval_{}__T_{}".format(LOGDIR, interval_size, T), colour_images)


    def bonus_landscape(self, player_visits, exploration_bonuses, max_bonus, args):
        interval = int(args.t_max / args.interval_size)
        scaling = 2

        if self.num_goals > 3:
            raise Exception("Cant do bonus landscape for >3 goals atm")

        # We want to show visualisations for the agent depending on which goals they've visited as well
        # Keep it seperate from the other one
        colour_images = []
        for i in range(0, args.t_max, interval // 10):
            # Works for num_goals <= 3
            # print(self.grid.shape)
            canvas = np.zeros((self.grid.shape[0] * self.num_goals, self.grid.shape[1] * self.num_goals, 3))
            grid_x = self.grid.shape[0]
            grid_y = self.grid.shape[1]

            for visit, bonus in zip(player_visits[i: i + interval], exploration_bonuses[i: i + interval]):
                relative_bonus = bonus / max_bonus
                px = visit[0]
                py = visit[1]

                np_goals = np.array(visit[2:])
                goal_colours = [ig for ig, c in enumerate(visit[2:]) if c == True]
                if goal_colours == []:
                    goal_colours = [0, 1, 2]
                x_place = px + grid_x * (np_goals == False).sum()
                yy = 0
                if (np_goals == False).sum() == 1:
                    yy = np.argwhere(np_goals == False)[0]
                elif (np_goals == False).sum() == 2:
                    yy = np.argwhere(np_goals == True)[0]
                y_place = py + grid_y * yy

                # print(x_place, y_place, goal_colours, canvas.shape)
                canvas[x_place, y_place, goal_colours] = max(relative_bonus, canvas[x_place, y_place, goal_colours[0]])

            if np.max(canvas) == 0:
                break
            canvas = canvas / max_bonus

            # TODO: Colour the unvisited goals
            for goal in self.goals_order:
                canvas[goal[0], goal[1], :] = 2 / 3
            if self.num_goals >= 2:
                for g in range(self.num_goals):
                    for go_i, goal in enumerate(self.goals_order):
                        if go_i != g:
                            canvas[goal[0] + grid_x, goal[1] + g * grid_y, :] = 2 / 3
            if self.num_goals >= 3:
                for g in range(self.num_goals):
                    for go_i, goal in enumerate(self.goals_order):
                        if go_i == g:
                            canvas[goal[0] + 2 * grid_x, goal[1] + g * grid_y, :] = 2 / 3

            # The walls
            for x in range(grid_x):
                for y in range(grid_y):
                    if self.grid[x, y] == 1:
                        canvas[x, y, :] = 1 / 3
                        for zx in range(1, self.num_goals):
                            for zy in range(self.num_goals):
                                canvas[zx * grid_x + x, zy * grid_y + y, :] = 1 / 3

            # Seperate the mazes
            canvas = np.insert(canvas, [(z + 1) * grid_x for z in range(self.num_goals - 1)], 1, axis=0)
            canvas = np.insert(canvas, [(z + 1) * grid_x for z in range(self.num_goals - 1)], 1, axis=1)
            canvas[0:grid_x, grid_y + 1:, :] = 0
            colour_maze = canvas

            colour_maze = np.clip(colour_maze, 0, 1) * 255
            # colour_maze = np.swapaxes(colour_maze, 0, 1)
            colour_images.append(colour_maze.astype(np.uint8))
        return colour_images

# # Methods
# def environment_specific_stuff():
#     if args.env.startswith("Maze"):
#         player_pos = env.env.player_pos
#         player_pos_with_goals = (player_pos[0], player_pos[1], env.env.maze[3, -1] != 2, env.env.maze[-1, -4] != 2, env.env.maze[-4, 0] != 2)
#         Player_Positions.append(player_pos)
#         Player_Positions_With_Goals.append(player_pos_with_goals)
#         with open("{}/logs/Player_Positions_In_Maze.txt".format(LOGDIR), "a") as file:
#             file.write(str(player_pos) + "\n")
#         with open("{}/logs/Player_Positions_In_Maze_With_Goals.txt".format(LOGDIR), "a") as file:
#             file.write(str(player_pos_with_goals) + "\n")

#         # TODO: Move this out into a post-processing step
#         if T % int(args.t_max / 2) == 0:
#             # Make a gif of the positions
#             for interval_size in [2, 10]:
#                 interval = int(args.t_max / interval_size)
#                 scaling = 2
#                 images = []
#                 for i in range(0, T, interval // 10):
#                     canvas = np.zeros((env.env.maze.shape[0], env.env.maze.shape[1]))
#                     for visit in Player_Positions[i: i + interval]:
#                         canvas[visit] += 1
#                     # Bit of a hack
#                     if np.max(canvas) == 0:
#                         break
#                     gray_maze = canvas / (np.max(canvas) / scaling)
#                     gray_maze = np.clip(gray_maze, 0, 1) * 255
#                     images.append(gray_maze.astype(np.uint8))
#                 save_video("{}/visitations/Visits__Interval_{}__T_{}".format(LOGDIR, interval_size, T), images)

#                 # We want to show visualisations for the agent depending on which goals they've visited as well
#                 # Keep it seperate from the other one
#                 colour_images = []
#                 for i in range(0, T, interval // 10):
#                     canvas = np.zeros((env.env.maze.shape[0] * 3, env.env.maze.shape[1] * 3, 3))
#                     maze_size = env.env.maze.shape[0]
#                     for visit in Player_Positions_With_Goals[i: i + interval]:
#                         px = visit[0]
#                         py = visit[1]
#                         g1 = visit[2]
#                         g2 = visit[3]
#                         g3 = visit[4]
#                         if not g1 and not g2 and not g3:
#                             # No Goals visited
#                             canvas[px, py, :] += 1
#                         elif g1 and not g2 and not g3:
#                             # Only g1
#                             canvas[px, py + maze_size, 0] += 1
#                         elif not g1 and g2 and not g3:
#                             # Only g2
#                             canvas[px + maze_size, py + maze_size, 1] += 1
#                         elif not g1 and not g2 and g3:
#                             # Only g3
#                             canvas[px + 2 * maze_size, py + maze_size, 2] += 1
#                         elif g1 and g2 and not g3:
#                             # g1 and g2
#                             canvas[px, py + maze_size * 2, 0: 2] += 1
#                         elif g1 and not g2 and g3:
#                             # g1 and g3
#                             canvas[px + maze_size, py + maze_size * 2, 0: 3: 2] += 1
#                         elif not g1 and g2 and g3:
#                             # g2 and g3
#                             canvas[px + maze_size * 2, py + maze_size * 2, 1: 3] += 1
#                         else:
#                             # print("ERROR", g1, g2, g3)
#                             pass
#                     if np.max(canvas) == 0:
#                         break
#                     canvas = canvas / (np.max(canvas) / scaling)
#                     # Colour the unvisited goals
#                     # player_pos_with_goals = (player_pos[0], player_pos[1], env.maze[3, -1] != 2, env.maze[-1, -4] != 2, env.maze[-4, 0] != 2)
#                     # only g1
#                     canvas[maze_size - 1, maze_size + maze_size - 4, :] = 1
#                     canvas[maze_size - 4, maze_size, :] = 1
#                     # only g2
#                     canvas[maze_size + 3, maze_size + maze_size - 1, :] = 1
#                     canvas[maze_size + maze_size - 4, maze_size, :] = 1
#                     # only g3
#                     canvas[2 * maze_size + 3, maze_size + maze_size - 1, :] = 1
#                     canvas[2 * maze_size + maze_size - 1, maze_size + maze_size - 4, :] = 1
#                     # g1 and g2
#                     canvas[maze_size - 4, 2 * maze_size] = 1
#                     # g1 and g3
#                     canvas[maze_size + maze_size - 1, 2 * maze_size + maze_size - 4] = 1
#                     # g2 and g2
#                     canvas[2 * maze_size + 3, 2 * maze_size + maze_size - 1] = 1
#                     # Seperate the mazes
#                     canvas = np.insert(canvas, [maze_size, 2 * maze_size], 0.5, axis=0)
#                     canvas = np.insert(canvas, [maze_size, 2 * maze_size], 0.5, axis=1)
#                     colour_maze = canvas

#                     colour_maze = np.clip(colour_maze, 0, 1) * 255
#                     colour_images.append(colour_maze.astype(np.uint8))
#                 save_video("{}/visitations/Goal_Visits__Interval_{}__T_{}".format(LOGDIR, interval_size, T), colour_images)

#                 # We want to save the positions where the exploration bonus was high
#                 if not args.count:
#                     continue
#                 colour_images = []
#                 for i in range(0, T, interval // 10):
#                     canvas = np.zeros((env.env.maze.shape[0] * 3, env.env.maze.shape[1] * 3, 3))
#                     maze_size = env.env.maze.shape[0]
#                     for visit, bonus in zip(Player_Positions_With_Goals[i: i + interval], Exploration_Bonus[i: i + interval]):
#                         relative_bonus = bonus / max_exp_bonus
#                         px = visit[0]
#                         py = visit[1]
#                         g1 = visit[2]
#                         g2 = visit[3]
#                         g3 = visit[4]
#                         # Assign the maximum bonus in that interval to the image
#                         if not g1 and not g2 and not g3:
#                             # No Goals visited
#                             canvas[px, py, :] = max(relative_bonus, canvas[px, py, 0])
#                         elif g1 and not g2 and not g3:
#                             # Only g1
#                             canvas[px, py + maze_size, 0] = max(relative_bonus, canvas[px, py + maze_size, 0])
#                         elif not g1 and g2 and not g3:
#                             # Only g2
#                             canvas[px + maze_size, py + maze_size, 1] = max(relative_bonus, canvas[px + maze_size, py + maze_size, 1])
#                         elif not g1 and not g2 and g3:
#                             # Only g3
#                             canvas[px + 2 * maze_size, py + maze_size, 2] = max(relative_bonus, canvas[px + 2 * maze_size, py + maze_size, 2])
#                         elif g1 and g2 and not g3:
#                             # g1 and g2
#                             canvas[px, py + maze_size * 2, 0: 2] = max(relative_bonus, canvas[px, py + maze_size * 2, 0])
#                         elif g1 and not g2 and g3:
#                             # g1 and g3
#                             canvas[px + maze_size, py + maze_size * 2, 0: 3: 2] = max(relative_bonus, canvas[px + maze_size, py + maze_size * 2, 0])
#                         elif not g1 and g2 and g3:
#                             # g2 and g3
#                             canvas[px + maze_size * 2, py + maze_size * 2, 1: 3] = max(relative_bonus, canvas[px + maze_size * 2, py + maze_size * 2, 1])
#                         else:
#                             # print("ERROR", g1, g2, g3)
#                             pass
#                     canvas = np.clip(canvas, 0, 1)
#                     # Colour the unvisited goals
#                     # player_pos_with_goals = (player_pos[0], player_pos[1], env.maze[3, -1] != 2, env.maze[-1, -4] != 2, env.maze[-4, 0] != 2)
#                     # only g1
#                     canvas[maze_size - 1, maze_size + maze_size - 4, :] = 1
#                     canvas[maze_size - 4, maze_size, :] = 1
#                     # only g2
#                     canvas[maze_size + 3, maze_size + maze_size - 1, :] = 1
#                     canvas[maze_size + maze_size - 4, maze_size, :] = 1
#                     # only g3
#                     canvas[2 * maze_size + 3, maze_size + maze_size - 1, :] = 1
#                     canvas[2 * maze_size + maze_size - 1, maze_size + maze_size - 4, :] = 1
#                     # g1 and g2
#                     canvas[maze_size - 4, 2 * maze_size] = 1
#                     # g1 and g3
#                     canvas[maze_size + maze_size - 1, 2 * maze_size + maze_size - 4] = 1
#                     # g2 and g2
#                     canvas[2 * maze_size + 3, 2 * maze_size + maze_size - 1] = 1
#                     # Seperate the mazes
#                     canvas = np.insert(canvas, [maze_size, 2 * maze_size], 0.5, axis=0)
#                     canvas = np.insert(canvas, [maze_size, 2 * maze_size], 0.5, axis=1)
#                     colour_maze = canvas

#                     colour_maze = np.clip(colour_maze, 0, 1) * 255
#                     colour_images.append(colour_maze.astype(np.uint8))
#                 save_video("{}/exp_bonus/High_Exp_Bonus__Interval_{}__T_{}".format(LOGDIR, interval_size, T), colour_images)