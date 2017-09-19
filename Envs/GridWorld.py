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

        # Counting stuff
        self.counts = np.empty_like(self.grid)

    def _step(self, a):
        info_dict = {}

        # Update counts
        self.counts[self.player_pos] += 1
        current_count = self.counts[self.player_pos]
        action_counts = []
        for aa in self.actions:
            new_player_pos = (self.player_pos[0] + aa[0], self.player_pos[1] + aa[1])
            # Clip
            if new_player_pos[0] < 0 or new_player_pos[0] >= self.grid.shape[0]\
            or new_player_pos[1] < 0 or new_player_pos[1] >= self.grid.shape[1]:
                new_player_pos = self.player_pos

            # Into a wall
            if self.grid[new_player_pos] == 1:
                new_player_pos = self.player_pos

            action_counts.append(self.counts[new_player_pos])

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

        # Fill in info dict with the action selection statistics
        new_state_count = self.counts[new_player_pos]
        count_list = [current_count] + action_counts + [new_state_count]
        info_dict["Action_Counts"] = np.array(count_list)

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

    def state_to_image(self, state):
        grid = state
        image = np.zeros(shape=(grid.shape[0], grid.shape[1], 3))
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] != 0:
                    image[x, y] = (255 * grid[x, y] / 3, 255 * grid[x, y] / 3, 255 * grid[x, y] / 3)
        return image

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

    def state_to_player_pos(self, state):
        internal_state = state[:, :, 0]
        goals_list = [internal_state[g[0], g[1]] > 0.6 and internal_state[g[0], g[1]] < 0.7 for g in self.goals_order]
        # print(goals_list)
        player_pos = list(np.argwhere(internal_state > 0.9)[0])
        joint = player_pos + goals_list
        # print(tuple(joint))
        return tuple(joint)

    def trained_on_states(self, player_visits, args):

        # interval = args.exp_replay_size

        if self.num_goals > 3:
            raise Exception("Cant do trained on states for >3 goals atm")

        # We want to show visualisations for the agent depending on which goals they've visited as well
        # Keep it seperate from the other one
        colour_images = []
        # Works for num_goals <= 3
        # print(self.grid.shape)
        canvas = np.zeros((self.grid.shape[0] * self.num_goals, self.grid.shape[1] * self.num_goals, 3))
        grid_x = self.grid.shape[0]
        grid_y = self.grid.shape[1]

        # end_t = int(args.t_max * i / 100) * args.batch_size
        # start_t = int(args.t_max * (i - 1) / 100) * args.batch_size
        # print("\n\n\n\n",start_t, end_t)

        for visit in player_visits:
            # print(visit)
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
            return
        canvas = canvas / np.max(canvas)

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
        return colour_images[0]

    def xp_replay_states(self, player_visits, args):

        # interval = args.exp_replay_size

        if self.num_goals > 3:
            raise Exception("Cant do xp replay states for >3 goals atm")

        # We want to show visualisations for the agent depending on which goals they've visited as well
        # Keep it seperate from the other one
        colour_images = []
        # Works for num_goals <= 3
        # print(self.grid.shape)
        canvas = np.zeros((self.grid.shape[0] * self.num_goals, self.grid.shape[1] * self.num_goals, 3))
        grid_x = self.grid.shape[0]
        grid_y = self.grid.shape[1]

        # end_t = int(args.t_max * i / 100)
        # start_t = max(0, end_t - args.exp_replay_size)

        for visit in player_visits:
            # print(visit)
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
            canvas[x_place, y_place, goal_colours] = 1

        if np.max(canvas) == 0:
            return
        # canvas = canvas / (np.max(canvas) / scaling)

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
        return colour_images[0]
        # save_video("{}/visitations/Goal_Visits__Interval_{}__T_{}".format(LOGDIR, interval_size, T), colour_images)

    def player_visits(self, player_visits, args):
        # Log the visitations
        with open("{}/logs/Player_Positions.txt".format(args.log_path), "a") as file:
            file.write('\n'.join(" ".join(str(x) for x in t) for t in player_visits))

        scaling = 2

        if self.num_goals > 3:
            raise Exception("Cant do state visitations for >3 goals atm")

        # We want to show visualisations for the agent depending on which goals they've visited as well
        # Keep it seperate from the other one
        colour_images = []
        # Works for num_goals <= 3
        # print(self.grid.shape)
        canvas = np.zeros((self.grid.shape[0] * self.num_goals, self.grid.shape[1] * self.num_goals, 3))
        grid_x = self.grid.shape[0]
        grid_y = self.grid.shape[1]

        for visit in player_visits:
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
            return
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
        return colour_images[0]
        # save_video("{}/visitations/Goal_Visits__Interval_{}__T_{}".format(LOGDIR, interval_size, T), colour_images)

    def bonus_landscape(self, player_visits, exploration_bonuses, max_bonus, args):
        # interval = int(args.t_max / args.interval_size)
        # scaling = 2

        if self.num_goals > 3:
            raise Exception("Cant do bonus landscape for >3 goals atm")

        # We want to show visualisations for the agent depending on which goals they've visited as well
        # Keep it seperate from the other one
        colour_images = []
        # for i in range(0, args.t_max, interval // 10):
        # Works for num_goals <= 3
        # print(self.grid.shape)
        canvas = np.zeros((self.grid.shape[0] * self.num_goals, self.grid.shape[1] * self.num_goals, 3))
        grid_x = self.grid.shape[0]
        grid_y = self.grid.shape[1]

        for visit, bonus in zip(player_visits, exploration_bonuses):
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
            return
        # canvas = canvas / np.max(canvas)

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
        return colour_images[0]

    def frontier(self, exp_model, args, max_bonus=None):
        if max_bonus is None:
            max_bonus = args.beta
        if self.num_goals > 1:
            raise Exception("Cannot do frontier for >1 goals atm")

        actions = 1
        if args.count_state_action:
            actions = args.actions
        canvas = np.zeros((self.grid.shape[0] * self.num_goals, self.grid.shape[1] * self.num_goals * actions, 3))
        grid_x = self.grid.shape[0]
        grid_y = self.grid.shape[1]
        grid = self.grid

        for a in range(actions):
            for x in range(grid_x):
                for y in range(grid_y):

                    if grid[x, y] == 1 or grid[x, y] == 2:
                        # If the position is a wall the player cannot ever be there
                        continue

                    state_copy = np.copy(self.grid)
                    state_copy[self.player_pos] = 0
                    state_copy[x, y] = 3
                    # print(state_copy)
                    state_copy = state_copy[:, :, np.newaxis] / 3

                    bonus, _ = exp_model.bonus(state_copy, action=a, dont_remember=True)
                    # print(x,y,bonus)
                    canvas[x, y + (grid_y * a), 0] = bonus

        # canvas /= np.max(canvas)
        canvas /= max_bonus

        # Walls
        for a in range(actions):
            for x in range(grid_x):
                for y in range(grid_y):
                    if grid[x, y] == 1 or grid[x, y] == 2:
                        canvas[x, y + (grid_y * a), :] = grid[x, y] / 3

        canvas = np.clip(canvas, 0, 1) * 255

        return canvas
