import gym
from gym import spaces
import pygame
import numpy as np

"""
0: Eempty Square
1: Wall
2: Up Laser Generator
3: Down Laser Generator
4: Right Laser Generator
5: Left Laser Generator
3 (temp): Laser Beam
6: Target
7: Bomb
8: Freeze Plate
9: Right Mirror
10: Left Mirror
"""
MAP = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 8, 8],
    [0, 0, 0, 0, 9, 10, 0, 0, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 5, 1, 3, 0, 1, 0, 0, 0],
    [4, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 6, 0, 0],
])


class DweepEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 520  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        # We have 4 more actions corresponding to diagonal movement (8 total)
        self.action_space = spaces.Discrete(8)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]), # right
            1: np.array([0, -1]), # up
            2: np.array([-1, 0]), # left
            3: np.array([0, 1]), # down
            4: np.array([1, -1]), # up and right
            5: np.array([-1, -1]), # up and left
            6: np.array([1, 1]), # down and right
            7: np.array([-1, 1]), # up and left
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.game_map = MAP.copy() # This map can change
        self.laser_map = np.full((MAP.shape[0], MAP.shape[1]), False) # Keeps track of laser paths
        self.update_lasers()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = np.array([0, 0]) # Top-left
        self._target_location = np.array([9, 7]) # In bottom-right corner

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3,4,5,6,7}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        new_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        x, y = new_location

        # Check for laser beam collision


        if self.game_map[x, y] in [1, 2, 3, 4, 5, 9, 10]:
            # This is a wall or laser generator or mirror, so we stay at old location
            reward = 0
            terminated = False
        elif self.laser_map[x, y]:
            # This is a laser, so we die
            self._agent_location = new_location
            reward = -1
            terminated = True
        elif self.game_map[x, y] in [0, 8]:
            # Dweep moves to a new square
            self._agent_location = new_location
            reward = 0
            terminated = False
        elif self.game_map[x, y] == 6:
            self._agent_location = new_location
            reward = 1 # Binary sparse rewards
            terminated = True
        else:
            raise ValueError("Invalid entry in map")

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {}
        # return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (50, 255, 255),
            pygame.Rect(
                pix_square_size * np.flip(self._target_location),
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent Dweep
        pygame.draw.circle(
            canvas,
            (255, 0, 255),
            (np.flip(self._agent_location) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw the walls, laser, laser generator, and freeze plates
        # TODO: Remove this hardcoded size
        for i in range(10):
            for j in range(10):
                loc = np.array([i, j])
                if self.game_map[i, j] == 1: # Draw wall
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),
                        pygame.Rect(
                            pix_square_size * np.flip(loc),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.game_map[i, j] == 2: # Draw laser generator
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 255),
                        pygame.Rect(
                            pix_square_size * np.flip(loc),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.game_map[i, j] == 8: # Draw freeze plate
                    pygame.draw.rect(
                        canvas,
                        (0, 100, 100),
                        pygame.Rect(
                            pix_square_size * np.flip(loc),
                            (pix_square_size, pix_square_size),
                        )
                    )
                if self.laser_map[i, j]:
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            pix_square_size * np.flip(loc + 0.25),
                            (pix_square_size / 2, pix_square_size),
                        )
                    )
                # Commenting out the old code for drawing lasers
                """
                elif self.game_map[i, j] == 3: # Draw laser
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            pix_square_size * np.flip(loc + 0.25),
                            (pix_square_size / 2, pix_square_size),
                        ),
                    )
                """
        
        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # Helper method to update laser paths
    def update_lasers(self):
        direction = np.array([0, 0])
        for row in range(self.game_map.shape[0]):
            for col in range(self.game_map.shape[1]):
                if self.game_map[row][col] == 2:
                    direction = np.array([-1, 0])
                elif self.game_map[row][col] == 3:
                    direction = np.array([1, 0])
                elif self.game_map[row][col] == 4:
                    direction = np.array([0, 1])
                elif self.game_map[row][col] == 5:
                    direction = np.array([0, -1])
                else:
                    continue
                # Following code only executes if the object is a laser generator
                # Trace the laser's path until it hits a wall, changing direction at mirrors
                loc = np.array([row, col])
                loc += direction
                while (loc[0] < self.laser_map.shape[0] and loc[0] >= 0 and loc[1] < self.laser_map.shape[1] and loc[1] >= 0):
                    if self.game_map[loc[0]][loc[1]] == 1:
                        break
                    elif self.game_map[loc[0]][loc[1]] in [2, 3, 4, 5]:
                        self.laser_map[loc[0]][loc[1]] = True
                        break
                    elif self.game_map[loc[0]][loc[1]] == 9:
                        self.laser_map[loc[0]][loc[1]] = True
                        direction = np.array([direction[1] * -1, direction[0] * -1])
                    elif self.game_map[loc[0]][loc[1]] == 10:
                        self.laser_map[loc[0]][loc[1]] = True
                        direction = np.array([direction[1], direction[0]])
                    else:
                        self.laser_map[loc[0]][loc[1]] = True
                    loc += direction
                
if __name__ == '__main__':
    env = DweepEnv(render_mode="human", size=10)
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # take a random action
        env.step(action)
    env.close()