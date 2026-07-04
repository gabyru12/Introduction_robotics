def compute_reward(self, obs=None, steering=0.0, throttle=0.0):
    terminated = False
    truncated = False
    reward = 0
    cruising_speed = obs[-8]
    slip_angle_deg = obs[-4]
    heading_error_deg = obs[-3]
    dist2centerline_point = obs[-2]

    x, y = self.vehicle_node.getField("translation").getSFVec3f()[:2]

    v_front = self.touch_front.getValue()
    v_left = self.touch_left.getValue()
    v_right = self.touch_right.getValue()
    v_rear = self.touch_rear.getValue()
    touch = float(max([v_front, v_left, v_right, v_rear]))

    if "arena" in self.world_name:
        car_pos = self.driver.getFromDef("VEHICLE").getField("translation").getSFVec3f()
        idx, _, _ = closest_centerline_point(car_pos[:2], self.centerline)
        progress = self.comulative_progress[idx]

        current_sector = self._get_arena_sector(x, y)
        if current_sector != self.prev_sector:
            prev_expected = (self.next_checkpoint - 1) % self._n_checkpoints
            if (current_sector == self.next_checkpoint and
                    self.prev_sector == prev_expected):
                reward += 5
                self.next_checkpoint = (self.next_checkpoint + 1) % self._n_checkpoints
                angle_deg = round(math.degrees(math.atan2(y - self._arena_cy, x - self._arena_cx)) % 360)
                print(
                    f"[checkpoint {current_sector}/{self._n_checkpoints}] next={self.next_checkpoint} angle={angle_deg}°")
            self.prev_sector = current_sector

        if touch > 0:
            if self.resetting:
                self.resetting = False
            else:
                reward += -10
                terminated = True
        elif dist2centerline_point >= 5:
            reward += -0.2
        elif progress > self.episode_progress:
            reward += 0.2
            if cruising_speed > 25:
                if 30 < heading_error_deg < 60:
                    reward += 1
                elif heading_error_deg < 30 and heading_error_deg > -90 and steering < 0:
                    reward += 0.2
                elif heading_error_deg > 60 and heading_error_deg < 90 and steering > 0:
                    reward += 0.2

        if progress > self.episode_progress or (self.episode_progress / self.comulative_progress[-1] > 0.9 and
                                                progress / self.comulative_progress[-1] < 0.1):
            self.episode_progress = progress

    elif "track" in self.world_name:
        car_pos = self.driver.getFromDef("VEHICLE").getField("translation").getSFVec3f()
        idx, _, _ = closest_centerline_point(car_pos[:2], self.centerline)
        progress = self.comulative_progress[idx]

        reward -= 0.05 * (dist2centerline_point ** 2)

        if touch > 0:
            if self.resetting:
                self.resetting = False
            else:
                reward += -10
                terminated = True
        elif progress >= self.episode_progress:
            if x < 130 or x > 170:
                heading_error = abs(heading_error_deg - 45)
                slip_error = abs(slip_angle_deg - 45)
                if heading_error < 45:
                    reward += math.exp(-(heading_error ** 2) / (2 * 10.0 ** 2))
                    reward += math.exp(-(slip_error ** 2) / (2 * 10.0 ** 2))
                else:
                    reward -= 1

        if progress > self.episode_progress or (self.episode_progress / self.comulative_progress[-1] > 0.9 and
                                                progress / self.comulative_progress[-1] < 0.1):
            self.episode_progress = progress

    return reward, terminated, truncated