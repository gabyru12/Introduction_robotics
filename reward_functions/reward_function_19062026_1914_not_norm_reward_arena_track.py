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
    touch = float(max([v_front, v_left, v_right]))

    if "arena" in self.world_name:
        if touch > 0:
            if self.resetting:
                self.resetting = False
            else:
                reward -= 100
                terminated = True
        elif dist2centerline_point >= 5:
            reward -= 7.5
            print("Penalty")
        else:
            reward += 0.2
            current_sector = self._get_arena_sector(x, y)
            if current_sector != self.prev_sector:
                prev_expected = (self.next_checkpoint - 1) % self._n_checkpoints
                if (current_sector == self.next_checkpoint and
                        self.prev_sector == prev_expected):
                    reward += 30
                    self.next_checkpoint = (self.next_checkpoint + 1) % self._n_checkpoints
                    angle_deg = round(math.degrees(math.atan2(y - self._arena_cy, x - self._arena_cx)) % 360)
                    print(
                        f"[checkpoint {current_sector}/{self._n_checkpoints}] next={self.next_checkpoint} angle={angle_deg}°")
                self.prev_sector = current_sector

            if cruising_speed > 25:
                if 30 < heading_error_deg < 60:
                    print("Drift")
                    reward += 50
                elif heading_error_deg < 30 and heading_error_deg > -90 and steering < 0:
                    print("Steering left to drift")
                    reward += 10
                elif heading_error_deg > 60 and heading_error_deg < 90 and steering > 0:
                    print("Steering right to maintain drift")
                    reward += 10
    elif "track" in self.world_name:
        car_pos = self.driver.getFromDef("VEHICLE").getField("translation").getSFVec3f()
        idx, _, _ = closest_centerline_point(car_pos[:2], self.centerline)
        progress = self.comulative_progress[idx]

        if touch > 0:
            if self.resetting:
                self.resetting = False
            else:
                reward -= 100
                terminated = True
        elif dist2centerline_point >= 5:
            reward += -2
        else:
            if x < 130 or x > 170:
                if slip_angle_deg > 30 and slip_angle_deg < 60:
                    reward += 10
                if heading_error_deg > 30 and heading_error_deg < 60:
                    reward += 5
            if x > 130 and x < 170:
                if abs(slip_angle_deg) < 10 and abs(heading_error_deg) < 10 and progress > self.episode_progress:
                    reward += 2

        if progress > self.episode_progress:
            self.episode_progress = progress

    return reward, terminated, truncated