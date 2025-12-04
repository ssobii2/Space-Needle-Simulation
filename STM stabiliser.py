"""
Needle Reaction Wheel Environment with Residual Reinforcement Learning
---------------------------------------------------------------------
Dependencies:
    pip install gymnasium mujoco stable-baselines3 numpy

This script:
  - Defines a MuJoCo XML model of a "needle" with 4 reaction wheels.
  - Implements a PD baseline controller (ClassicRateDamping).
  - Wraps it in a Gymnasium environment.
  - Supports training with PPO using residual RL.
  - Can visualize a learned policy interactively in the MuJoCo viewer.

"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData
from stable_baselines3 import PPO


# ---------- MuJoCo XML ----------
XML = """
<mujoco model="needle_rw">
  <option timestep="0.001" gravity="0 0 0" integrator="RK4"/>
  <default>
    <motor ctrllimited="true" ctrlrange="-0.05 0.05"/>
    <geom rgba="0.8 0.8 0.9 1" contype="0" conaffinity="0"/>
    <joint damping="1e-6"/>
  </default>

  <worldbody>
    <body name="base">
      <body name="gimbal_x">
        <joint name="jx" type="hinge" axis="1 0 0" damping="3e-1"/>
        <inertial pos="0 0 0" mass="5e-3" diaginertia="1e-4 1e-4 1.1e-4"/>
        <body name="gimbal_y">
          <joint name="jy" type="hinge" axis="0 1 0" damping="1e-1"/>
          <inertial pos="0 0 0" mass="5e-3" diaginertia="1e-4 1e-4 1.3e-4"/>
          <body name="needle">
            <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.006" mass="0.20"/>
            <geom name="pivot_ball" type="sphere" pos="0 0 0.00" size="0.025" density="0" rgba="0.7 0.7 0.8 1" contype="0" conaffinity="0"/>
            <site name="cg" pos="0 0 0" size="0.002"/>
            <body name="head" pos="0 0 0.08">
              <body name="rw1" pos="0 0.06 0">
                <joint name="j_rw1" type="hinge" axis="1 0 0" damping="8e-2" armature="2e-4" range="-1e6 1e6"/>
                <geom type="cylinder" fromto="-0.03 0 0 0.03 0 0" size="0.01" mass="0.03"/>
              </body>
              <body name="rw3" pos="0 -0.06 0">
                <joint name="j_rw3" type="hinge" axis="1 0 0" damping="8e-2" armature="2e-4" range="-1e6 1e6"/>
                <geom type="cylinder" fromto="-0.03 0 0 0.03 0 0" size="0.01" mass="0.03"/>
              </body>
              <body name="rw2" pos="0.06 0 0">
                <joint name="j_rw2" type="hinge" axis="0 1 0" damping="8e-2" armature="2e-4" range="-1e6 1e6"/>
                <geom type="cylinder" fromto="0 -0.03 0 0 0.03 0" size="0.01" mass="0.03"/>
              </body>
              <body name="rw4" pos="-0.06 0 0">
                <joint name="j_rw4" type="hinge" axis="0 1 0" damping="8e-2" armature="2e-4" range="-1e6 1e6"/>
                <geom type="cylinder" fromto="0 -0.03 0 0 0.03 0" size="0.01" mass="0.03"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="m_rw1" joint="j_rw1" gear="1" ctrlrange="-0.005 0.005"/> 
    <motor name="m_rw3" joint="j_rw3" gear="1" ctrlrange="-0.005 0.005"/> 
    <motor name="m_rw2" joint="j_rw2" gear="1" ctrlrange="-0.005 0.005"/> 
    <motor name="m_rw4" joint="j_rw4" gear="1" ctrlrange="-0.005 0.005"/> 
    <motor name="m_jx" joint="jx" gear="1" ctrlrange="-0.05 0.05"/>
    <motor name="m_jy" joint="jy" gear="1" ctrlrange="-0.05 0.05"/>


  </actuator>
</mujoco>
"""


# ---------- Utility ----------
def dof_for_joint(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"Joint '{name}' not found in model")
    return int(model.jnt_dofadr[jid])


def act_for(model, name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise KeyError(f"Actuator '{name}' not found in model")
    return int(aid)


# ---------- Baseline controller: PD + filtered D ----------
class ClassicRateDamping:
    def __init__(self, model):
        self.dt = model.opt.timestep
        # Gains
        self.Kp_x = 0.2
        self.Kp_y = 0.2
        self.Kd_x = 0.03
        self.Kd_y = 0.03
        self.Ki_x = 0.0
        self.Ki_y = 0.0
        self.Kbal_x = 5e-4
        self.Kbal_y = 5e-4
        self.rate_fc = 0.7
        self.alpha_rate = np.exp(-2 * np.pi * self.rate_fc * self.dt)
        self.state = {"int_x": 0.0, "int_y": 0.0, "ox_f": 0.0, "oy_f": 0.0}

    def compute(self, model, data, adr):
        jx = data.qpos[adr["jx"]]
        jy = data.qpos[adr["jy"]]
        ox = data.qvel[adr["jx"]]
        oy = data.qvel[adr["jy"]]
        s = self.state
        s["ox_f"] = self.alpha_rate * s["ox_f"] + (1 - self.alpha_rate) * ox
        s["oy_f"] = self.alpha_rate * s["oy_f"] + (1 - self.alpha_rate) * oy
        s["int_x"] += jx * self.dt
        s["int_y"] += jy * self.dt
        ix = np.clip(s["int_x"], -5.0, 5.0)
        iy = np.clip(s["int_y"], -5.0, 5.0)
        tau_x = -(self.Kp_x * jx + self.Kd_x * s["ox_f"] + self.Ki_x * ix)
        tau_y = -(self.Kp_y * jy + self.Kd_y * s["oy_f"] + self.Ki_y * iy)
        w1, w3 = data.qvel[adr["rw1"]], data.qvel[adr["rw3"]]
        w2, w4 = data.qvel[adr["rw2"]], data.qvel[adr["rw4"]]
        u1 = -0.5 * tau_x - self.Kbal_x * (w1 - w3)
        u3 = -0.5 * tau_x + self.Kbal_x * (w1 - w3)
        u2 = -0.5 * tau_y - self.Kbal_y * (w2 - w4)
        u4 = -0.5 * tau_y + self.Kbal_y * (w2 - w4)
        return np.array([u1, u3, u2, u4], dtype=np.float64)


# ---------- Gymnasium environment ----------
class NeedleRWEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, residual_rl=True, frame_skip=10, episode_len=5.0, seed=None, allow_user_injection=False, start_at_origin=False):
        super().__init__()
        self.tau_rw = 0.6  # seconds; increase to slow further
        self.u_phys = np.zeros(4, dtype=np.float64)  # filtered RW commands
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.frame_skip = frame_skip
        self.max_steps = int(episode_len / (self.dt * self.frame_skip))
        self.residual_rl = residual_rl
        self.allow_user_injection = allow_user_injection
        self.start_at_origin = start_at_origin

        # indices
        self.adr = {
            "jx": dof_for_joint(self.model, "jx"),
            "jy": dof_for_joint(self.model, "jy"),
            "rw1": dof_for_joint(self.model, "j_rw1"),
            "rw3": dof_for_joint(self.model, "j_rw3"),
            "rw2": dof_for_joint(self.model, "j_rw2"),
            "rw4": dof_for_joint(self.model, "j_rw4"),
        }
        self.act_id = {
            "rw1": act_for(self.model, "m_rw1"),
            "rw3": act_for(self.model, "m_rw3"),
            "rw2": act_for(self.model, "m_rw2"),
            "rw4": act_for(self.model, "m_rw4"),
            "jx": act_for(self.model, "m_jx"),
            "jy": act_for(self.model, "m_jy"),
        }

        # actuator slew limiter
        self.u_prev = np.zeros(self.model.nu, dtype=np.float64)
        self.du_max = 3e-4

        # controller
        self.classic = ClassicRateDamping(self.model)

        # disturbance
        self.ext = dict(amp_x=0.0, amp_y=0.0, fx=0.3, fy=0.3, phx=0.0, phy=0.0)
        self.t = 0.0
        self.step_count = 0

        # observation space
        high = np.array([np.pi, np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # action space
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        self.ctrl_lo = float(lo[self.act_id["rw1"]])
        self.ctrl_hi = float(hi[self.act_id["rw1"]])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # reward weights
        self.w_ang = 4.0
        self.w_rate = 0.5
        self.w_wheel = 1e-3
        self.w_ctrl = 5e-3
        self.w_residual = 1e-2

        self.np_random, _ = gym.utils.seeding.np_random(seed)

    # ---------- helpers ----------
    def _slew(self, idx, target):
        lo = self.u_prev[idx] - self.du_max
        hi = self.u_prev[idx] + self.du_max
        val = float(np.clip(target, lo, hi))
        self.u_prev[idx] = val
        return val

    def _obs(self):
        jx = self.data.qpos[self.adr["jx"]]
        jy = self.data.qpos[self.adr["jy"]]
        ox = self.data.qvel[self.adr["jx"]]
        oy = self.data.qvel[self.adr["jy"]]
        w1 = self.data.qvel[self.adr["rw1"]]
        w3 = self.data.qvel[self.adr["rw3"]]
        w2 = self.data.qvel[self.adr["rw2"]]
        w4 = self.data.qvel[self.adr["rw4"]]
        return np.array([jx, jy, ox, oy, w1, w3, w2, w4], dtype=np.float32)

    def _scale_action(self, a):
        return self.ctrl_lo + 0.5 * (a + 1.0) * (self.ctrl_hi - self.ctrl_lo)

    def _apply_disturbance(self, t):
        self.data.qfrc_applied[:] = 0.0
        ax = self.ext["amp_x"] * np.sin(2 * np.pi * self.ext["fx"] * t + self.ext["phx"])
        ay = self.ext["amp_y"] * np.cos(2 * np.pi * self.ext["fy"] * t + self.ext["phy"])
        self.data.qfrc_applied[self.adr["jx"]] = ax
        self.data.qfrc_applied[self.adr["jy"]] = ay

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.u_prev[:] = 0.0
        self.classic.state.update({"int_x": 0.0, "int_y": 0.0, "ox_f": 0.0, "oy_f": 0.0})
        self.t = 0.0
        self.step_count = 0
        self.data.ctrl[:] = 0.0
        # random initial tilt/rate
        start_origin = self.start_at_origin
        if options is not None and "start_at_origin" in options:
            start_origin = bool(options["start_at_origin"])

        if start_origin:
            # exact origin
            self.data.qpos[self.adr["jx"]] = 0.0
            self.data.qpos[self.adr["jy"]] = 0.0
            self.data.qvel[self.adr["jx"]] = 0.0
            self.data.qvel[self.adr["jy"]] = 0.0
        else:
            # randomized start
            self.data.qpos[self.adr["jx"]] = self.np_random.uniform(-0.1, 0.1)
            self.data.qpos[self.adr["jy"]] = self.np_random.uniform(-0.1, 0.1)
            self.data.qvel[self.adr["jx"]] = self.np_random.uniform(-0.1, 0.1)
            self.data.qvel[self.adr["jy"]] = self.np_random.uniform(-0.1, 0.1)

        # zero wheel speeds
        for k in ["rw1", "rw3", "rw2", "rw4"]:
            self.data.qvel[self.adr[k]] = 0.0

        # disturbances: random unless you disable later for visualize
        if self.allow_user_injection:
            # keep the demo clean of background disturbances
            self.ext.update(amp_x=0.0, amp_y=0.0)
        else:
            # randomize during training
            self.ext["amp_x"] = self.np_random.uniform(0.0, 0.004)
            self.ext["amp_y"] = self.np_random.uniform(0.0, 0.004)
            self.ext["fx"] = self.np_random.uniform(0.2, 0.6)
            self.ext["fy"] = self.np_random.uniform(0.2, 0.6)
            self.ext["phx"] = self.np_random.uniform(0, 2 * np.pi)
            self.ext["phy"] = self.np_random.uniform(0, 2 * np.pi)
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()
        action = np.clip(action, -1.0, 1.0)
        u_rl = self._scale_action(action)

        u_classic = self.classic.compute(self.model, self.data, self.adr) if self.residual_rl else np.zeros(4,
                                                                                                            dtype=np.float64)
        u_cmd = u_classic + u_rl  # final RW commands
        alpha = min(self.dt / max(self.tau_rw, 1e-6), 1.0)  # actuator filter gain

        # Internal flag: allow manual torque injection via viewer sliders on jx/jy
        allow_user = getattr(self, "allow_user_injection", False)

        for _ in range(self.frame_skip):
            t = self.t
            self._apply_disturbance(t)

            if not allow_user:
                self.data.ctrl[self.act_id["jx"]] = 0.0
                self.data.ctrl[self.act_id["jy"]] = 0.0

            # Filtered RW commands
            for i, key in enumerate(["rw1", "rw3", "rw2", "rw4"]):
                self.u_phys[i] += alpha * (u_cmd[i] - self.u_phys[i])
                self.data.ctrl[self.act_id[key]] = self._slew(self.act_id[key], self.u_phys[i])

            mujoco.mj_step(self.model, self.data)
            self.t += self.dt

        jx = self.data.qpos[self.adr["jx"]]
        jy = self.data.qpos[self.adr["jy"]]
        ox = self.data.qvel[self.adr["jx"]]
        oy = self.data.qvel[self.adr["jy"]]
        w = [self.data.qvel[self.adr[k]] for k in ["rw1", "rw2", "rw3", "rw4"]]

        ang_cost = self.w_ang * (jx ** 2 + jy ** 2)
        rate_cost = self.w_rate * (ox ** 2 + oy ** 2)
        wheel_cost = self.w_wheel * sum(np.square(w))
        ctrl_cost = self.w_ctrl * np.sum(u_cmd ** 2)
        residual_cost = self.w_residual * np.sum(u_rl ** 2) if self.residual_rl else 0.0
        reward = -(ang_cost + rate_cost + wheel_cost + ctrl_cost + residual_cost)

        terminated = bool(abs(jx) > 0.7 or abs(jy) > 0.7 or any(abs(x) > 400 for x in w))
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        return self._obs(), reward, terminated, truncated, {}

# ---------- Train ----------
def train(residual_rl=True, total_timesteps=300_000):
    env = NeedleRWEnv(residual_rl=residual_rl, frame_skip=10, episode_len=5.0, seed=42)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_needle_rw_residual" if residual_rl else "ppo_needle_rw_end2end")
    env.close()

# ---------- Visualize ----------
def visualize(model_path="ppo_needle_rw_residual.zip", residual_rl=True):
    """Local visualization - note: viewer API may vary by MuJoCo version"""
    try:
        from mujoco import viewer as mj_viewer
        env = NeedleRWEnv(residual_rl=residual_rl, frame_skip=5, allow_user_injection=True, start_at_origin=True)
        model = PPO.load(model_path)
        
        # Try the newer viewer API
        with mj_viewer.launch_passive(env.model, env.data) as viewer:
            obs, _ = env.reset(options={"start_at_origin": True})
            env.ext.update(amp_x=0.0, amp_y=0.0)
            
            while viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = env.step(action)
                
                if term or trunc:
                    obs, _ = env.reset(options={"start_at_origin": True})
                
                viewer.sync()
        
        env.close()
    except (AttributeError, ImportError):
        print("Note: MuJoCo viewer not available. Use the web interface instead.")
        env.close()


if __name__ == "__main__":
    import os
    model_path = "ppo_needle_rw_residual.zip"
    if not os.path.exists(model_path):
        print("Model not found. Training new model...")
        train(residual_rl=True, total_timesteps=200_000)
    else:
        print(f"Using existing model: {model_path}")
    visualize(model_path, residual_rl=True)
