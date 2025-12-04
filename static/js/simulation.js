/**
 * Needle Reaction Wheel Environment - JavaScript Port
 * Ported from STM stabiliser.py to run in browser with MuJoCo.js
 */

class ClassicRateDamping {
    constructor(model) {
        this.dt = model.opt.timestep;
        // Gains (matching Python version)
        this.Kp_x = 0.2;
        this.Kp_y = 0.2;
        this.Kd_x = 0.03;
        this.Kd_y = 0.03;
        this.Ki_x = 0.0;
        this.Ki_y = 0.0;
        this.Kbal_x = 5e-4;
        this.Kbal_y = 5e-4;
        this.rate_fc = 0.7;
        this.alpha_rate = Math.exp(-2 * Math.PI * this.rate_fc * this.dt);
        this.state = {
            int_x: 0.0,
            int_y: 0.0,
            ox_f: 0.0,
            oy_f: 0.0
        };
    }

    compute(model, data, adr) {
        const qpos = data.qpos;
        const qvel = data.qvel;
        const jx = qpos[adr.jx];
        const jy = qpos[adr.jy];
        const ox = qvel[adr.jx];
        const oy = qvel[adr.jy];
        const s = this.state;
        
        // Filtered rates
        s.ox_f = this.alpha_rate * s.ox_f + (1 - this.alpha_rate) * ox;
        s.oy_f = this.alpha_rate * s.oy_f + (1 - this.alpha_rate) * oy;
        
        // Integrals
        s.int_x += jx * this.dt;
        s.int_y += jy * this.dt;
        const ix = Math.max(-5.0, Math.min(5.0, s.int_x));
        const iy = Math.max(-5.0, Math.min(5.0, s.int_y));
        
        // Torques
        const tau_x = -(this.Kp_x * jx + this.Kd_x * s.ox_f + this.Ki_x * ix);
        const tau_y = -(this.Kp_y * jy + this.Kd_y * s.oy_f + this.Ki_y * iy);
        
        // Reaction wheel velocities
        const w1 = qvel[adr.rw1];
        const w3 = qvel[adr.rw3];
        const w2 = qvel[adr.rw2];
        const w4 = qvel[adr.rw4];
        
        // Reaction wheel commands
        const u1 = -0.5 * tau_x - this.Kbal_x * (w1 - w3);
        const u3 = -0.5 * tau_x + this.Kbal_x * (w1 - w3);
        const u2 = -0.5 * tau_y - this.Kbal_y * (w2 - w4);
        const u4 = -0.5 * tau_y + this.Kbal_y * (w2 - w4);
        
        return new Float64Array([u1, u3, u2, u4]);
    }
}

class NeedleRWEnv {
    constructor(model, data, mujoco, residualRL = true, frameSkip = 5, episodeLen = 5.0, allowUserInjection = true) {
        this.model = model;
        this.data = data;
        this.mujoco = mujoco;
        this.residualRL = residualRL;
        this.frameSkip = frameSkip;
        this.allowUserInjection = allowUserInjection;
        this.dt = model.opt.timestep;
        
        // Episode management - match Python exactly
        this.episodeLen = episodeLen;
        this.maxSteps = Math.floor(episodeLen / (this.dt * this.frameSkip));
        
        // Actuator parameters
        this.tau_rw = 0.6; // seconds
        this.u_phys = new Float64Array(4);
        this.u_prev = new Float64Array(model.nu);
        this.du_max = 3e-4;
        
        // Controller
        this.classic = new ClassicRateDamping(model);
        
        // Disturbance parameters
        this.ext = {
            amp_x: 0.0,
            amp_y: 0.0,
            fx: 0.3,
            fy: 0.3,
            phx: 0.0,
            phy: 0.0
        };
        this.t = 0.0;
        this.stepCount = 0;
        
        // Reward weights - match Python exactly
        this.w_ang = 4.0;
        this.w_rate = 0.5;
        this.w_wheel = 1e-3;
        this.w_ctrl = 5e-3;
        this.w_residual = 1e-2;
        
        // Get actuator and joint indices using mujoco_wasm_contrib API
        this.actId = this._getActuatorIndices();
        this.adr = this._getJointIndices();
        
        // Action scaling - get from model actuator_ctrlrange
        // Based on XML: reaction wheels have ctrlrange="-0.005 0.005"
        this.ctrlLo = -0.005;
        this.ctrlHi = 0.005;
        // Try to get actual range from model if available
        try {
            const rw1Idx = this.actId.rw1;
            // Access actuator control range - format may vary by API
            if (model.actuator_ctrlrange) {
                // If it's a flat array: [lo1, hi1, lo2, hi2, ...]
                if (Array.isArray(model.actuator_ctrlrange) && model.actuator_ctrlrange.length > rw1Idx * 2) {
                    this.ctrlLo = model.actuator_ctrlrange[rw1Idx * 2];
                    this.ctrlHi = model.actuator_ctrlrange[rw1Idx * 2 + 1];
                }
            }
        } catch (e) {
            console.warn('Could not get actuator range, using defaults:', e);
        }
    }
    
    _getActuatorIndices() {
        // Based on XML order in STM stabiliser.py:
        // m_rw1, m_rw3, m_rw2, m_rw4, m_jx, m_jy
        // We'll use hardcoded indices since we know the exact structure
        return {
            rw1: 0,  // m_rw1
            rw3: 1,  // m_rw3
            rw2: 2,  // m_rw2
            rw4: 3,  // m_rw4
            jx: 4,   // m_jx
            jy: 5    // m_jy
        };
    }
    
    _getJointIndices() {
        // Based on XML order in STM stabiliser.py:
        // jx, jy, j_rw1, j_rw3, j_rw2, j_rw4
        // These are DOF addresses (indices into qpos/qvel arrays)
        return {
            jx: 0,   // jx
            jy: 1,   // jy
            rw1: 2,  // j_rw1
            rw3: 3,  // j_rw3
            rw2: 4,  // j_rw2
            rw4: 5   // j_rw4
        };
    }
    
    _slew(idx, target) {
        const lo = this.u_prev[idx] - this.du_max;
        const hi = this.u_prev[idx] + this.du_max;
        const val = Math.max(lo, Math.min(hi, target));
        this.u_prev[idx] = val;
        return val;
    }
    
    _obs() {
        // Ensure forward kinematics are up to date before reading qpos/qvel
        // This ensures we're reading the current state, not stale values
        this.mujoco.mj_forward(this.model, this.data);
        
        // Access data buffers (typed arrays) using mujoco_wasm_contrib API
        // Match Python exactly: [jx, jy, ox, oy, w1, w3, w2, w4]
        const qpos = this.data.qpos;
        const qvel = this.data.qvel;
        return new Float32Array([
            qpos[this.adr.jx],   // jx
            qpos[this.adr.jy],   // jy
            qvel[this.adr.jx],   // ox (angular velocity of jx)
            qvel[this.adr.jy],   // oy (angular velocity of jy)
            qvel[this.adr.rw1],  // w1
            qvel[this.adr.rw3],  // w3
            qvel[this.adr.rw2],  // w2
            qvel[this.adr.rw4]   // w4
        ]);
    }
    
    _scaleAction(a) {
        // Scale action from [-1, 1] to [ctrl_lo, ctrl_hi]
        return this.ctrlLo + 0.5 * (a + 1.0) * (this.ctrlHi - this.ctrlLo);
    }
    
    _applyDisturbance(t) {
        // Clear applied forces
        const qfrc_applied = this.data.qfrc_applied;
        for (let i = 0; i < qfrc_applied.length; i++) {
            qfrc_applied[i] = 0.0;
        }
        
        // Apply sinusoidal disturbances
        const ax = this.ext.amp_x * Math.sin(2 * Math.PI * this.ext.fx * t + this.ext.phx);
        const ay = this.ext.amp_y * Math.cos(2 * Math.PI * this.ext.fy * t + this.ext.phy);
        qfrc_applied[this.adr.jx] = ax;
        qfrc_applied[this.adr.jy] = ay;
    }
    
    reset(startAtOrigin = true) {
        // Reset simulation using mujoco_wasm_contrib API
        this.mujoco.mj_resetData(this.model, this.data);
        
        // Reset internal state
        this.u_prev.fill(0.0);
        this.u_phys.fill(0.0);
        this.classic.state = {
            int_x: 0.0,
            int_y: 0.0,
            ox_f: 0.0,
            oy_f: 0.0
        };
        this.t = 0.0;
        this.stepCount = 0;
        
        // Clear all controls
        const ctrl = this.data.ctrl;
        for (let i = 0; i < ctrl.length; i++) {
            ctrl[i] = 0.0;
        }
        
        // Set initial positions
        const qpos = this.data.qpos;
        const qvel = this.data.qvel;
        if (startAtOrigin) {
            qpos[this.adr.jx] = 0.0;
            qpos[this.adr.jy] = 0.0;
            qvel[this.adr.jx] = 0.0;
            qvel[this.adr.jy] = 0.0;
        }
        
        // Zero wheel speeds
        qvel[this.adr.rw1] = 0.0;
        qvel[this.adr.rw3] = 0.0;
        qvel[this.adr.rw2] = 0.0;
        qvel[this.adr.rw4] = 0.0;
        
        // Disable disturbances for manual control
        if (this.allowUserInjection) {
            this.ext.amp_x = 0.0;
            this.ext.amp_y = 0.0;
        }
        
        // Ensure forward kinematics are updated after reset
        // This ensures qpos/qvel are correctly reflected in the observation
        this.mujoco.mj_forward(this.model, this.data);
        
        // Return Gymnasium-style tuple: (obs, info)
        const obs = this._obs();
        return { obs, info: {} };
    }
    
    step(action) {
        // Clip and scale action - match Python exactly: flatten and clip to [-1, 1]
        // Python: action = np.asarray(action, dtype=np.float64).flatten()
        // Python: action = np.clip(action, -1.0, 1.0)
        let actionArray;
        if (Array.isArray(action)) {
            actionArray = action.flat(); // Flatten nested arrays
        } else if (action instanceof Float32Array || action instanceof Float64Array || action instanceof Array) {
            actionArray = Array.from(action);
        } else {
            actionArray = [action];
        }
        // Clip to [-1, 1] range
        const clippedAction = actionArray.map(a => Math.max(-1.0, Math.min(1.0, Number(a))));
        const uRL = new Float64Array(clippedAction.map(a => this._scaleAction(a)));
        
        // Compute classic controller output
        const uClassic = this.residualRL 
            ? this.classic.compute(this.model, this.data, this.adr)
            : new Float64Array(4);
        
        // Final reaction wheel commands
        const uCmd = new Float64Array(4);
        for (let i = 0; i < 4; i++) {
            uCmd[i] = uClassic[i] + uRL[i];
        }
        
        // Actuator filter gain
        const alpha = Math.min(this.dt / Math.max(this.tau_rw, 1e-6), 1.0);
        
        // Step simulation with frame_skip - match Python exactly
        for (let _ = 0; _ < this.frameSkip; _++) {
            const t = this.t;
            this._applyDisturbance(t);
            
            // Preserve user injection if allowed (Python: if not allow_user, clear ctrl)
            const ctrl = this.data.ctrl;
            if (!this.allowUserInjection) {
                // Only clear if user injection is not allowed
                ctrl[this.actId.jx] = 0.0;
                ctrl[this.actId.jy] = 0.0;
            }
            // If allowUserInjection is true, the torque set by controls is preserved here
            
            // Filtered reaction wheel commands
            const rwKeys = ['rw1', 'rw3', 'rw2', 'rw4'];
            for (let i = 0; i < 4; i++) {
                this.u_phys[i] += alpha * (uCmd[i] - this.u_phys[i]);
                ctrl[this.actId[rwKeys[i]]] = this._slew(this.actId[rwKeys[i]], this.u_phys[i]);
            }
            
            // Step MuJoCo simulation - this applies all controls including user torque
            // mj_step updates qpos, qvel based on applied controls
            // Note: mj_step internally calls mj_forward, so forward kinematics are updated
            this.mujoco.mj_step(this.model, this.data);
            this.t += this.dt;
        }
        
        this.stepCount++;
        
        // Compute reward, terminated, truncated - match Python exactly
        const qpos = this.data.qpos;
        const qvel = this.data.qvel;
        const jx = qpos[this.adr.jx];
        const jy = qpos[this.adr.jy];
        const ox = qvel[this.adr.jx];
        const oy = qvel[this.adr.jy];
        const w = [
            qvel[this.adr.rw1],
            qvel[this.adr.rw2],
            qvel[this.adr.rw3],
            qvel[this.adr.rw4]
        ];
        
        // Reward computation - match Python exactly
        const ang_cost = this.w_ang * (jx * jx + jy * jy);
        const rate_cost = this.w_rate * (ox * ox + oy * oy);
        const wheel_cost = this.w_wheel * (w[0]*w[0] + w[1]*w[1] + w[2]*w[2] + w[3]*w[3]);
        const ctrl_cost = this.w_ctrl * (uCmd[0]*uCmd[0] + uCmd[1]*uCmd[1] + uCmd[2]*uCmd[2] + uCmd[3]*uCmd[3]);
        const residual_cost = this.residualRL 
            ? this.w_residual * (uRL[0]*uRL[0] + uRL[1]*uRL[1] + uRL[2]*uRL[2] + uRL[3]*uRL[3])
            : 0.0;
        const reward = -(ang_cost + rate_cost + wheel_cost + ctrl_cost + residual_cost);
        
        // Termination conditions - match Python exactly
        // terminated = bool(abs(jx) > 0.7 or abs(jy) > 0.7 or any(abs(x) > 400 for x in w))
        const terminated = Math.abs(jx) > 0.7 || Math.abs(jy) > 0.7 || 
                           w.some(x => Math.abs(x) > 400);
        const truncated = this.stepCount >= this.maxSteps;
        
        // Return observation AFTER stepping - this ensures model sees current state
        // _obs() will call mj_forward internally to ensure qpos/qvel are current
        const obs = this._obs();
        
        // Return Gymnasium-style tuple: (obs, reward, terminated, truncated, info)
        return { obs, reward, terminated, truncated, info: {} };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NeedleRWEnv, ClassicRateDamping };
}

