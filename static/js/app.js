/**
 * Main Application Entry Point
 * Initializes MuJoCo simulation, ONNX model, viewer, and controls
 */

// MuJoCo XML model (extracted from STM stabiliser.py)
const MUJOCO_XML = `<mujoco model="needle_rw">
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
</mujoco>`;

class NeedleStabilizerApp {
    constructor() {
        this.model = null;
        this.data = null;
        this.env = null;
        this.onnxModel = null;
        this.viewer = null;
        this.controls = null;
        this.obs = null;
        this.running = false;
        this.animationFrameId = null;
        this.stepCount = 0;
        
        // UI elements
        this.canvas = document.getElementById('simCanvas');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.statusText = document.getElementById('statusText');
        this.statusDot = document.getElementById('statusDot');
    }
    
    async init() {
        try {
            this.updateStatus('initializing', 'Initializing...');
            
            // Load MuJoCo model
            await this._loadMuJoCoModel();
            
            // Load ONNX model
            await this._loadONNXModel();
            
            // Initialize environment
            this._initEnvironment();
            
            // Initialize viewer
            this._initViewer();
            
            // Initialize controls
            this._initControls();
            
            // Setup UI event handlers
            this._setupUIHandlers();
            
            this.updateStatus('ready', 'Ready');
            console.log('Application initialized successfully');
        } catch (error) {
            console.error('Initialization error:', error);
            const errorMsg = error.message || error.toString() || 'Unknown error';
            this.updateStatus('error', 'Initialization Error');
            alert('Failed to initialize application: ' + errorMsg);
        }
    }
    
    async _loadMuJoCoModel() {
        // Load MuJoCo WASM module using mujoco_wasm_contrib
        if (!window.loadMuJoCo) {
            throw new Error('MuJoCo loader not available. Please ensure mujoco_wasm_contrib is loaded.');
        }
        
        console.log('Loading MuJoCo WASM module...');
        const mujoco = await window.loadMuJoCo();
        
        // Set up Emscripten's Virtual File System
        mujoco.FS.mkdir('/working');
        mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
        
        // Write XML model to virtual file system
        mujoco.FS.writeFile('/working/model.xml', MUJOCO_XML);
        
        // Load model and create data using correct API
        this.mujoco = mujoco;
        this.model = mujoco.MjModel.loadFromXML('/working/model.xml');
        this.data = new mujoco.MjData(this.model);
        
        console.log('MuJoCo model loaded');
    }
    
    async _loadONNXModel() {
        this.onnxModel = new ONNXModel();
        await this.onnxModel.load('/ppo_needle_rw_residual.onnx');
        console.log('ONNX model loaded');
    }
    
    _initEnvironment() {
        this.env = new NeedleRWEnv(
            this.model,
            this.data,
            this.mujoco,  // mujoco module
            true,  // residualRL
            5,     // frameSkip - match Python visualize (frame_skip=5) for proper movement scale
            5.0,   // episodeLen (match Python visualize: default 5.0)
            true   // allowUserInjection
        );
        
        // Reset environment
        const resetResult = this.env.reset(true);
        this.obs = resetResult.obs || resetResult; // Handle both old and new return format
        // Update forward kinematics after reset
        this.mujoco.mj_forward(this.model, this.data);
        console.log(`Environment initialized (maxSteps: ${this.env.maxSteps})`);
    }
    
    _initViewer() {
        this.viewer = new MuJoCoViewer(this.canvas, this.model, this.data, this.mujoco);
        
        // Handle canvas resize
        const resizeObserver = new ResizeObserver(() => {
            const rect = this.canvas.getBoundingClientRect();
            this.viewer.resize(rect.width, rect.height);
        });
        resizeObserver.observe(this.canvas);
        
        // Initial render to show the scene
        this.viewer.render();
        
        console.log('Viewer initialized');
    }
    
    _initControls() {
        this.controls = new Controls(this.env, () => {
            // When controls change, ensure simulation is running and render immediately
            if (!this.running) {
                // Auto-start simulation when user interacts
                this.start();
            }
            // Force immediate physics steps to apply torque (don't wait for model)
            // This matches Python: torque is applied directly, physics moves the needle immediately,
            // then model responds in the next loop iteration
            if (this.running) {
                // CRITICAL: Apply torque and step physics immediately WITHOUT model prediction
                // This gives rapid needle movement like Python viewer
                // The model will respond in the next main loop iteration
                const numQuickPhysicsSteps = 10; // Step physics more times for immediate visual feedback
                
                for (let i = 0; i < numQuickPhysicsSteps; i++) {
                    // Step physics directly (user torque is already set in data.ctrl)
                    // This is like doing mj_step without model action - just apply user torque
                    this.mujoco.mj_step(this.model, this.data);
                }
                
                // Update forward kinematics and render immediately
                this.mujoco.mj_forward(this.model, this.data);
                this._updateUI();
                this.viewer.render();
            } else {
                // Just render if not running
                this.viewer.render();
            }
        });
        console.log('Controls initialized');
    }
    
    _setupUIHandlers() {
        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());
        this.resetBtn.addEventListener('click', () => this.reset());
    }
    
    start() {
        if (this.running) return;
        
        this.running = true;
        this.startBtn.disabled = true;
        this.stopBtn.disabled = false;
        this.updateStatus('running', 'Running');
        
        // Start simulation loop
        this._simulationLoop();
    }
    
    stop() {
        this.running = false;
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.updateStatus('stopped', 'Stopped');
        
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }
    
    reset() {
        this.stop();
        this.stepCount = 0;
        const resetResult = this.env.reset(true);
        this.obs = resetResult.obs || resetResult; // Handle both old and new return format
        // Update forward kinematics after reset to ensure visualization is correct
        this.mujoco.mj_forward(this.model, this.data);
        // Force render to show reset state
        this.viewer.render();
        this._updateUI();
    }
    
    async _simulationLoop() {
        if (!this.running) return;
        
        try {
            // CRITICAL: Run multiple steps per frame to match Python viewer's high frequency
            // Python viewer runs in a tight loop at very high frequency (likely 100+ fps)
            // We need to do multiple model predictions and steps per animation frame
            const stepsPerFrame = 5; // Do 5 steps per animation frame for faster response
            
            for (let i = 0; i < stepsPerFrame; i++) {
                // CRITICAL: Update forward kinematics FIRST to ensure state is current
                // This is especially important when user torque is applied via sliders
                this.mujoco.mj_forward(this.model, this.data);
                
                // CRITICAL: Get fresh observation BEFORE calling model
                // The observation must reflect the CURRENT state of the simulation
                const currentObs = this.env._obs();
                
                // Debug: Log observation values occasionally (every 60 steps)
                if (this.stepCount % 60 === 0) {
                    const qpos = this.data.qpos;
                    const qvel = this.data.qvel;
                    const ctrl = this.data.ctrl;
                    console.log(`[Step ${this.stepCount}] jx: ${qpos[this.env.adr.jx].toFixed(4)}, jy: ${qpos[this.env.adr.jy].toFixed(4)}, m_jx: ${ctrl[this.env.actId.jx].toFixed(4)}, m_jy: ${ctrl[this.env.actId.jy].toFixed(4)}`);
                    console.log(`[Step ${this.stepCount}] obs: [${Array.from(currentObs).map(v => v.toFixed(4)).join(', ')}]`);
                    console.log(`[Step ${this.stepCount}] RW ctrl: m_rw1=${ctrl[this.env.actId.rw1].toFixed(6)}, m_rw2=${ctrl[this.env.actId.rw2].toFixed(6)}, m_rw3=${ctrl[this.env.actId.rw3].toFixed(6)}, m_rw4=${ctrl[this.env.actId.rw4].toFixed(6)}`);
                    console.log(`[Step ${this.stepCount}] RW vel: w1=${qvel[this.env.adr.rw1].toFixed(4)}, w2=${qvel[this.env.adr.rw2].toFixed(4)}, w3=${qvel[this.env.adr.rw3].toFixed(4)}, w4=${qvel[this.env.adr.rw4].toFixed(4)}`);
                }
                
                // Get action from model (always call model to stabilize)
                // Match Python: action, _ = model.predict(obs, deterministic=True)
                // Python runs in a tight loop: while viewer.is_running(): model.predict() -> env.step()
                const [action] = await this.onnxModel.predict(currentObs, true);
                
                // Debug: Log action values occasionally
                if (this.stepCount % 60 === 0) {
                    console.log(`[Step ${this.stepCount}] action: [${Array.from(action).map(v => v.toFixed(4)).join(', ')}]`);
                }
                
                // Step environment (this applies the model action + user torque)
                // Match Python: obs, reward, term, trunc, _ = env.step(action)
                const stepResult = this.env.step(action);
                this.obs = stepResult.obs || stepResult; // Handle both old and new return format
                const reward = stepResult.reward !== undefined ? stepResult.reward : 0;
                const terminated = stepResult.terminated || false;
                const truncated = stepResult.truncated || false;
                this.stepCount++;
                
                // Handle termination/truncation immediately
                if (terminated || truncated) {
                    // Debug: Log termination events
                    const qpos = this.data.qpos;
                    const qvel = this.data.qvel;
                    const jx = qpos[this.env.adr.jx];
                    const jy = qpos[this.env.adr.jy];
                    const w = [
                        qvel[this.env.adr.rw1],
                        qvel[this.env.adr.rw2],
                        qvel[this.env.adr.rw3],
                        qvel[this.env.adr.rw4]
                    ];
                    console.log(`[Step ${this.stepCount}] EPISODE END: terminated=${terminated}, truncated=${truncated}`);
                    console.log(`[Step ${this.stepCount}] Final state: jx=${jx.toFixed(4)}, jy=${jy.toFixed(4)}, w=[${w.map(x => x.toFixed(2)).join(', ')}]`);
                    if (terminated) {
                        if (Math.abs(jx) > 0.7) console.log(`  -> jx exceeded limit: ${Math.abs(jx).toFixed(4)} > 0.7`);
                        if (Math.abs(jy) > 0.7) console.log(`  -> jy exceeded limit: ${Math.abs(jy).toFixed(4)} > 0.7`);
                        if (w.some(x => Math.abs(x) > 400)) {
                            const exceeded = w.findIndex(x => Math.abs(x) > 400);
                            console.log(`  -> wheel ${exceeded} exceeded limit: ${Math.abs(w[exceeded]).toFixed(2)} > 400`);
                        }
                    }
                    if (truncated) {
                        console.log(`  -> max steps reached: ${this.stepCount} >= ${this.env.maxSteps}`);
                    }
                    
                    // Reset environment
                    const resetResult = this.env.reset(true);
                    this.obs = resetResult.obs || resetResult;
                    this.stepCount = 0;
                    console.log(`[Step ${this.stepCount}] Environment reset after episode end`);
                    break; // Exit inner loop after reset
                }
            }
            
            // Update UI with current state
            this._updateUI();
            
            // Render (render calls mj_forward internally)
            this.viewer.render();
            
            // Continue loop immediately - Python runs continuously without delays
            // Use requestAnimationFrame to sync with display refresh rate (~60fps)
            // But we do 5 steps per frame, so effectively ~300 model predictions/second
            this.animationFrameId = requestAnimationFrame(() => this._simulationLoop());
        } catch (error) {
            console.error('Simulation loop error:', error);
            this.stop();
        }
    }
    
    _updateUI() {
        const qpos = this.data.qpos;
        const qvel = this.data.qvel;
        const ctrl = this.data.ctrl;
        
        // Update joint positions
        document.getElementById('jx').textContent = qpos[this.env.adr.jx].toFixed(3);
        document.getElementById('jy').textContent = qpos[this.env.adr.jy].toFixed(3);
        document.getElementById('j_rw1').textContent = qpos[this.env.adr.rw1].toFixed(3);
        document.getElementById('j_rw3').textContent = qpos[this.env.adr.rw3].toFixed(3);
        document.getElementById('j_rw2').textContent = qpos[this.env.adr.rw2].toFixed(3);
        document.getElementById('j_rw4').textContent = qpos[this.env.adr.rw4].toFixed(3);
        
        // Update control values (reaction wheel torques)
        document.getElementById('m_rw1').textContent = ctrl[this.env.actId.rw1].toFixed(3);
        document.getElementById('m_rw3').textContent = ctrl[this.env.actId.rw3].toFixed(3);
        document.getElementById('m_rw2').textContent = ctrl[this.env.actId.rw2].toFixed(3);
        document.getElementById('m_rw4').textContent = ctrl[this.env.actId.rw4].toFixed(3);
        
        // Update m_jx and m_jy display to show current control values (torques being applied)
        document.getElementById('m_jx').textContent = ctrl[this.env.actId.jx].toFixed(3);
        document.getElementById('m_jy').textContent = ctrl[this.env.actId.jy].toFixed(3);
        
        // Don't update slider positions from state - sliders are for setting torque, not displaying joint angles
    }
    
    updateStatus(status, text) {
        this.statusText.textContent = text;
        this.statusDot.className = 'status-dot ' + status;
    }
}

// Initialize app when DOM is ready and MuJoCo is loaded
let app;
function initializeApp() {
    if (window.loadMuJoCo) {
        app = new NeedleStabilizerApp();
        app.init();
    } else {
        // Wait for MuJoCo to be ready
        window.addEventListener('mujoco-ready', () => {
            app = new NeedleStabilizerApp();
            app.init();
        });
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

