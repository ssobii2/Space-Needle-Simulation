/**
 * UI Controls Handler
 * Manages slider controls for manual torque injection
 */

class Controls {
    constructor(env, updateCallback) {
        this.env = env;
        this.updateCallback = updateCallback;
        this.isUserDragging = false;
        
        // Get slider elements
        this.sliderMjx = document.getElementById('slider_m_jx');
        this.sliderMjy = document.getElementById('slider_m_jy');
        this.displayMjx = document.getElementById('m_jx');
        this.displayMjy = document.getElementById('m_jy');
        
        if (!this.sliderMjx || !this.sliderMjy) {
            console.warn('Slider elements not found');
            return;
        }
        
        this._setupEventListeners();
    }
    
    _setupEventListeners() {
        // m_jx slider
        const handleMjxInput = () => {
            this.isUserDragging = true;
            const value = parseFloat(this.sliderMjx.value);
            this.displayMjx.textContent = value.toFixed(3);
            this._applyTorque(value, null);
        };
        
        const handleMjxRelease = () => {
            this.isUserDragging = false;
            // Auto-reset to zero on release
            this.sliderMjx.value = 0;
            this.displayMjx.textContent = '0.000';
            this._applyTorque(0, null);
        };
        
        this.sliderMjx.addEventListener('input', handleMjxInput);
        this.sliderMjx.addEventListener('change', handleMjxInput);
        this.sliderMjx.addEventListener('mouseup', handleMjxRelease);
        this.sliderMjx.addEventListener('touchend', handleMjxRelease);
        this.sliderMjx.addEventListener('mouseleave', () => {
            if (this.isUserDragging) {
                handleMjxRelease();
            }
        });
        
        // m_jy slider
        const handleMjyInput = () => {
            this.isUserDragging = true;
            const value = parseFloat(this.sliderMjy.value);
            this.displayMjy.textContent = value.toFixed(3);
            this._applyTorque(null, value);
        };
        
        const handleMjyRelease = () => {
            this.isUserDragging = false;
            // Auto-reset to zero on release
            this.sliderMjy.value = 0;
            this.displayMjy.textContent = '0.000';
            this._applyTorque(null, 0);
        };
        
        this.sliderMjy.addEventListener('input', handleMjyInput);
        this.sliderMjy.addEventListener('change', handleMjyInput);
        this.sliderMjy.addEventListener('mouseup', handleMjyRelease);
        this.sliderMjy.addEventListener('touchend', handleMjyRelease);
        this.sliderMjy.addEventListener('mouseleave', () => {
            if (this.isUserDragging) {
                handleMjyRelease();
            }
        });
    }
    
    _applyTorque(torqueX, torqueY) {
        if (!this.env) return;
        
        // Set control values directly (like MuJoCo viewer)
        const ctrl = this.env.data.ctrl;
        if (torqueX !== null) {
            ctrl[this.env.actId.jx] = torqueX;
        }
        if (torqueY !== null) {
            ctrl[this.env.actId.jy] = torqueY;
        }
        
        // Trigger update callback if provided
        if (this.updateCallback) {
            this.updateCallback();
        }
    }
    
    updateFromState(state) {
        // Update sliders to match current state (when model stabilizes)
        if (!this.isUserDragging) {
            if (state.jx !== undefined) {
                this.sliderMjx.value = state.jx;
                this.displayMjx.textContent = state.jx.toFixed(3);
            }
            if (state.jy !== undefined) {
                this.sliderMjy.value = state.jy;
                this.displayMjy.textContent = state.jy.toFixed(3);
            }
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Controls;
}

