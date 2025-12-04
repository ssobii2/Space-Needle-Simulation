/**
 * MuJoCo.js Viewer Wrapper
 * Provides 3D visualization and camera controls for MuJoCo simulation
 */

class MuJoCoViewer {
    constructor(canvas, model, data, mujoco) {
        this.canvas = canvas;
        this.model = model;
        this.data = data;
        this.mujoco = mujoco;
        
        if (!window.THREE) {
            throw new Error('Three.js not loaded. Please ensure Three.js is loaded before initializing viewer.');
        }
        
        // Camera state - view to show needle straight/vertical like second image
        this.camera = {
            azimuth: 90.0,     // View from side (along Y axis) to show needle straight
            elevation: -25.0,  // Look down at angle to see needle and wheels
            distance: 0.8,     // Zoomed out for better overview
            lookat: [0.0, 0.0, -0.05]  // Focus on pivot area
        };
        
        // Mouse interaction state
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        
        // Initialize Three.js renderer
        this._initRenderer();
        this._setupCameraControls();
    }
    
    _initRenderer() {
        const THREE = window.THREE;
        
        // Create Three.js scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0.2, 0.2, 0.2);
        
        // Get canvas container dimensions (not just canvas element)
        const container = this.canvas.parentElement;
        const width = container ? container.clientWidth : (this.canvas.clientWidth || 800);
        const height = container ? container.clientHeight : (this.canvas.clientHeight || 600);
        
        // Set canvas size to match container
        this.canvas.width = width;
        this.canvas.height = height;
        
        this.camera3d = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false
        });
        this.renderer.setSize(width, height, false); // false = don't update style
        this.renderer.setPixelRatio(window.devicePixelRatio);
        
        // Add lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);
        
        // Create geometry meshes based on XML model structure
        this._createMeshes();
        
        console.log('MuJoCo renderer initialized');
    }
    
    _createMeshes() {
        const THREE = window.THREE;
        
        // Create meshes for each body based on XML structure
        // Base (invisible, just for hierarchy)
        this.baseGroup = new THREE.Group();
        this.scene.add(this.baseGroup);
        
        // Gimbal X group
        this.gimbalXGroup = new THREE.Group();
        this.baseGroup.add(this.gimbalXGroup);
        
        // Gimbal Y group
        this.gimbalYGroup = new THREE.Group();
        this.gimbalXGroup.add(this.gimbalYGroup);
        
        // Needle (capsule: fromto="0 0 0 0 0 -0.35" size="0.006")
        // Use cylinder as fallback if CapsuleGeometry not available
        let needleGeometry;
        if (THREE.CapsuleGeometry) {
            needleGeometry = new THREE.CapsuleGeometry(0.006, 0.35);
        } else {
            // Fallback to cylinder
            needleGeometry = new THREE.CylinderGeometry(0.006, 0.006, 0.35, 16);
        }
        const needleMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x888888,  // Uniform grey to match screenshot
            metalness: 0.3,
            roughness: 0.7
        });
        this.needleMesh = new THREE.Mesh(needleGeometry, needleMaterial);
        this.needleMesh.rotation.x = Math.PI / 2; // Rotate to vertical
        this.needleMesh.position.set(0, 0, -0.175); // Center of capsule
        this.gimbalYGroup.add(this.needleMesh);
        
        // Pivot ball (sphere: pos="0 0 0.00" size="0.025")
        const pivotGeometry = new THREE.SphereGeometry(0.025, 16, 16);
        const pivotMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x888888,  // Uniform grey to match screenshot
            metalness: 0.5,
            roughness: 0.5
        });
        this.pivotMesh = new THREE.Mesh(pivotGeometry, pivotMaterial);
        this.pivotMesh.position.set(0, 0, 0);
        this.gimbalYGroup.add(this.pivotMesh);
        
        // Head group (pos="0 0 0.08")
        this.headGroup = new THREE.Group();
        this.headGroup.position.set(0, 0, 0.08);
        this.gimbalYGroup.add(this.headGroup);
        
        // Reaction wheels (cylinders)
        const rwGeometry = new THREE.CylinderGeometry(0.01, 0.01, 0.06, 16);
        const rwMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x888888,  // Uniform grey to match screenshot (same as needle and pivot)
            metalness: 0.3,
            roughness: 0.7
        });
        
        // RW1 (pos="0 0.06 0", axis="1 0 0")
        this.rw1Mesh = new THREE.Mesh(rwGeometry, rwMaterial);
        this.rw1Mesh.rotation.z = Math.PI / 2; // Rotate to align with X axis
        this.rw1Mesh.position.set(0, 0.06, 0);
        this.headGroup.add(this.rw1Mesh);
        
        // RW3 (pos="0 -0.06 0", axis="1 0 0")
        this.rw3Mesh = new THREE.Mesh(rwGeometry, rwMaterial);
        this.rw3Mesh.rotation.z = Math.PI / 2;
        this.rw3Mesh.position.set(0, -0.06, 0);
        this.headGroup.add(this.rw3Mesh);
        
        // RW2 (pos="0.06 0 0", axis="0 1 0")
        this.rw2Mesh = new THREE.Mesh(rwGeometry, rwMaterial);
        this.rw2Mesh.rotation.x = Math.PI / 2; // Rotate to align with Y axis
        this.rw2Mesh.position.set(0.06, 0, 0);
        this.headGroup.add(this.rw2Mesh);
        
        // RW4 (pos="-0.06 0 0", axis="0 1 0")
        this.rw4Mesh = new THREE.Mesh(rwGeometry, rwMaterial);
        this.rw4Mesh.rotation.x = Math.PI / 2;
        this.rw4Mesh.position.set(-0.06, 0, 0);
        this.headGroup.add(this.rw4Mesh);
    }
    
    _setupCameraControls() {
        // Mouse down - only left button for rotation
        this.canvas.addEventListener('mousedown', (e) => {
            // Only allow left mouse button (button 0)
            if (e.button === 0) {
                this.isDragging = true;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
                this.canvas.style.cursor = 'grabbing';
                e.preventDefault();
            }
        });
        
        // Mouse move - only rotation, no panning
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const width = this.canvas.width;
                const height = this.canvas.height;
                const deltaX = e.clientX - this.lastMouseX;
                const deltaY = e.clientY - this.lastMouseY;
                
                // Normalize by window size
                const dx = deltaX / width;
                const dy = deltaY / height;
                
                // Only rotate camera around fixed lookat point (no panning)
                // Horizontal drag: rotate azimuth (rotate around vertical axis)
                this.camera.azimuth += dx * 180; // Scale to degrees
                
                // Vertical drag: rotate elevation (look up/down)
                this.camera.elevation += dy * 180; // Scale to degrees
                this.camera.elevation = Math.max(-90, Math.min(90, this.camera.elevation));
                
                // Keep lookat point fixed - don't allow panning
                // this.camera.lookat remains at [0.0, 0.0, -0.05]
                
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
                
                e.preventDefault();
            }
        });
        
        // Mouse up
        this.canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.canvas.style.cursor = 'default';
        });
        
        // Mouse leave
        this.canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
            this.canvas.style.cursor = 'default';
        });
        
        // Mouse wheel for zoom
        this.canvas.addEventListener('wheel', (e) => {
            const zoomSpeed = 0.1;
            // Scroll up zooms in (decreases distance), scroll down zooms out (increases distance)
            this.camera.distance -= e.deltaY * zoomSpeed * 0.01;
            this.camera.distance = Math.max(0.1, Math.min(10.0, this.camera.distance));
            
            e.preventDefault();
        });
    }
    
    render() {
        if (!this.renderer || !this.scene || !this.camera3d) {
            return;
        }
        
        try {
            // Update forward kinematics to get body positions/orientations
            this.mujoco.mj_forward(this.model, this.data);
            
            // Update camera position based on azimuth/elevation/distance
            // Match MuJoCo viewer camera system:
            // - Azimuth: rotation around Z axis (0° = +X, 90° = +Y)
            // - Elevation: angle from horizontal plane (negative = looking down)
            const THREE = window.THREE;
            const azimuthRad = (this.camera.azimuth * Math.PI) / 180;
            const elevationRad = (this.camera.elevation * Math.PI) / 180;
            
            // Spherical to Cartesian conversion
            const cosElev = Math.cos(elevationRad);
            const x = this.camera.distance * cosElev * Math.cos(azimuthRad);
            const y = this.camera.distance * cosElev * Math.sin(azimuthRad);
            const z = this.camera.distance * Math.sin(elevationRad);
            
            this.camera3d.position.set(
                this.camera.lookat[0] + x,
                this.camera.lookat[1] + y,
                this.camera.lookat[2] + z
            );
            this.camera3d.lookAt(
                this.camera.lookat[0],
                this.camera.lookat[1],
                this.camera.lookat[2]
            );
            
            // Update mesh positions/rotations from MuJoCo data
            // IMPORTANT: qpos values are joint angles in radians (absolute, not incremental)
            // We directly set rotations from qpos - Three.js will handle the transformation
            const qpos = this.data.qpos;
            
            // Joint indices: jx=0, jy=1, j_rw1=2, j_rw3=3, j_rw2=4, j_rw4=5
            // CRITICAL: These must match the indices used in simulation.js (_getJointIndices)
            if (qpos.length >= 6) {
                // Gimbal X rotation around X axis (pitch)
                // qpos[0] is the absolute rotation angle in radians
                // In MuJoCo: jx rotates around axis="1 0 0" (X axis)
                const jxAngle = qpos[0];
                
                // Gimbal Y rotation around Y axis (yaw)
                // qpos[1] is the absolute rotation angle in radians
                // In MuJoCo: jy rotates around axis="0 1 0" (Y axis)
                const jyAngle = qpos[1];
                
                // Apply rotations directly - Three.js will handle the transformation
                // The hierarchy ensures Y rotation is applied in local frame after X rotation
                this.gimbalXGroup.rotation.x = jxAngle;
                this.gimbalYGroup.rotation.y = jyAngle;
                
                // Reaction wheel rotations
                // qpos[2-5] are absolute joint angles in radians
                // Three.js applies rotations in order: Z, Y, X (Euler order)
                // For RW1 and RW3: initial Z rotation (PI/2) aligns cylinder with X axis
                // Then we add the joint rotation around the local X axis
                const initialRW1Z = Math.PI / 2; // Set in _createMeshes to align with X axis
                this.rw1Mesh.rotation.set(qpos[2], 0, initialRW1Z); // X, Y, Z order
                this.rw3Mesh.rotation.set(qpos[3], 0, initialRW1Z);
                
                // For RW2 and RW4: initial X rotation (PI/2) aligns cylinder with Y axis
                // Then we add the joint rotation around the local Y axis
                const initialRW2X = Math.PI / 2; // Set in _createMeshes to align with Y axis
                this.rw2Mesh.rotation.set(initialRW2X, qpos[4], 0); // X, Y, Z order
                this.rw4Mesh.rotation.set(initialRW2X, qpos[5], 0);
            }
            
            // Render the scene
            this.renderer.render(this.scene, this.camera3d);
        } catch (e) {
            console.warn('Rendering error:', e);
        }
    }
    
    resize(width, height) {
        if (this.canvas) {
            this.canvas.width = width;
            this.canvas.height = height;
        }
        
        if (this.camera3d) {
            this.camera3d.aspect = width / height;
            this.camera3d.updateProjectionMatrix();
        }
        
        if (this.renderer) {
            this.renderer.setSize(width, height, false); // false = don't update CSS style
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MuJoCoViewer;
}

