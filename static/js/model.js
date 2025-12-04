/**
 * ONNX.js Model Loader and Inference
 * Loads and runs PPO model inference in the browser
 */

class ONNXModel {
    constructor() {
        this.session = null;
        this.loaded = false;
    }
    
    async load(modelPath) {
        try {
            // Check for ONNX Runtime Web (ort) - the modern replacement for ONNX.js
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime Web not loaded. Please include ort.min.js');
            }
            
            console.log(`Loading ONNX model from ${modelPath}...`);
            
            // Fetch the model file first, then create session from ArrayBuffer
            // This is more reliable than passing a URL directly
            const response = await fetch(modelPath);
            if (!response.ok) {
                throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
            }
            
            const arrayBuffer = await response.arrayBuffer();
            console.log(`Model file fetched, size: ${arrayBuffer.byteLength} bytes`);
            
            // Create inference session using ONNX Runtime Web
            // ONNX Runtime Web supports opset 18
            console.log('Creating ONNX Runtime session...');
            console.log('ArrayBuffer size:', arrayBuffer.byteLength);
            
            // Try multiple execution provider configurations
            const sessionAttempts = [
                {
                    name: 'webgl',
                    options: {
                        executionProviders: ['webgl'],
                        graphOptimizationLevel: 'disabled',
                        enableCpuMemArena: false,
                        enableMemPattern: false,
                        logSeverityLevel: 0,
                    },
                },
                {
                    name: 'wasm',
                    options: {
                        executionProviders: ['wasm'],
                        graphOptimizationLevel: 'disabled',
                        enableCpuMemArena: false,
                        enableMemPattern: false,
                        logSeverityLevel: 0,
                    },
                },
                {
                    name: 'wasm (single-thread, no SIMD)',
                    options: {
                        executionProviders: ['wasm'],
                        graphOptimizationLevel: 'disabled',
                        enableCpuMemArena: false,
                        enableMemPattern: false,
                        logSeverityLevel: 0,
                        extra: {
                            session: {
                                interOpNumThreads: 1,
                                intraOpNumThreads: 1,
                            },
                        },
                    },
                },
            ];
            
            let sessionCreated = false;
            let lastError = null;
            
            for (const attempt of sessionAttempts) {
                try {
                    console.log(`Attempting ONNX session with ${attempt.name} backend...`);
                    this.session = await ort.InferenceSession.create(arrayBuffer, attempt.options);
                    console.log(`ONNX Runtime session created successfully with ${attempt.name}`);
                    sessionCreated = true;
                    break;
                } catch (sessionError) {
                    console.error(`Failed to create session with ${attempt.name}:`, sessionError);
                    lastError = sessionError;
                }
            }
            
            if (!sessionCreated) {
                throw new Error(`All ONNX session creation attempts failed. Last error: ${lastError?.message || lastError}`);
            }
            
            // Get input/output information
            const inputNames = this.session.inputNames;
            const outputNames = this.session.outputNames;
            
            console.log('Model loaded successfully');
            console.log('Input names:', inputNames);
            console.log('Output names:', outputNames);
            
            this.loaded = true;
            return true;
        } catch (error) {
            console.error('Failed to load ONNX model:', error);
            // Provide more detailed error information
            let errorDetails = 'Unknown error';
            if (error.message) {
                errorDetails = error.message;
                console.error('Error message:', error.message);
            }
            if (error.code) {
                errorDetails += ` (code: ${error.code})`;
                console.error('Error code:', error.code);
            }
            if (error.name) {
                console.error('Error name:', error.name);
            }
            // Create a more informative error
            const detailedError = new Error(`Failed to load ONNX model: ${errorDetails}`);
            detailedError.originalError = error;
            throw detailedError;
        }
    }
    
    async predict(observation, deterministic = true) {
        if (!this.loaded || !this.session) {
            throw new Error('Model not loaded. Call load() first.');
        }
        
        try {
            // Convert observation to tensor
            // Observation shape: [8] -> [1, 8] for batch
            // Match Python: obs is np.array([jx, jy, ox, oy, w1, w3, w2, w4], dtype=np.float32)
            let obsArray;
            if (observation instanceof Float32Array) {
                obsArray = Array.from(observation);
            } else if (Array.isArray(observation)) {
                obsArray = observation;
            } else {
                obsArray = Array.from(observation);
            }
            
            // Ensure exactly 8 elements
            if (obsArray.length !== 8) {
                throw new Error(`Observation must have 8 elements, got ${obsArray.length}`);
            }
            
            const obsTensor = new ort.Tensor('float32', new Float32Array(obsArray), [1, 8]);
            
            // Run inference
            const feeds = { observation: obsTensor };
            const results = await this.session.run(feeds);
            
            // Extract action (first output)
            // Python: action, _ = model.predict(obs, deterministic=True)
            const actionTensor = results[this.session.outputNames[0]];
            const actionArray = Array.from(actionTensor.data);
            
            // Return action and logits (if available) - match Python return format
            return [actionArray, null];
        } catch (error) {
            console.error('Model inference error:', error);
            throw error;
        }
    }
    
    isLoaded() {
        return this.loaded;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ONNXModel;
}

