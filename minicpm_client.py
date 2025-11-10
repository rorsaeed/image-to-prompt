# minicpm_client.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
import numpy as np
import math
import os
import time
from pathlib import Path
import mimetypes

class MiniCPMClient:
    """A client for MiniCPM-V models supporting both image and video analysis."""
    
    # Supported model configurations
    SUPPORTED_MODELS = {
        "MiniCPM-V-4.5-int4": {
            "model_id": "openbmb/MiniCPM-V-4_5-int4",
            "description": "4-bit quantized, reduced memory usage",
            "memory_efficient": True
        },
        "MiniCPM-V-4-int4": {
            "model_id": "openbmb/MiniCPM-V-4-int4", 
            "description": "4-bit quantized, reduced memory usage",
            "memory_efficient": True
        },
        "MiniCPM-V-4.5": {
            "model_id": "openbmb/MiniCPM-V-4_5",
            "description": "Latest V4.5 with enhanced capabilities",
            "memory_efficient": False
        },
        "MiniCPM-V-4.5-Abliterated": {
            "model_id": "huihui-ai/Huihui-MiniCPM-V-4_5-abliterated",
            "description": "Reduced safety filtering",
            "memory_efficient": False
        },
        "MiniCPM-V-4": {
            "model_id": "openbmb/MiniCPM-V-4",
            "description": "V4.0 full precision, higher quality",
            "memory_efficient": False
        }
    }
    
    def __init__(self, model_name="MiniCPM-V-4.5-int4", device="auto"):
        """
        Initialize MiniCPM client.
        
        Args:
            model_name: Name of the model to use (from SUPPORTED_MODELS)
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.model_config = self.SUPPORTED_MODELS.get(model_name)
        if not self.model_config:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_id = self.model_config["model_id"]
        self.device = self._determine_device(device)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Video processing parameters
        self.max_num_frames = 180
        self.max_num_packing = 3
        self.time_scale = 0.1
        self.default_fps = 3
        
        print(f"Initialized MiniCPM client with model: {model_name}")
        print(f"Model ID: {self.model_id}")
        print(f"Device: {self.device}")
    
    def _determine_device(self, device):
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the model and tokenizer."""
        if self.is_loaded:
            return
        
        print(f"Loading MiniCPM model: {self.model_name}")
        print("This may take a few minutes on first load...")
        
        try:
            # Load model with appropriate settings
            if self.device == "cuda":
                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    attn_implementation='sdpa',
                    torch_dtype=torch.bfloat16
                )
                self.model = self.model.eval().cuda()
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                self.model = self.model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            self.is_loaded = True
            print(f"✓ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def get_gpu_memory_info(self):
        """Get current GPU memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
        return "CUDA not available"
    
    def _clear_video_cache(self):
        """Clear any video processing related cached data."""
        # Clear any cached video frames or processing data
        cache_attrs = ['_cached_frames', '_cached_video_data', '_video_cache', 
                      '_frame_cache', '_processed_frames', '_video_tensors']
        for attr in cache_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
    
    def _clear_attention_caches(self):
        """Clear attention mechanism caches (SDPA specific)."""
        if self.model is not None:
            # Clear any cached attention weights or states
            try:
                # Recursively clear attention caches in all modules
                for module in self.model.modules():
                    # Clear any cached attention states
                    if hasattr(module, '_cached_attention'):
                        delattr(module, '_cached_attention')
                    if hasattr(module, 'attention_cache'):
                        delattr(module, 'attention_cache')
                    if hasattr(module, '_attention_cache'):
                        delattr(module, '_attention_cache')
                    if hasattr(module, 'past_key_values'):
                        delattr(module, 'past_key_values')
                    if hasattr(module, '_past_key_values'):
                        delattr(module, '_past_key_values')
                    
                    # Clear SDPA specific caches
                    if hasattr(module, 'sdpa_cache'):
                        delattr(module, 'sdpa_cache')
                    if hasattr(module, '_sdpa_cache'):
                        delattr(module, '_sdpa_cache')
            except Exception as e:
                # Ignore errors during cache clearing
                pass
    
    def _aggressive_cuda_cleanup(self):
        """Perform aggressive CUDA memory cleanup."""
        import gc
        
        # Clear transformers cache first
        try:
            from transformers import AutoModel, AutoTokenizer
            # Clear any cached models in transformers
            if hasattr(AutoModel, '_modules'):
                AutoModel._modules.clear()
            if hasattr(AutoTokenizer, '_modules'):
                AutoTokenizer._modules.clear()
        except:
            pass
        
        # Multiple rounds of cleanup for maximum effectiveness
        for i in range(5):  # Increased from 3 to 5 rounds
            # Synchronize all CUDA operations
            torch.cuda.synchronize()
            
            # Clear all caches
            torch.cuda.empty_cache()
            
            # Collect IPC handles
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
            
            # Clear any CUDA context cache
            if hasattr(torch.cuda, 'reset_max_memory_allocated'):
                torch.cuda.reset_max_memory_allocated()
            if hasattr(torch.cuda, 'reset_max_memory_cached'):
                torch.cuda.reset_max_memory_cached()
            
            # Force garbage collection between rounds
            gc.collect()
            
            # Small delay to allow cleanup to complete
            import time
            time.sleep(0.2)  # Increased delay
        
        # Final comprehensive cleanup
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Clear any remaining cached allocations
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            torch.cuda.reset_accumulated_memory_stats()
        
        # Try to clear CUDA context completely
        try:
            # Force CUDA context reset (this is aggressive)
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_snapshot'):
                torch.cuda.memory._record_memory_history(enabled=False)
        except:
            pass
        
        # Final garbage collection
        gc.collect()
    
    def unload_model(self):
        """Unload the model to free memory."""
        import gc
        
        # Print memory usage before cleanup
        if torch.cuda.is_available():
            print(f"Before unload: {self.get_gpu_memory_info()}")
        
        if self.model is not None:
            # Clear any attention caches (SDPA specific)
            self._clear_attention_caches()
            
            # Move model to CPU first to free GPU memory
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            # Clear any model state
            if hasattr(self.model, 'eval'):
                self.model.eval()
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear any cached variables that might hold references
        if hasattr(self, '_last_image'):
            delattr(self, '_last_image')
        if hasattr(self, '_last_video'):
            delattr(self, '_last_video')
        
        # Clear any video processing cache
        self._clear_video_cache()
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            # Comprehensive CUDA cleanup
            self._aggressive_cuda_cleanup()
            
            print(f"After unload: {self.get_gpu_memory_info()}")
        
        self.is_loaded = False
        print("Model unloaded and GPU memory aggressively cleared")
    
    def set_video_config(self, max_num_frames=None, max_num_packing=None, fps=None):
        """
        Configure video processing parameters.
        
        Args:
            max_num_frames: Maximum number of frames to analyze (default: 180)
            max_num_packing: Frame grouping parameter (1-6, default: 3)
            fps: Frames per second for sampling (default: 3, None for auto-calculation)
        """
        if max_num_frames is not None:
            self.max_num_frames = max_num_frames
        if max_num_packing is not None:
            self.max_num_packing = min(max(max_num_packing, 1), 6)
        if fps is not None:
            self.default_fps = fps
        
        print(f"Video config updated: MAX_NUM_FRAMES={self.max_num_frames}, MAX_NUM_PACKING={self.max_num_packing}, FPS={self.default_fps}")
    
    def _map_to_nearest_scale(self, values, scale):
        """Map values to nearest scale points."""
        tree = cKDTree(np.asarray(scale)[:, None])
        _, indices = tree.query(np.asarray(values)[:, None])
        return np.asarray(scale)[indices]
    
    def _group_array(self, arr, size):
        """Group array elements into chunks of specified size."""
        return [arr[i:i+size] for i in range(0, len(arr), size)]
    
    def _uniform_sample(self, l, n):
        """Uniformly sample n elements from list l."""
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    
    def encode_video(self, video_path, fps=None, force_packing=None):
        """
        Encode video for processing with MiniCPM.
        
        Args:
            video_path: Path to video file
            fps: Frames per second for sampling (None for auto-calculation)
            force_packing: Force specific packing number (1-6)
        
        Returns:
            tuple: (frames, frame_ts_id_group, metadata)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            video_fps = vr.get_avg_fps()
            video_duration = len(vr) / video_fps
            total_frames = len(vr)
            
            print(f"Video: {video_path}")
            print(f"Duration: {video_duration:.2f}s, Total frames: {total_frames}, Video FPS: {video_fps:.2f}")
            
            # Auto-calculate FPS if set to 0 or None
            print(f"  - Debug: Input fps = {fps}, type = {type(fps)}")
            if fps is None or fps == 0.0:
                # Comprehensive auto-calculation for maximum frame coverage
                # Factor 1: Maximum possible frames we can process (MAX_NUM_FRAMES * MAX_NUM_PACKING)
                max_processable_frames = self.max_num_frames * self.max_num_packing
                
                # Factor 2: Actual frames in video
                actual_frames = total_frames
                
                # Factor 3: Video duration
                duration = video_duration
                
                # Calculate optimal FPS for comprehensive coverage
                # Option 1: FPS to process all frames (if video has fewer frames than max processable)
                fps_for_all_frames = actual_frames / duration if duration > 0 else video_fps
                
                # Option 2: FPS to utilize maximum processing capacity
                fps_for_max_capacity = max_processable_frames / duration if duration > 0 else video_fps
                
                # Option 3: Native video FPS (to avoid upsampling)
                native_fps = video_fps
                
                print(f"  - Debug calculations:")
                print(f"    - fps_for_all_frames = {actual_frames} / {duration} = {fps_for_all_frames}")
                print(f"    - fps_for_max_capacity = {max_processable_frames} / {duration} = {fps_for_max_capacity}")
                print(f"    - native_fps = {native_fps}")
                print(f"    - actual_frames ({actual_frames}) <= max_processable_frames ({max_processable_frames})? {actual_frames <= max_processable_frames}")
                
                # Choose the optimal FPS to maximize frame utilization
                if actual_frames <= max_processable_frames:
                    # Video has fewer frames than we can process - analyze all frames
                    choose_fps = min(fps_for_all_frames, native_fps)
                    coverage_type = "ALL_FRAMES"
                    target_frames = actual_frames
                    print(f"    - Using ALL_FRAMES: min({fps_for_all_frames}, {native_fps}) = {choose_fps}")
                else:
                    # Video has more frames than we can process - use maximum capacity
                    choose_fps = fps_for_max_capacity  # Don't limit by native FPS for max capacity
                    coverage_type = "MAX_CAPACITY"
                    target_frames = max_processable_frames
                    print(f"    - Using MAX_CAPACITY: {fps_for_max_capacity} = {choose_fps}")
                
                print(f"Auto-calculating FPS for comprehensive analysis:")
                print(f"  - Video: {actual_frames} frames, {duration:.2f}s, native FPS: {native_fps:.2f}")
                print(f"  - Max processable: {max_processable_frames} frames ({self.max_num_frames} × {self.max_num_packing})")
                print(f"  - Coverage type: {coverage_type}")
                print(f"  - Target frames to analyze: {target_frames}")
                print(f"  - Calculated FPS: {choose_fps:.2f}")
            else:
                choose_fps = fps
                print(f"  - Using provided FPS: {choose_fps}")
            
            # Only apply minimum FPS if the calculated FPS is actually problematic (near zero)
            print(f"  - Debug: choose_fps = {choose_fps}, type = {type(choose_fps)}")
            if choose_fps < 0.01:
                print(f"  - Debug: choose_fps ({choose_fps}) < 0.01, applying override")
                choose_fps = 0.1
                print(f"  - Applied minimum FPS override: {choose_fps:.2f}")
            else:
                print(f"  - Debug: choose_fps ({choose_fps}) >= 0.01, no override needed")
            
            # Calculate optimal frame sampling and packing for comprehensive coverage
            total_desired_frames = round(video_duration * choose_fps)
            
            # Ensure we don't exceed the maximum processable frames
            max_processable = self.max_num_frames * self.max_num_packing
            total_desired_frames = min(total_desired_frames, max_processable)
            
            # Also ensure we don't exceed actual video frames
            total_desired_frames = min(total_desired_frames, total_frames)
            
            # Calculate optimal packing strategy
            if total_desired_frames <= self.max_num_frames:
                # No packing needed - can fit all desired frames in one batch
                packing_nums = 1
                choose_frames = total_desired_frames
            else:
                # Need packing - distribute frames across multiple batches
                packing_nums = math.ceil(total_desired_frames / self.max_num_frames)
                packing_nums = min(packing_nums, self.max_num_packing)  # Respect max packing limit
                choose_frames = total_desired_frames
            
            print(f"Frame selection strategy:")
            print(f"  - Desired frames: {total_desired_frames} (from {choose_fps:.2f} FPS × {video_duration:.2f}s)")
            print(f"  - Packing strategy: {packing_nums} batch(es)")
            print(f"  - Frames per batch: {choose_frames // packing_nums if packing_nums > 0 else choose_frames}")
            print(f"  - Total coverage: {(choose_frames / total_frames * 100):.1f}% of video frames")
            
            # Apply force_packing if specified
            if force_packing:
                packing_nums = min(force_packing, self.max_num_packing)
            
            # Ensure at least 1 frame is selected
            choose_frames = max(choose_frames, 1)
            
            # Calculate actual FPS with safety check
            actual_fps = choose_frames / video_duration if video_duration > 0 else 0
            
            print(f"Sampling config: FPS={choose_fps} → Actual FPS={actual_fps:.2f}")
            print(f"Frames to analyze: {choose_frames}, Packing: {packing_nums}")
            
            # Sample frames uniformly
            frame_idx = [i for i in range(0, len(vr))]
            frame_idx = np.array(self._uniform_sample(frame_idx, choose_frames))
            
            # Extract frames
            frames = vr.get_batch(frame_idx).asnumpy()
            
            # Calculate temporal IDs
            frame_idx_ts = frame_idx / video_fps
            scale = np.arange(0, video_duration, self.time_scale)
            frame_ts_id = self._map_to_nearest_scale(frame_idx_ts, scale) / self.time_scale
            frame_ts_id = frame_ts_id.astype(np.int32)
            
            assert len(frames) == len(frame_ts_id)
            
            # Convert to PIL Images
            frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
            frame_ts_id_group = self._group_array(frame_ts_id, packing_nums)
            
            metadata = {
                "video_path": video_path,
                "video_duration": video_duration,
                "video_fps": video_fps,
                "total_frames": total_frames,
                "analyzed_frames": len(frames),
                "sampling_fps": choose_fps,
                "actual_fps": actual_fps,
                "packing_nums": packing_nums
            }
            
            return frames, frame_ts_id_group, metadata
            
        except Exception as e:
            print(f"Error encoding video: {e}")
            raise
    
    def analyze_image(self, image_path, prompt, system_prompt="", enable_thinking=False, stream=True):
        """
        Analyze a single image with the given prompt.
        
        Args:
            image_path: Path to image file
            prompt: User prompt for analysis
            system_prompt: System prompt (optional)
            enable_thinking: Enable thinking mode
            stream: Stream response
        
        Returns:
            str: Analysis result
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare messages
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            msgs = [{'role': 'user', 'content': [image, full_prompt]}]
            
            # Generate response
            answer = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                enable_thinking=enable_thinking,
                stream=stream
            )
            
            if stream:
                generated_text = ""
                for new_text in answer:
                    generated_text += new_text
                return generated_text
            else:
                return answer
                
        except Exception as e:
            print(f"Error analyzing image: {e}")
            raise
    
    def analyze_video(self, video_path, prompt, system_prompt="", fps=None, force_packing=None, enable_thinking=False):
        """
        Analyze a video with the given prompt.
        
        Args:
            video_path: Path to video file
            prompt: User prompt for analysis
            system_prompt: System prompt (optional)
            fps: Frames per second for sampling
            force_packing: Force specific packing number
            enable_thinking: Enable thinking mode
        
        Returns:
            tuple: (analysis_result, metadata)
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Encode video
            frames, frame_ts_id_group, metadata = self.encode_video(video_path, fps, force_packing)
            
            # Prepare messages
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            msgs = [{'role': 'user', 'content': frames + [full_prompt]}]
            
            # Generate response
            answer = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                use_image_id=False,
                max_slice_nums=1,
                temporal_ids=frame_ts_id_group,
                enable_thinking=enable_thinking
            )
            
            return answer, metadata
            
        except Exception as e:
            print(f"Error analyzing video: {e}")
            raise
    
    def get_model_info(self):
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "description": self.model_config["description"],
            "memory_efficient": self.model_config["memory_efficient"],
            "device": self.device,
            "is_loaded": self.is_loaded,
            "video_config": {
                "max_num_frames": self.max_num_frames,
                "max_num_packing": self.max_num_packing,
                "default_fps": self.default_fps
            }
        }
    
    @classmethod
    def get_supported_models(cls):
        """Get list of supported models with descriptions."""
        return cls.SUPPORTED_MODELS