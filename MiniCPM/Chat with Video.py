## The 3d-resampler compresses multiple frames into 64 tokens by introducing temporal_ids. 
# To achieve this, you need to organize your video data into two corresponding sequences: 
#   frames: List[Image]
#   temporal_ids: List[List[Int]].

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord
from scipy.spatial import cKDTree
import numpy as np
import math

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True,  # or openbmb/MiniCPM-o-2_6
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True)  # or openbmb/MiniCPM-o-2_6

MAX_NUM_FRAMES=180 # Indicates the maximum number of frames received after the videos are packed. The actual maximum number of valid frames is MAX_NUM_FRAMES * MAX_NUM_PACKING.
MAX_NUM_PACKING=3  # indicates the maximum packing number of video frames. valid range: 1-6
TIME_SCALE = 0.1 

def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]

def encode_video(video_path, choose_fps=3, force_packing=None):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps
        
    if choose_fps * int(video_duration) <= MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, video_duration))
        
    else:
        packing_nums = math.ceil(video_duration * choose_fps / MAX_NUM_FRAMES)
        if packing_nums <= MAX_NUM_PACKING:
            choose_frames = round(video_duration * choose_fps)
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing_nums = MAX_NUM_PACKING

    frame_idx = [i for i in range(0, len(vr))]      
    frame_idx =  np.array(uniform_sample(frame_idx, choose_frames))

    if force_packing:
        packing_nums = min(force_packing, MAX_NUM_PACKING)
    
    print(video_path, ' duration:', video_duration)
    print(f'get video frames={len(frame_idx)}, packing_nums={packing_nums}')
    
    frames = vr.get_batch(frame_idx).asnumpy()

    frame_idx_ts = frame_idx / fps
    scale = np.arange(0, video_duration, TIME_SCALE)

    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    assert len(frames) == len(frame_ts_id)

    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
    frame_ts_id_group = group_array(frame_ts_id, packing_nums)
    
    return frames, frame_ts_id_group


video_path="video_test.mp4"
fps = 5 # fps for video
force_packing = None # You can set force_packing to ensure that 3D packing is forcibly enabled; otherwise, encode_video will dynamically set the packing quantity based on the duration.
frames, frame_ts_id_group = encode_video(video_path, fps, force_packing=force_packing)

question = "Describe the video"
msgs = [
    {'role': 'user', 'content': frames + [question]}, 
]


answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    use_image_id=False,
    max_slice_nums=1,
    temporal_ids=frame_ts_id_group
)
print(answer)
