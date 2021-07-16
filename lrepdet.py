import os
import cv2
import sys
import json
import copy
import tqdm
import shutil
import random
import pickle
import imageio
import numpy as np
from pathlib import Path
from scipy.stats import norm

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lkeypoints
import lglobalvars
import lvizgen
from lstm_models.reparam import deserialize_prim_J

# Module for detecting repetitions/for-loop in motion programs

PRIM_GEN_EXP_NAME = str(sys.argv[1])
video_name = str(sys.argv[2])
filter_ = int(sys.argv[3])

guass_thres = 8
max_loop_size = 10
loop_size_a1 = max_loop_size + 1
window_sizes = {1: 20, 2: 10, 3: 12, \
                4: 12, 5: 15, 6: 12, \
                7: 14, 8: 16, 9: 18, 10: 20}

pose_type = 'coco'
if pose_type == 'coco':
    consider_kp = lglobalvars.coco_joints
elif pose_type == 'aist':
    consider_kp = lglobalvars.aist_joints
else:
    consider_kp = lglobalvars.posewarp_joints

data_dir = Path(f'./visualization/static/data/{PRIM_GEN_EXP_NAME}')
video_segments = sorted(os.listdir(data_dir))
if filter_ == 1:
    res_dir = Path(f'./visualization/static/data/{PRIM_GEN_EXP_NAME}_{video_name}_detectloops_maxres_REPRO')
else:
    res_dir = Path(f'./visualization/static/data/{PRIM_GEN_EXP_NAME}_{video_name}_detectloops_maxres_nofilter_REPRO')
res_dir.mkdir(parents=True, exist_ok=True)

# sw_dir = Path(f'./visualization/static/data/{PRIM_GEN_EXP_NAME}_{video_name}_detectloops_maxres')

def load_pose_taichi(pose_dir):
    alphapose_results = Path(pose_dir) / 'alphapose-results.json' 
    with open(alphapose_results) as f:
        info = json.load(f)


    vid_idx_info = {}
    for i, x in enumerate(info):
        if x['idx'] in vid_idx_info:
            vid_idx_info[x['idx']] += x['score']  
        else:
            vid_idx_info[x['idx']] = x['score']  

    max_idx_info = list(vid_idx_info.keys())[0]
    for elem in vid_idx_info:
        if vid_idx_info[elem] > vid_idx_info[max_idx_info]:
            max_idx_info = elem

    info_filter = {}
    pose_map = []
    for i, x in enumerate(info):
        curr_vid_name = x['image_id'].rsplit('_', 2)[0]
        if x['idx'] == max_idx_info:
            info_filter[x['image_id']] = x
            pose_map.append(int(x['image_id'].split('.')[0]))

    return pose_map

def load_pose_cardio(pose_dir):
    pose_path = Path(pose_dir) / 'poses'
    pose_files = sorted(os.listdir(pose_path), key=(lambda x: int(x.split('.')[0])))
    pose_map = []
    for elem in pose_files:
        pose_map.append(int(elem.split('.')[0]))
    return pose_map

total_count = 0 
for video in tqdm.tqdm(video_segments):
    if video != video_name:
        continue
    
    print("==================================================")
    print("Loading primitives...")
    curr_file = open(data_dir / video / f'base.json', 'r')
    keypoint_data = json.load(curr_file)
    
    frames = keypoint_data['frames']
    filename = keypoint_data['filename']
    synt_args = keypoint_data['synt_args']
    input_keypoints = keypoint_data['input_keypoints']
    generated_keypoints = keypoint_data['generated_keypoints']
    all_prim = keypoint_data['primitives']
    curr_file.close()

    curr_file = open(data_dir / video / 'base' / 'keypoint0' / '0.json', 'r')
    keypoint_data = json.load(curr_file)
    center = keypoint_data['center']
    scale = keypoint_data['scale']
    lglobalvars.center, lglobalvars.scale = center, scale
    lglobalvars.H = keypoint_data['H']
    lglobalvars.W = keypoint_data['W']
    curr_file.close()
    print("Done!")
    print("==================================================")

    values = {'times': []}
    for idx in range(1, loop_size_a1):
        values[idx] = []

    if (res_dir / f'{video}_times.txt').is_file():
        with open(res_dir / f'{video}_times.txt', 'r') as f:
            temp = f.readlines()
            for elem in range(len(temp)):
                temp[elem] = int(temp[elem].rstrip())
            values['times'] = temp
        
        for i in range(1, loop_size_a1):
            with open(res_dir / f'{video}_numbody{i}.txt', 'r') as f:
                temp = f.readlines()
                for elem in range(len(temp)):
                    temp[elem] = float(temp[elem].rstrip())
                values[i] = temp
    else:
        values['times'] = [0]
        for curr_iter, prim_id in enumerate(all_prim[str(0)]):
            values['times'].append(values['times'][-1] + all_prim[str(0)][prim_id][-2])

        with open(res_dir / f'{video}_times.txt', 'w') as f:
            for item in values['times']:
                f.write("%s\n" % item)

        for num_body in range(1, loop_size_a1):
            total_prim_num = len(all_prim[str(0)])
            last_iter = num_body - 1

            ## initialize empty list for storing points
            points = {}
            for slide_num in range(max(0, total_prim_num - window_sizes[num_body])):
                points[slide_num] = {}
                for kp in consider_kp:
                    points[slide_num][kp] = {}
                    for curr_iter in range(num_body):
                        points[slide_num][kp][f'{curr_iter:02d}'] = []

            ## store the start/mid/end points for windows, times and ptypes too
            times = {}
            ptypes = {}
            for slide_num in tqdm.tqdm(range(max(0, total_prim_num - window_sizes[num_body]))):
                times[slide_num] = {}
                ptypes[slide_num] = {}
                for kp in consider_kp:
                    ptypes[slide_num][kp] = {}
                    for curr_iter, prim_id in enumerate(all_prim[str(kp)]):
                        if curr_iter < slide_num or curr_iter >= slide_num + window_sizes[num_body]:
                            continue
                        shifted_iter = curr_iter - slide_num
                        temp_line = shifted_iter % num_body
                        prim_time = all_prim[str(kp)][prim_id][-2]
                        prim_type = all_prim[str(kp)][prim_id][-1]
                        
                        # store the start/mid/end-point
                        gt_kp = lkeypoints.trace_prim(all_prim[str(kp)][prim_id], prim_time) 
                        points[slide_num][kp][f'{temp_line:02d}'].append([gt_kp[0][0], gt_kp[0][1], gt_kp[(len(gt_kp) - 1) // 2][0], 
                            gt_kp[(len(gt_kp) - 1) // 2][1], gt_kp[len(gt_kp) - 1][0], gt_kp[len(gt_kp) - 1][1]])

                        # store the primitive types
                        if temp_line not in ptypes[slide_num][kp]:
                            ptypes[slide_num][kp][temp_line] = [prim_type]
                        else:
                            ptypes[slide_num][kp][temp_line].append(prim_type)

                        # store the times of primitive
                        if kp == 7:
                            if temp_line not in times[slide_num]:
                                times[slide_num][temp_line] = [prim_time]
                            else:
                                times[slide_num][temp_line].append(prim_time)

            ## compute the params for these points
            params = {}
            for slide_num in range(max(0, total_prim_num - window_sizes[num_body])):
                params[slide_num] = {}
                for kp in consider_kp:
                    params[slide_num][kp] = {}
                    for key in points[slide_num][kp]:
                        curr_mean = np.mean(np.asarray(points[slide_num][kp][key]), axis=0)
                        curr_std = 0.1 * np.cov(np.asarray(points[slide_num][kp][key]).T)
                        if len(points[slide_num][kp][key]) == 1:
                            curr_std = np.zeros_like(curr_std)
                        params[slide_num][kp][key] = (curr_mean, curr_std)

            ## dump these params
            with open(res_dir / f'{video}_params{num_body}.pkl', 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

            ## compute matrix norm on covariance matrices
            values[num_body] = []
            for slide_num in range(max(0, total_prim_num - window_sizes[num_body])):
                temp_val = 0
                for kp in consider_kp:
                    curr_kp_val = 0
                    for key in params[slide_num][kp]:
                        curr_kp_val += np.linalg.norm(params[slide_num][kp][key][1])
                    curr_kp_val /= len(params[slide_num][kp])
                    temp_val = max(temp_val, curr_kp_val)
                temp_val /= (len(consider_kp) * len(params[slide_num][0]))
                values[num_body].append(temp_val)

            ## dump these values too
            with open(res_dir / f'{video}_numbody{num_body}.txt', 'w') as f:
                for item in values[num_body]:
                    f.write("%s\n" % item)

        fig = plt.figure()
        ax = plt.axes()
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'lightcoral', 'olive', 'magenta', 'hotpink']
        for num_body in range(1, loop_size_a1):
            temp_plot_values = copy.deepcopy(values[num_body])
            for elem in range(len(temp_plot_values)):
                temp_plot_values[elem] = guass_thres + 5 if temp_plot_values[elem] > guass_thres + 5 else temp_plot_values[elem]
            plt.plot(values['times'][:len(temp_plot_values)], temp_plot_values, color=colors[num_body - 1], label=f'loop {num_body}')
        plt.title(f"cov vs. frames for {video_name} (exp {PRIM_GEN_EXP_NAME})")
        plt.xlabel("# frames")
        plt.ylabel("covariance")
        plt.legend()
        plt.savefig(res_dir / 'graph.png')

    ## create idx to frame num mapping
    pose_path = Path(f'/viscam/u/sumith/motion2prog/data/ytcardio/{video_name}')
    if 'cardio' in video_name:
        pose_map = load_pose_cardio(pose_path)
    elif 'taichi' in video_name:
        pose_map = load_pose_taichi(pose_path)

    ## create loop detection windows
    loop_start_end = {}
    for i in range(1, loop_size_a1):
        loop_start_end[i] = []
        for elem in range(len(values[i])):
            if values[i][elem] < guass_thres:
                if len(loop_start_end[i]) != 0 and loop_start_end[i][-1][1] == elem - 1 + window_sizes[i]:
                    loop_start_end[i][-1][1] = elem + window_sizes[i]
                else:
                    loop_start_end[i].append([elem, elem + window_sizes[i]])

    if filter_ == 1:
        for outer_idx in range(2, loop_size_a1):
            exist_set = set()
            for inner_idx in range(1, outer_idx):
                for elem in loop_start_end[inner_idx]:
                    curr_set = set(range(elem[0], elem[1] + 1))
                    exist_set = exist_set.union(curr_set)
            new_loop_start_end = []
            for elem in loop_start_end[outer_idx]:
                curr_set = set(range(elem[0], elem[1] + 1))
                curr_set = curr_set.difference(exist_set)
                curr_set = sorted(list(curr_set))
                start_end = []
                for idx, prim in enumerate(curr_set):
                    if len(start_end) == 1:
                        if idx != len(curr_set) - 1 and prim + 1 == curr_set[idx + 1]:
                            continue
                        else:
                            start_end.append(prim)
                            if start_end[1] - start_end[0] > 2 * outer_idx:
                                new_loop_start_end.append(copy.deepcopy(start_end))
                            start_end = []
                    else:
                        start_end.append(prim)
            loop_start_end[outer_idx] = new_loop_start_end

    ## remove extra prims to match loop body size
    for curr_idx in range(1, loop_size_a1):
        for inner_idx in range(len(loop_start_end[curr_idx])):
            loop_start_end[curr_idx][inner_idx][1] -= \
                (loop_start_end[curr_idx][inner_idx][1] - \
                    loop_start_end[curr_idx][inner_idx][0]) % curr_idx

    def synt_execute_forloop(all_prim, num_body, base_dir=None, iters=10, cov_f=0.1, loop_name=None):
        base_dir_str = str(base_dir)
        points = {}
        for kp in consider_kp:
            points[kp] = {}
            for curr_iter in range(num_body):
                points[kp][f'{curr_iter:02d}'] = []

        times, ptypes = {}, {}
        start_angles, end_angles = {}, {}
        for kp in consider_kp:
            ptypes[kp] = {}
            for curr_iter, prim_id in enumerate(all_prim[kp]):
                if curr_iter == 0 or curr_iter == len(all_prim[kp]) - 1:
                    continue
                temp_line = curr_iter % num_body
                prim_time = all_prim[kp][prim_id][-2]
                prim_type = all_prim[kp][prim_id][-1]

                gt_kp = lkeypoints.trace_prim(all_prim[kp][prim_id], prim_time) 
                points[kp][f'{temp_line:02d}'].append([gt_kp[0][0], gt_kp[0][1], gt_kp[(len(gt_kp) - 1) // 2][0], 
                    gt_kp[(len(gt_kp) - 1) // 2][1], gt_kp[len(gt_kp) - 1][0], gt_kp[len(gt_kp) - 1][1]])

                if temp_line not in ptypes[kp]:
                    ptypes[kp][temp_line] = [prim_type]
                else:
                    ptypes[kp][temp_line].append(prim_type)

                if kp == 0:
                    if temp_line not in times:
                        times[temp_line] = [prim_time]
                    else:
                        times[temp_line].append(prim_time)

        params = {}
        for kp in consider_kp:
            params[kp] = {}
            for key in points[kp]:
                curr_mean = np.mean(np.asarray(points[kp][key]), axis=0)
                curr_std = cov_f * np.cov(np.asarray(points[kp][key]).T)
                if len(points[kp][key]) == 1:
                    curr_std = np.zeros_like(curr_std)
                params[kp][key] = (curr_mean, curr_std)

        with open(loop_name, 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

        loop_kp = {}
        color_codes = []
        end_angles_plt_data = {}
        for num_iter in range(iters):
            temp_line = num_iter % num_body
            prim_time = random.sample(times[temp_line], 1)[0]        
            color_codes += [num_iter % 2] * prim_time
            for kp in consider_kp:
                if num_iter == 0:
                    loop_kp[kp] = []
            
                temp_line = num_iter % num_body
                start_x, start_y, mid_x, mid_y, end_x, end_y = \
                    np.random.multivariate_normal(params[kp][f'{temp_line:02d}'][0], params[kp][f'{temp_line:02d}'][1], 1)[0].T

                point_params = [[start_x, start_y], [mid_x, mid_y], [end_x, end_y]]
                prim_type = random.sample(ptypes[kp][temp_line], 1)[0]
                sampled_prim = deserialize_prim_J([start_x, start_y, mid_x, mid_y, end_x, end_y], prim_time, prim_type, dtype='threepoint')
                sampled_kp = lkeypoints.trace_prim(sampled_prim, prim_time)

                loop_kp[kp] += sampled_kp

        return loop_kp, color_codes

    ## detetected loop boundaries

    print("[info] detected loop boundaries!")
    print(loop_start_end)

    from forloop_labels import labels
    curr_labels = labels[video_name]
    num_labels = len(curr_labels) // 3
    wrong_prec = 0
    recall_table = [False] * num_labels
    num_detect = 0

    LOOP_ITERS = 0
    for i in range(2, loop_size_a1):
        for idx, elem in enumerate(loop_start_end[i]):
            left_idx_start = values['times'][elem[0]]
            right_idx_start = values['times'][elem[1] + 1]
            left_frame_start = pose_map[left_idx_start]
            right_frame_start = pose_map[right_idx_start]
            curr_frames = set(range(left_frame_start, right_frame_start))
            best_match_idx = 0
            curr_max_int = 0
            for match_idx in range(num_labels):
                match_frames = set(range(int(curr_labels[3*match_idx]), int(curr_labels[3*match_idx + 1])))
                if len(match_frames.intersection(curr_frames)) > curr_max_int:
                    curr_max_int = len(match_frames.intersection(curr_frames))
                    best_match_idx = match_idx                   
            loop_name = f'loop{i}_{idx}'
            match_left_frame = curr_labels[3*best_match_idx]
            match_right_frame = curr_labels[3*best_match_idx + 1]
            match_desc = curr_labels[3*best_match_idx + 2]
            best_match_frames = set(range(int(match_left_frame), int(match_right_frame)))
            loop_quality = len(best_match_frames.intersection(curr_frames)) / len(best_match_frames.union(curr_frames))
            loop_quality_mean = len(best_match_frames.intersection(curr_frames)) / min(len(best_match_frames), len(curr_frames))
                        
            if right_frame_start < max_frame_consider + 200:
                curr_all_prim = {}
                for kp in consider_kp:
                    curr_all_prim[kp] = {}
                    for prim_idx in range(elem[0], elem[1] + 1):
                        curr_all_prim[kp][prim_idx] = all_prim[str(kp)][str(prim_idx)]
                synt_execute_forloop(curr_all_prim, i,  iters=30, loop_name=f'{video_name}_{LOOP_ITERS}_{match_desc}_{loop_quality}')
                LOOP_ITERS += 1

            if loop_quality >= 0.5:
                recall_table[best_match_idx] = True
                print(f'IoU\t{loop_quality:.3f}\tIoM\t{loop_quality_mean:.3f}')
                print(f'{loop_name}\t{left_frame_start}\t{right_frame_start}\t{best_match_idx}\t{match_left_frame}\t{match_right_frame}\t{match_desc}')
                num_detect += 1
            else:
                print(f'bad prec: {loop_name}')
                wrong_prec += 1
    
    for idx, elem in enumerate(recall_table):
        if not elem:
            match_left_frame = curr_labels[3*idx]
            match_right_frame = curr_labels[3*idx + 1]
            match_desc = curr_labels[3*idx + 2]
            print(f'bad recall {match_left_frame} {match_right_frame} {match_desc}')

    print(f"total labels: {num_labels}")
    print(f"total detect: {num_detect}")
    recall_x = len([x for x in recall_table if x]) 
    recall = recall_x * 100 / num_labels
    precision = num_detect * 100 / (num_detect + wrong_prec)
    print(f'precision {precision}')
    print(f'labels {recall_x}')
    print(f'recall {recall}')

    res2_dir = Path(f'./visualization/static/data/{PRIM_GEN_EXP_NAME}_{video_name}_detectloops_maxres_regulardemo')

    for i in range(2, loop_size_a1):
        for idx, elem in enumerate(loop_start_end[i]):
            sample_dir = res2_dir / f'loop{i}_{idx}'
            frames_dir = res2_dir / f'loop{i}_{idx}' / 'frames'
            base_dir = res2_dir / f'loop{i}_{idx}' / 'base'
            frames_dir.mkdir(parents=True, exist_ok=True)
            base_dir.mkdir(parents=True, exist_ok=True)

            left_idx_start = values['times'][elem[0]]
            right_idx_start = values['times'][elem[1] + 1]
            if pose_type == 'aist':
                left_frame_start = left_idx_start
                right_frame_start = right_idx_start
            else:
                left_frame_start = pose_map[left_idx_start]
                right_frame_start = pose_map[right_idx_start]

            curr_inp_kp = {}
            curr_gen_kp = {}
            curr_all_prim = {}
            for kp in consider_kp:
                curr_inp_kp[kp] = input_keypoints[str(kp)][left_idx_start:right_idx_start]
                curr_gen_kp[kp] = generated_keypoints[str(kp)][left_idx_start:right_idx_start]
                curr_all_prim[kp] = {}
                for prim_idx in range(elem[0], elem[1] + 1):
                    curr_all_prim[kp][prim_idx] = all_prim[str(kp)][str(prim_idx)]

            _, color_codes = lkeypoints.trace_keypoints(curr_all_prim)
            
            curr_loop_kp, loop_color_codes = synt_execute_forloop(curr_all_prim, i,  base_dir=base_dir, iters=30)

            for frame_idx in range(left_frame_start, right_frame_start):
                shutil.copy2(f'./data/ytcardio/{video_name}/frames/{frame_idx:05d}.jpg', frames_dir)
            os.system(f"ffmpeg -hide_banner -r 10  -f image2 -i '{frames_dir}/%*.jpg' \
                            -vcodec libx264 -crf 25 -pix_fmt yuv420p {sample_dir}/video.mp4")

            lvizgen.gen_trace_viz_prog(curr_inp_kp, curr_gen_kp, curr_all_prim, base_dir, time_start=0, forloop=True)
            lvizgen.gen_pose(curr_inp_kp, base_dir, forloop=True, pose_type=pose_type)
            lvizgen.gen_pred(curr_gen_kp, color_codes, base_dir, forloop=True, pose_type=pose_type)

            lvizgen.gen_pred(curr_loop_kp, loop_color_codes, base_dir, filename=f'v_loop', forloop=True, pose_type=pose_type)

            json_dump = {'filename': video_name, 'synt_args': synt_args, \
                    'frames_dir': str(frames_dir), \
                    'output_dir': str(base_dir), \
                    'frames': list(range(right_frame_start - left_frame_start)), \
                    'ref_frames': list(range(right_frame_start - left_frame_start)), \
                    'primitives': curr_all_prim, \
                    'input_keypoints': curr_inp_kp, \
                    'generated_keypoints': curr_gen_kp, \
                    f'loop_keypoints': curr_loop_kp, \
                    'input_keypoints_unscaled': lkeypoints.unnormalize(curr_inp_kp), \
                    'generated_keypoints_unscaled': lkeypoints.unnormalize(curr_gen_kp), \
                    'loop_keypoints_unscaled': lkeypoints.unnormalize(curr_loop_kp)
                    }

            with open(sample_dir / f"base.json", 'w') as outfile:
                json.dump(json_dump, outfile)

