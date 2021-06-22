import os
import sys
import json
import math
import shutil
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import lvizgen
import lkeypoints
import lglobalvars
import lprimitives

def main():
    parser = argparse.ArgumentParser()

    # input/output paths
    parser.add_argument('-d', '--directory', default='./data/output/test/',
            help='Directory containing frames and poses')
    parser.add_argument('-o', '--output-dir', default='./visualization/static/data/13Sept_debug',
            help='Output directory name')

    # keypoint loading parameters (configure pose detectors/body type)
    parser.add_argument('-k', '--keypoint-type', type=str, default='posewarp',
            help='Keypoint body format to use (default is 14 posewarp keypoints)')
    parser.add_argument('-m', '--mat-file', help='Use .mat file to precomputed poses (for posewarp data)', action='store_true')
    parser.add_argument('-a', '--alphapose', help='Use Alphapose poses (if neither -m or -a, then use OpenPose)', action='store_true')
    parser.add_argument('-c', '--normalize', help='Normalize the videos w.r.t skeleton', action='store_true')
    
    # segmentation parameters
    parser.add_argument('-s', '--single-primitive', action='store_true',
            help='Fit one primitive per keypoint instead of a sequence of primitives')
    parser.add_argument('-r', '--reg', type=float, default=1600,
            help='Regularization term to be used in DP (use -1 to infer automatically)')
    parser.add_argument('-p', '--points-pp', type=int, default=5,
            help='Search only for primitive lengths in multiples of points_pp (reduces search space)')
    parser.add_argument('-w', '--window', type=int, default=20,
            help='Window around which primitive is searched (i.e. max prim len = points_pp * window) (prefers even number)')
    
    # primitive fitting parameters
    parser.add_argument('--stat-thres', type=float, default=30,
            help='Points lying inside square of this edge length will be marked stationary')
    parser.add_argument('--span-thres', type=float, default=56.56,
            help='If primitive start/end points are closer than this, then label it stationary')
    parser.add_argument('--r-penalty', help='Add radius penalty for large/small circles', action='store_true')
    
    # for synthesizing primitives in video subsequence
    parser.add_argument('--first-n', type=int, default=0,
            help='If not 0, then use only the first n frames')
    parser.add_argument('--start-f', type=int, default=0,
            help='If not both start_f and end_f not 0, then use [start_f:end_f] frames')
    parser.add_argument('--end-f', type=int, default=0,
            help='If not both start_f and end_f not 0, then use [start_f:end_f] frames')

    # misc parameters
    parser.add_argument('--cores', type=int, default=1, help='# of cores used to parallelize DP precomputation')
    parser.add_argument('--no-acc', help='Do not use acceleration term for primitives', action='store_true')
    parser.add_argument('-x', '--no-videos', help='Do not synthesize visualization videos', action='store_true')
    
    global ARGS
    ARGS = parser.parse_args()
    lglobalvars.ARGS = ARGS
    print("=" * 50)

    # set input paths
    data_path = os.path.abspath(ARGS.directory)
    print(f"Input video: {data_path}")
    poses_dir = os.path.join(data_path, "poses/")
    frames_dir = os.path.join(data_path, "frames/")

    # set output paths 
    lglobalvars.filename = ARGS.directory.rstrip("/").split("/")[-1]
    output_dir = os.path.abspath(os.path.join(ARGS.output_dir, lglobalvars.filename)) + "/"

    if not os.path.exists(output_dir):
        temp_path = Path(output_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        if not ARGS.no_videos:
            os.system("cp -r " + frames_dir + " " + output_dir)
            os.system("cp " + data_path + "/video.* " + output_dir)
            # convert avi to mp4
            if os.path.isfile(output_dir + "/video.avi"): 
                os.system("ffmpeg -i " + output_dir + "/video.avi " + output_dir + "/video.mp4")
                os.system("rm " + output_dir + "/video.avi")
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    # save the video resolution as a global variable 
    im = Image.open(frames_dir + next(os.walk(frames_dir))[2][0])
    lglobalvars.W, lglobalvars.H = im.size
    
    # load the keypoints from pose detector
    pose_detector = None
    if ARGS.alphapose:
        pose_detector = "alphapose"
    if ARGS.mat_file:
        pose_detector = "mat_file"
    pose_detector = "openpose" if pose_detector is None else pose_detector
    keypoints = lkeypoints.load(data_path, pose_detector, pose_type=ARGS.keypoint_type)

    # fix missing / badly detected keypoints
    keypoints = lkeypoints.fix_failed(keypoints)

    # use only a subsequence of video, if specified
    if ARGS.first_n != 0:
        keypoints, _ = lkeypoints.prune(keypoints, first_n=ARGS.first_n)

    if ARGS.start_f != 0 or ARGS.end_f != 0:
        keypoints = lkeypoints.keypoints_between(keypoints, ARGS.start_f, ARGS.end_f)

    _, _, files = next(os.walk(frames_dir))
    frames = sorted(files)

    # center & normalize the keypoints wrt body size
    if ARGS.normalize:
        lglobalvars.center = [(keypoints[9][0][0] + keypoints[12][0][0]) // 2, (keypoints[9][0][1] + keypoints[12][0][1]) // 2]
        lglobalvars.scale = lglobalvars.scale_factor / lkeypoints.get_scale(keypoints)
        print(f"Re-center video at: {lglobalvars.center}")
        print(f"Scale video by: {lglobalvars.scale}")
        unscaled_keypoints = keypoints
        keypoints = lkeypoints.normalize(keypoints)

    print("=" * 50)

    #############################################################################
    # Main synthesis                                                            # 
    #############################################################################
    base_dir = output_dir + "base"

    if not os.path.exists(base_dir):
        print("Starting the synthesis module...")
        type_fit = "single_primitive" if ARGS.single_primitive else "dp_all"

        synt_args = {"points_pp": ARGS.points_pp, "stat_thres": ARGS.stat_thres, \
            "span_thres": ARGS.span_thres, "r_penalty": ARGS.r_penalty, \
            "no_acc": ARGS.no_acc, "REG": ARGS.reg, 'cores': ARGS.cores, \
            "window": ARGS.window}
        lglobalvars.synt_args = synt_args

        all_prim = lkeypoints.generate_all_primitives(keypoints, type_fit, synt_args)
        base_prim = all_prim
        os.mkdir(base_dir)

        new_keypoints, color_codes = lkeypoints.trace_keypoints(all_prim)

        print("Generating keypoint visualization & programs...")
        lvizgen.gen_trace_viz_prog(keypoints, new_keypoints, all_prim, base_dir, time_start=0)

        def dummy_func(keypoints):
            if lglobalvars.ARGS.normalize:
                keypoints_un = lkeypoints.unnormalize(keypoints)
                return keypoints_un
            return keypoints

        json_dump = {'filename': ARGS.directory[4:], 'synt_args': synt_args, \
                    'frames': list(range(len(keypoints[0]))), \
                    'input_keypoints': keypoints, 'primitives': all_prim, \
                    'ref_frames': list(range(len(keypoints[0]))), 'generated_keypoints': new_keypoints, \
                    'input_keypoints_unscaled': dummy_func(keypoints), \
                    'generated_keypoints_unscaled': dummy_func(new_keypoints)}

        with open(output_dir + "base.json", 'w') as outfile:
            json.dump(json_dump, outfile)
        
        print("Generating detected pose visualization...")
        lvizgen.gen_pose(keypoints, base_dir, pose_type=ARGS.keypoint_type, cores=ARGS.cores)

        print("Generating predicted pose visualization...")
        lvizgen.gen_pred(new_keypoints, color_codes, base_dir, pose_type=ARGS.keypoint_type, cores=ARGS.cores)

        if not ARGS.no_videos:
            print("Generating video + gt pose visualization...")
            lvizgen.gen_video_pose(keypoints, frames_dir, base_dir)

            print("Generating overlay of detected and predicted pose visualization...")
            lvizgen.gen_overlay(keypoints, new_keypoints, color_codes, base_dir)

    print("Completed.")

if __name__== "__main__":
    main()
