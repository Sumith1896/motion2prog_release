import os
import json
import tqdm

import cv2
import numpy as np

import lkeypoints
import lglobalvars

def tuple_int(x): return (int(x[0]), int(x[1]))


#############################################################################
# Visualize predicted pose sequences                                        # 
#############################################################################
def wrapper_dump_pred(i, keypoints, pose_type, color_code):
    for idx in range(len(keypoints[0])):
        img = np.full((lglobalvars.H, lglobalvars.W, 3), 255, dtype=np.uint8)
        colors = [lglobalvars.GREEN, lglobalvars.DPINK]
        if pose_type == 'default' or pose_type == 'mpii':
            curr_limbs = lglobalvars.posewarp_limbs
        else:
            curr_limbs = lglobalvars.coco_limbs

        for j in curr_limbs:
            cv2.line(img, tuple_int(keypoints[j[0]][idx]), tuple_int(keypoints[j[1]][idx]), colors[color_code[idx]], 3)
        cv2.imwrite("0" * (5 - len(str(i*300 + idx))) + str(i*300 + idx) + '.png', img)

def gen_pred(new_keypoints, color_code, output_dir, filename='v_pred', forloop=False, pose_type='default', cores=None):
    if forloop or lglobalvars.ARGS.normalize:
        new_keypoints_un = lkeypoints.unnormalize(new_keypoints)
    curr_dir = os.getcwd()
    os.chdir(output_dir)
    colors = [lglobalvars.GREEN, lglobalvars.DPINK]

    if cores is None or cores == 1:
        for i in tqdm.tqdm(range(len(new_keypoints_un[0]))):
            img = np.full((lglobalvars.H, lglobalvars.W, 3), 255, dtype=np.uint8)
            if pose_type == 'default' or pose_type == 'mpii':
                curr_limbs = lglobalvars.posewarp_limbs
            else:
                curr_limbs = lglobalvars.coco_limbs

            for j in curr_limbs:
                cv2.line(img, new_keypoints_un[j[0]][i], new_keypoints_un[j[1]][i], colors[color_code[i]], 3)
            cv2.imwrite("0" * (5 - len(str(i))) + str(i) + '.png', img)
    else:

        def smaller_kp(i, keypoints):
            new_keypoints = {}
            for joint in keypoints.keys():
                new_keypoints[joint] = keypoints[joint][i*300:(i + 1)*300]
            return new_keypoints

        def smaller_cc(i, color_code):
            return color_code[i*300:(i + 1)*300]
        
        from joblib import Parallel, delayed
        Parallel(n_jobs=cores, backend='multiprocessing')\
                            (delayed(wrapper_dump_pred)(i, smaller_kp(i, new_keypoints_un), pose_type, smaller_cc(i, color_code)) \
                            for i in tqdm.tqdm(range(len(new_keypoints_un[0]) // 300 + 1)))


    os.system(f"ffmpeg -hide_banner -loglevel panic -r 10 -f image2 -pattern_type glob -i '*.png' -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" \
                      -vcodec libx264 -crf 25  -pix_fmt yuv420p {filename}.mp4")
    os.system(f"mkdir {filename}")
    os.system(f"mv *.png {filename}/")
    os.chdir(curr_dir)


#############################################################################
# Visualize input overlayed w/ predicted pose sequences                     # 
#############################################################################
def gen_overlay(keypoints, new_keypoints, color_code, output_dir, kd_metrics=None, \
                    filename='v_overlay', forloop=False, intplt_f=None):
    if forloop or lglobalvars.ARGS.normalize:
        keypoints_un = lkeypoints.unnormalize(keypoints)
        new_keypoints_un = lkeypoints.unnormalize(new_keypoints)
    curr_dir = os.getcwd()
    os.chdir(output_dir)
    colors = [lglobalvars.GREEN, lglobalvars.DPINK]

    for i in range(len(new_keypoints_un[0])):
        img = np.full((lglobalvars.H, lglobalvars.W, 3), 255, dtype=np.uint8)
        for j in lglobalvars.posewarp_limbs:
            cv2.line(img, new_keypoints_un[j[0]][i], new_keypoints_un[j[1]][i], colors[color_code[i]], 3)
            cv2.line(img, tuple_int(keypoints_un[j[0]][i]), tuple_int(keypoints_un[j[1]][i]), lglobalvars.BLACK, 3)
        if intplt_f != None:
            if i % intplt_f == 0:
                # import ipdb; ipdb.set_trace()
                img[:30,-30:] = lglobalvars.GREEN
        if kd_metrics != None:
            text_color = (255,0,0)
            texts = kd_metrics[i]
            for _i,_t in enumerate(texts):
                img = cv2.putText(img, _t, (5,20*(_i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.imwrite("0" * (5 - len(str(i))) + str(i) + '.png', img)

    os.system(f"ffmpeg -hide_banner -loglevel panic -r 10 -f image2 -pattern_type glob -i '*.png' -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" \
                      -vcodec libx264 -crf 25  -pix_fmt yuv420p {filename}.mp4")
    os.system(f"mkdir {filename}")
    os.system(f"mv *.png {filename}/")
    os.chdir(curr_dir)


#############################################################################
# Visualize input pose sequences                                            # 
#############################################################################
def wrapper_dump_pose(i, keypoints, pose_type):
    for idx in range(len(keypoints[0])):
        img = np.full((lglobalvars.H, lglobalvars.W, 3), 255, dtype=np.uint8)
        if pose_type == 'default' or pose_type == 'mpii':
            curr_limbs = lglobalvars.posewarp_limbs
        else:
            curr_limbs = lglobalvars.coco_limbs

        for j in curr_limbs:
            cv2.line(img, tuple_int(keypoints[j[0]][idx]), tuple_int(keypoints[j[1]][idx]), lglobalvars.BLACK, 3)
        cv2.imwrite("0" * (5 - len(str(i*300 + idx))) + str(i*300 + idx) + '.png', img)

def gen_pose(keypoints, output_dir, filename='v_pose', forloop=False, pose_type='default', cores=None):
    if forloop or lglobalvars.ARGS.normalize:
        keypoints_un = lkeypoints.unnormalize(keypoints)
    curr_dir = os.getcwd()
    os.chdir(output_dir)

    if cores is None or cores == 1:
        for i in tqdm.tqdm(range(len(keypoints_un[0]))):
            img = np.full((lglobalvars.H, lglobalvars.W, 3), 255, dtype=np.uint8)
            if pose_type == 'default' or pose_type == 'mpii':
                curr_limbs = lglobalvars.posewarp_limbs
            else:
                curr_limbs = lglobalvars.coco_limbs

            for j in curr_limbs:
                cv2.line(img, tuple_int(keypoints_un[j[0]][i]), tuple_int(keypoints_un[j[1]][i]), lglobalvars.BLACK, 3)
            cv2.imwrite("0" * (5 - len(str(i))) + str(i) + '.png', img)
    else:
        def smaller_kp(i, keypoints):
            new_keypoints = {}
            for joint in keypoints.keys():
                new_keypoints[joint] = keypoints[joint][i*300:(i + 1)*300]
            return new_keypoints

        from joblib import Parallel, delayed
        Parallel(n_jobs=cores, backend='multiprocessing')\
                            (delayed(wrapper_dump_pose)(i, smaller_kp(i, keypoints_un), pose_type) \
                            for i in tqdm.tqdm(range(len(keypoints_un[0]) // 300 + 1)))

    os.system(f"ffmpeg -hide_banner -loglevel panic -r 10 -f image2 -pattern_type glob -i '*.png' -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" \
                      -vcodec libx264 -crf 25  -pix_fmt yuv420p {filename}.mp4")
    os.system(f"mkdir {filename}")
    os.system(f"mv *.png {filename}/")
    os.chdir(curr_dir)


#############################################################################
# Visualize primitive on an image                                           # 
#############################################################################
def draw_prim(img, prim, color):
    if prim[-1] == "LINE":
        cv2.line(img, (int(np.poly1d(prim[2])(0)), int(np.poly1d(prim[3])(0))), \
            (int(np.poly1d(prim[2])(prim[-2] - 1)), int(np.poly1d(prim[3])(prim[-2] - 1))), \
            color, 2)
    elif prim[-1] == "CIRCLE":
        if int(prim[3]) >= 0:
            cv2.circle(img, (int(prim[2][0]), int(prim[2][1])), int(prim[3]), color, 2)
    else:
        cv2.circle(img, (int(prim[0][0]), int(prim[0][1])), 2, color, -1)
    return img


#############################################################################
# Visualize programs and dump individual keypoint programs to txt file      # 
#############################################################################
def gen_trace_viz_prog(keypoints, new_keypoints, all_prim, output_dir, prune_acc=False, time_start=0, \
                        old_prim=None, forloop=False):
    if forloop or lglobalvars.ARGS.normalize:
            new_keypoints_un = lkeypoints.unnormalize(new_keypoints)
            keypoints_un = lkeypoints.unnormalize(keypoints)
    
    curr_dir = os.getcwd()
    os.chdir(output_dir)

    for joint in keypoints_un.keys():
        os.mkdir("keypoint" + str(joint))
        os.chdir("keypoint" + str(joint))

        img = np.full((lglobalvars.H, lglobalvars.W, 3), 255, dtype=np.uint8)
        for i in range(len(keypoints_un[joint])):
            cv2.circle(img, tuple_int(keypoints_un[joint][i]), 4, lglobalvars.GREY, -1)
        for i in range(len(new_keypoints_un[joint])):
            cv2.circle(img, tuple_int(new_keypoints_un[joint][i]), 4, lglobalvars.ORANGE, -1)
        cv2.imwrite('trace_prediction.png', img)

        prog_file = open('program.txt', 'w')
        curr_ptr = 0
        for x in all_prim[joint]:
            elem = all_prim[joint][x]
            if not forloop:
                curr_prim_data = {}
                curr_prim_data['prim'] = elem
                curr_kp = lkeypoints.normalize_list(keypoints_un[joint][curr_ptr:curr_ptr + elem[-2]])
                for i in range(len(curr_kp)):
                    curr_kp[i] = (float(curr_kp[i][0]), float(curr_kp[i][1]))
                curr_prim_data['points'] = curr_kp
                curr_prim_data['frames'] = list(range(len(keypoints_un[joint])))[curr_ptr:curr_ptr + elem[-2]]
                curr_prim_data['filename']= lglobalvars.filename
                curr_prim_data['H']= lglobalvars.H
                curr_prim_data['W']= lglobalvars.W
                curr_prim_data['center']= (float(lglobalvars.center[0]), float(lglobalvars.center[1]))
                curr_prim_data['scale']= float(lglobalvars.scale)
                curr_prim_data['prim_id'] = x
                curr_prim_data['synt_args'] = lglobalvars.synt_args

                curr_ptr += elem[-2]
                
                with open(str(x) + '.json', 'w') as fp:
                    json.dump(curr_prim_data, fp)

            if elem[-1] == "LINE":

                if prune_acc:
                    new_x_eq = lkeypoints.prune_acc_func(elem[2], time_start)
                    new_y_eq = lkeypoints.prune_acc_func(elem[3], time_start)
                    x_eq = f"{new_x_eq[0]:.2f}*t + {new_x_eq[1]:.2f}"
                    y_eq = f"{new_y_eq[0]:.2f}*t + {new_y_eq[1]:.2f}"
                else:
                    x_eq = f"{elem[2][0]:.2f}*t**2 + {elem[2][1]:.2f}*t + {elem[2][2]:.2f}"
                    y_eq = f"{elem[3][0]:.2f}*t**2 + {elem[3][1]:.2f}*t + {elem[3][2]:.2f}"
                prog_file.write(f"linear_motion(x={x_eq}, y={y_eq}, T={elem[4]})\n")

            elif elem[-1] == "CIRCLE":
                c_eq = f"({elem[2][0]:.2f}, {elem[2][1]:.2f})"
                if prune_acc:
                    new_a_eq = lkeypoints.prune_acc_func(elem[4], time_start)
                    ang_eq = f"{new_a_eq[0]:.2f}*t + {new_a_eq[1]:.2f}"
                else:
                    ang_eq = f"{elem[4][0]:.2f}*t**2 + {elem[4][1]:.2f}*t + {elem[4][2]:.2f}"
                prog_file.write(f"circular_motion(c={c_eq}, r={elem[3]:.2f}, angle={ang_eq}, T={elem[5]})\n")

            elif elem[-1] == "LINE_S":
                s_eq = f"({elem[0][0]:.2f}, {elem[0][1]:.2f})"
                e_eq = f"({elem[1][0]:.2f}, {elem[1][1]:.2f})"
                prog_file.write(f"linear_simple(s={s_eq}, e={e_eq}, T={elem[2]})\n")

            else:
                prog_file.write(f"stationary(x={elem[0][0]:.2f}, y={elem[0][1]:.2f}, T={elem[1]})\n")

        prog_file.close()
        os.chdir("../")

    os.chdir(curr_dir)
