from flask import request
from flask import Flask, url_for
from flask import render_template
from flask_paginate import Pagination, get_page_args

import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

import ast
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def homepage():
    folders = [name for name in os.listdir('static/data/') \
                    if os.path.isdir(os.path.join('static/data/', name))]
    headers = ["Models", "Model names", ""]
    body = sorted(folders) 
    return render_template("/list.html", headers=headers, body=body)

@app.route('/<model>')
def browse_model(model):
    folders =  [name for name in os.listdir('static/data/' + model) \
                    if os.path.isdir(os.path.join('static/data/' + model, name))]
    headers = ["Data elements ", "Video name", model]
    body = sorted(folders) 
    return render_template("/list.html", headers=headers, body=body)

@app.route('/<model>/<video>')
def folder_landing(model, video, folder="base"):
    pose_vid =  "data/" + model + "/" + video + "/" + folder + "/v_pose.mp4"
    if folder.startswith("base"):
        input_vid = "data/" + model + "/" + video + "/video.mp4"

    input_vid = url_for('static', filename=input_vid)
    pose_vid = url_for('static', filename=pose_vid)

    def _get_url(vidname):
        vid =  "data/" + model + "/" + video + "/" + folder + f"/{vidname}.mp4"
        vid = url_for('static', filename=vid)
        return vid

    p = {"dataname": video, "folder": folder, "input_vid": input_vid, "pose_vid": pose_vid, \
        "pred_vid": _get_url('v_pred')}


    mapping = ["head (0)", "neck (1)", "r-shoulder (2)", "r-elbow (3)", "r-wrist (4)", \
        "l-shoulder (5)", "l-elbow (6)", "l-wrist (7)", "r-hip (9)", \
        "r-knee (10)", "r-ankle (11)", "l-hip (12)", "l-knee (13)", "l-ankle (14)"]
    keypoints = []
    
    posewarp_joints = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
    for i in posewarp_joints:
        keypoint_dir = "data/" + model + "/" + video + "/" + folder + "/keypoint" + str(i)
        prog_file = open("static/" + keypoint_dir + "/program.txt", "r")
        img_url = keypoint_dir + "/trace_prediction.png"
        program = prog_file.read()
        program = "<br/>".join(program.split("\n"))
        keypoints.append((mapping[i - 1 if i > 8 else i], program, url_for('static', filename=img_url)))
        prog_file.close()
    p["keypoints"] = keypoints
    return render_template("/example.html", p=p)
