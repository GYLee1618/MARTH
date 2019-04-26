import os

import time
import datetime
import cv2
import pytesseract
import numpy as np
import uuid
import json

import functools
import logging
import collections

from imutils.video import VideoStream
import imutils
import pafy

from multiprocessing import Process,Manager
import multiprocessing

from PIL import Image

import websockets
import asyncio
import os
import ffmpeg
import sys
import subprocess
import io

sys.path.insert(0,'../ssd1351_driver/lib')
from client import ScreenClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_predictor(checkpoint_path):
    logger.info('loading model')
    import tensorflow as tf
    import model
    from icdar import restore_rectangle
    import lanms
    from eval import resize_image, sort_poly, detect

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)

    def predictor(img):
        """
        :return: {
            'text_lines': [
                {
                    'score': ,
                    'x0': ,
                    'y0': ,
                    'x1': ,
                    ...
                    'y3': ,
                }
            ],
            'rtparams': {  # runtime parameters
                'image_size': ,
                'working_size': ,
            },
            'timing': {
                'net': ,
                'restore': ,
                'nms': ,
                'cpuinfo': ,
                'meminfo': ,
                'uptime': ,
            }
        }
        """
        start_time = time.time()
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        start = time.time()
        score, geometry = sess.run(
            [f_score, f_geometry],
            feed_dict={input_images: [im_resized[:,:,::-1]]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        if boxes is not None:
            scores = boxes[:,8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('[timing] {}'.format(duration))

        text_lines = []
        if boxes is not None:
            text_lines = []
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten())))
                tl['score'] = float(score)
                text_lines.append(tl)
        ret = text_lines
        return ret


    return predictor

def dict_to_array(t):
    d = np.float32([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                    t['y2'], t['x3'], t['y3']])
    d = d.reshape(-1, 2)
    return d

# resizes arbitrary quadrilaterals around their center
def resize(res,scale):
    zero = np.average(res,axis=0)
    res_centered = res-zero
    return res_centered*scale+zero

def get_sub_img(img,res):   
    width = np.amax(res[:,0]) - np.amin(res[:,0])
    height = np.amax(res[:,1]) - np.amin(res[:,1])
    dst = np.float32([(0,0),(width,0),(width,height),(0,height)])
    # import pdb
    # pdb.set_trace()
    M = cv2.getPerspectiveTransform(res,dst)
    out = cv2.warpPerspective(img, M, (width,height))
    return out

def image_to_string(procnum,res,config,return_dict):
    text = pytesseract.image_to_string(res['img'], config=config)
    return_dict[procnum] = {'box':res['coords'],'text':text}


async def video_thread(uri,fifo):
    print("Starting video thread")
    with open(fifo,'wb') as f: 
        async with websockets.connect(uri) as websocket:
            # msg = await websocket.recv()
            while (True):
                msg = bytearray(await websocket.recv())
                f.write(msg)

def sync_video_thread(uri,fifo):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(video_thread(uri,fifo))

def decoder_thread(fifo,fifo1):
    print("Starting decoder thread")
    subprocess.Popen("ffmpeg -hide_banner -i "+fifo+" -y -vf fps=.2 -updatefirst 1 pics/tmp.jpg",shell=True)

client = ScreenClient('199.98.27.185',1351)

ws_thread = multiprocessing.Process(target=sync_video_thread,args=("ws://199.98.27.185:8084","/tmp/fifo"))
dc_thread = multiprocessing.Process(target=decoder_thread,args=('/tmp/fifo','/tmp/fifo1'))

predictor = get_predictor('/home/gavin/Downloads/east_icdar2015_resnet_v1_50_rbox')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',fourcc, 0.2, (320,320)) 
manager = Manager()
OCR_res = manager.dict()
# start the FPS throughput estimator
# fps = FPS().start()

ws_thread.start()
dc_thread.start()

client.clearScreen()

# start = datetime.datetime.now()
# Image.fromarray(image.astype('uint8'),'RGB').show()
while (True):
    
    client.setColor('WHITE')
    start = datetime.datetime.now()
    print('====FRAME====')
    frame = cv2.imread('pics/tmp.jpg')
    # import pdb

    # pdb.set_trace()
    # frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        print('no frame')
        time.sleep(5)
        continue

    W = frame.shape[0]
    H = frame.shape[1]

    image = np.rot90(frame.copy()[int(W/2-160):int(W/2+160),int(H/2-160):int(H/2+160),:],1,(0,1))

    results = predictor(image)
    new_res = list()
    STR = datetime.datetime.now()
    for res in results:
        res = resize(dict_to_array(res),1.2)
        tmp = get_sub_img(image,res)
        new_res.append({'coords':res, 'img':tmp})
        # Image.fromarray(image.astype('uint8'),'RGB').show()
        # Image.fromarray(tmp.astype('uint8'),'RGB').show()
    proc = datetime.datetime.now()
    config = ("-l eng --oem 1 --psm 6")
    # OCR_res = list()
    processes = list()
    for i,res in enumerate(new_res):
        processes.append(Process(target=image_to_string,args=(i,res,config,OCR_res)))
        processes[i].start()

    for p in processes:
        p.join()

    for i in range(len(new_res)):
        if OCR_res[i]['text']:
            cv2.putText(image, OCR_res[i]['text'], (int(OCR_res[i]['box'][0,0]), int(OCR_res[i]['box'][0,1] - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            cv2.polylines(image, [OCR_res[i]['box'].astype(np.int32)], isClosed=True, color=(255, 255, 0))
            
            xmax = max(0,min(127,int(np.amax(OCR_res[i]['box'][:,0])*128/320)))
            ymax = max(0,min(127,int(np.amax(OCR_res[i]['box'][:,1])*128/320)))
            xmin = max(0,min(127,int(np.amin(OCR_res[i]['box'][:,0])*128/320)))
            ymin = max(0,min(127,int(np.amin(OCR_res[i]['box'][:,1])*128/320)))

            client.drawText(xmin, ymax + 1,OCR_res[i]['text'])
            w = xmax-xmin
            h = ymax-ymin
            client.drawRect(xmin,ymin,w,h)
            print(xmin,ymin,w,h)
            # import pdb
            # pdb.set_trace()

            print("{}\n-------------".format(OCR_res[i]['text']))


    # end = datetime.datetime.now()

    # print("Frame Time:\t{}\nSTR Time:\t{}\nProc Time:\t{}\nOCR Time\t{}\n".format(end-start,STR-start,proc-STR,end-proc))
    out.write(image)
    end = datetime.datetime.now()
    time.sleep(max(0,5-(end-start).microseconds/1000000))

    client.setColor('BLACK')

    for i in range(len(new_res)):
        if OCR_res[i]['text']:
            # client.drawText(int(OCR_res[i]['box'][0,0]/2), int(OCR_res[i]['box'][0,1]/2 - 1),OCR_res[i]['text'])
            xmax = max(0,min(127,int(np.amax(OCR_res[i]['box'][:,0])*128/320)))
            ymax = max(0,min(127,int(np.amax(OCR_res[i]['box'][:,1])*128/320)))
            xmin = max(0,min(127,int(np.amin(OCR_res[i]['box'][:,0])*128/320)))
            ymin = max(0,min(127,int(np.amin(OCR_res[i]['box'][:,1])*128/320)))
            w = xmax-xmin
            h = ymax-ymin
            client.drawText(xmin, ymax + 1,OCR_res[i]['text'])
            client.drawRect(xmin,ymin,w,h)

            print("{}\n-------------".format(OCR_res[i]['text']))
        # cv2.imshow('video',image)

        # import pdb
        # pdb.set_trace()
sys.exit()
