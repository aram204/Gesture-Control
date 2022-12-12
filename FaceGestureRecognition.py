from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
from djitellopy import Tello
import mediapipe as mp
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
me = Tello()
me.connect()
me.streamoff()
me.streamon()

print(me.get_battery())

modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"
flag = False
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 30  # minimum size of face
        threshold = [0.7,0.8,0.8]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        batch_size =130 #1000
        image_size = 182
        input_image_size = 160
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile,encoding='latin1')

        width = 320
        height = 240


        # resize frame (optional)
        print('Start Recognition')
        t1= time.time()
        count = 0
        count_of_flight = 0
        while True:

            frame_read = me.get_frame_read()
            frame = frame_read.frame
            frame = cv2.resize(frame, (width, height))
            print(type(frame))




            cv2.imshow('Face Recognition', frame)


            timer =time.time()
            result_names = 0
            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            faceNum = bounding_boxes.shape[0]
            if faceNum > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]
                cropped = []
                scaled = []
                scaled_reshape = []
                for i in range(faceNum):
                    emb_array = np.zeros((1, embedding_size))
                    xmin = int(det[i][0])
                    ymin = int(det[i][1])
                    xmax = int(det[i][2])
                    ymax = int(det[i][3])
                    try:
                        # inner exception
                        if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                            print('Face is very close!')
                            continue
                        cropped.append(frame[ymin:ymax, xmin:xmax,:])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                        if best_class_probabilities>0.998:
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                    cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                    cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 0), thickness=1, lineType=1)


                        else :
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                            cv2.putText(frame, "?", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 0), thickness=1, lineType=1)
                    except:

                        print("error")
            out.write(frame)
            cv2.imwrite(f'img{count}.jpg', frame)
            count+=1
            t2 = time.time()
            endtimer = time.time()
            fps = 1/(endtimer-timer)
            cv2.rectangle(frame,(15,30),(135,60),(0,255,255),-1)
            cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow('Face Recognition', frame)
            key= cv2.waitKey(1)
            print(t2-t1)
            if key == 113 or result_names == 'Aram' :

                flag = True
                break
            elif t2 - t1 >= 30 and count_of_flight == 0 :
                me.takeoff()


                me.land()
                print('land')
                count_of_flight+=1


        out.release()
        cv2.destroyAllWindows()




if flag:
    me.takeoff()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils


    def orientation(coordinate_landmark_0, coordinate_landmark_9, shape=(1, 1)):
        x0 = coordinate_landmark_0.x * shape[0]
        y0 = coordinate_landmark_0.y * shape[1]
        x9 = coordinate_landmark_9.x * shape[0]
        y9 = coordinate_landmark_9.y * shape[1]
        if abs(x9 - x0) < 0.05:
            m = 1000000000
        else:
            m = abs((y9 - y0) / (x9 - x0))
        if m >= 0 and m <= 1:
            if x9 > x0:
                return "Right"
            else:
                return "Left"
        if m > 1:
            if y9 < y0:
                return "Up"
            else:
                return "Down"


    def dist(l1, l2):
        return ((((l2[0] - l1[0]) ** 2) + ((l2[1] - l1[1]) ** 2)) ** 0.5)


    def finger(handlandmarks, shape=(1, 1)):
        try:
            needful = [handlandmarks.landmark[0].x * shape[0], handlandmarks.landmark[0].y * shape[1]]
            d06 = dist(needful, [handlandmarks.landmark[6].x * shape[0],
                                 handlandmarks.landmark[6].y * shape[1]])
            d08 = dist(needful, [handlandmarks.landmark[8].x * shape[0],
                                 handlandmarks.landmark[8].y * shape[1]])
            d010 = dist(needful, [handlandmarks.landmark[10].x * shape[0],
                                  handlandmarks.landmark[10].y * shape[1]])
            d012 = dist(needful, [handlandmarks.landmark[12].x * shape[0],
                                  handlandmarks.landmark[12].y * shape[1]])
            d014 = dist(needful, [handlandmarks.landmark[14].x * shape[0],
                                  handlandmarks.landmark[14].y * shape[1]])
            d016 = dist(needful, [handlandmarks.landmark[16].x * shape[0],
                                  handlandmarks.landmark[16].y * shape[1]])
            d018 = dist(needful, [handlandmarks.landmark[18].x * shape[0],
                                  handlandmarks.landmark[18].y * shape[1]])
            d020 = dist(needful, [handlandmarks.landmark[20].x * shape[0],
                                  handlandmarks.landmark[20].y * shape[1]])

            closed = []
            if d06 > d08:
                closed.append(1)
            if d010 > d012:
                closed.append(2)
            if d014 > d016:
                closed.append(3)
            if d018 > d020:
                closed.append(4)
            return closed
        except:
            pass


    ######################################################################
    width = 320  # WIDTH OF THE IMAGE
    height = 240  # HEIGHT OF THE IMAGE
    startCounter = 0  # 0 FOR FIGHT 1 FOR TESTING
    ######################################################################.

    # CONNECT TO TELLO

    me.for_back_velocity = 0
    me.left_right_velocity = 0
    me.up_down_velocity = 0
    me.yaw_velocity = 0
    me.speed = 0

    print(me.get_battery())
    me.streamoff()
    me.streamon()

    with mp_hands.Hands(min_detection_confidence=0.8,
                        min_tracking_confidence=0.8, max_num_hands=1) as hands:
        while True:
            frame_read = me.get_frame_read()
            myFrame = frame_read.frame
            img = cv2.resize(myFrame, (width, height))

            image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            frame_shape = (image.shape[1], image.shape[0])

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    closed = finger(hand_landmarks, frame_shape)
                    orient = orientation(hand_landmarks.landmark[0],
                                         hand_landmarks.landmark[9], frame_shape)
                    if closed == [2, 3]:
                        print("flip")
                        cv2.putText(image, "Flip", (500, 200), 2,
                                        fontScale=1, color=(0, 128, 0), thickness=2)
                        me.flip_back()
                    elif closed == [2, 3, 4]:
                        if orient == "Up":
                            print("up")
                            cv2.putText(image, "Up", (500, 200), 2,
                                        fontScale=1, color=(0, 128, 0), thickness=2)
                            me.move_up(40)
                        elif orient == "Down":
                            print("down")
                            cv2.putText(image, "Down", (500, 200), 2,
                                        fontScale=1, color=(0, 128, 0), thickness=2)
                            me.move_down(40)
                        elif orient == "Right":
                            print("right")
                            cv2.putText(image, "Right", (500, 200), 2,
                                        fontScale=1, color=(0, 128, 0), thickness=2)
                            me.move_left(50)
                        elif orient == "Left":
                            print("left")
                            cv2.putText(image, "Left", (500, 200), 2,
                                        fontScale=1, color=(0, 128, 0), thickness=2)
                            me.move_right(50)
                    elif closed == [1, 2, 3, 4]:
                        if orient == "Right" or orient == "Left":
                            if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y:
                                cv2.putText(image, "Forward", (500, 200), 2,
                                            fontScale=1, color=(0, 128, 0), thickness=2)
                                print("forward")
                                me.move_forward(50)
                            else:
                                cv2.putText(image, "Land", (500, 200), 2,
                                            fontScale=1, color=(0, 128, 0), thickness=2)
                                print("land")
                                me.land()
                                sys.exit()
                        else:
                            cv2.putText(image, "Back", (500, 200), 2,
                                        fontScale=1, color=(0, 128, 0), thickness=2)
                            print("back")
                            me.move_back(50)

            cv2.imshow("MyResult", img)

                    # WAIT FOR THE 'Q' BUTTON TO STOP
            if cv2.waitKey(1) & 0xFF == ord('q'):
                me.land()
                break
