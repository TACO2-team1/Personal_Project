import PIL.Image as Image
import numpy as np
import os
import utils
import time

anno_bbox_filename = r"./CelebA/Anno/list_bbox_celeba.txt"
anno_landmark_filename = r"./CelebA/Anno/list_landmarks_celeba.txt"
img_dir = r"./CelebA/Img/img_celeba.7z/img_celeba"

crop_random_seeds = [0.1,0.2,0.95,0.95,0.95,0.95,0.99,0.99,0.99,0.99]

def generate_data(face_size, begin_, end_, save_path):
    print("generate size: {}*{} image" .format(face_size,face_size))
    print("save_path: {}".format(save_path))

    positive_image_dir = os.path.join(save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")
    landmark_image_dir = os.path.join(save_path, str(face_size), "landmark")
    image_dirs = [positive_image_dir, negative_image_dir, part_image_dir, landmark_image_dir]

    for dir_path in image_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")
    landmark_anno_filename = os.path.join(save_path, str(face_size), "landmark.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0
    landmark_count = 0

    positive_anno_file = open(positive_anno_filename, "w")
    negative_anno_file = open(negative_anno_filename, "w")
    part_anno_file = open(part_anno_filename, "w")
    landmark_anno_file = open(landmark_anno_filename, "w")

    raw_bbox_anno_file = open(anno_bbox_filename)
    raw_landmark_anno_file = open(anno_landmark_filename)

    raw_examples_count=int(raw_bbox_anno_file.readline())
    raw_bbox_anno_file.readline()
    raw_landmark_anno_file.readline()
    raw_landmark_anno_file.readline()

    begin_time=time.time()
    for _ in range(raw_examples_count):
        line_bbox=raw_bbox_anno_file.readline()
        line_landmark=raw_landmark_anno_file.readline()
        
        if _<begin_ or _>=end_:
            continue

        if _%100==0:
            print("image: [{}/{} ({:2f}%)]\texamples: {}\ttime: {:.2f}s\t".format(_-begin_,
                end_-begin_,100.0*(_-begin_)/(end_-begin_),
                positive_count+negative_count+part_count+landmark_count,time.time()-begin_time))
        strs=line_bbox.split()
        strs_landmark=line_landmark.split()
        # print(strs_landmark)

        image_filename = strs[0].strip()
        image_filename = os.path.join(img_dir, image_filename)
        # print(image_filename)
        with Image.open(image_filename) as img:
            img_w, img_h = img.size
            x1 = float(strs[1].strip())
            y1 = float(strs[2].strip())
            w = float(strs[3].strip())
            h = float(strs[4].strip())
            x2 = float(x1 + w)
            y2 = float(y1 + h)

            if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                continue

            px1 = float(strs_landmark[1].strip())
            py1 = float(strs_landmark[2].strip())
            px2 = float(strs_landmark[3].strip())
            py2 = float(strs_landmark[4].strip())
            px3 = float(strs_landmark[5].strip())
            py3 = float(strs_landmark[6].strip())
            px4 = float(strs_landmark[7].strip())
            py4 = float(strs_landmark[8].strip())
            px5 = float(strs_landmark[9].strip())
            py5 = float(strs_landmark[10].strip())


            
            boxes = [[x1, y1, x2, y2]]

            cx = x1+w/2
            cy = y1+h/2
            side_len = max(w, h)
            seed = crop_random_seeds[np.random.randint(0, len(crop_random_seeds))]

            for __ in range(5):
                _side_len = side_len + np.random.randint(int(-side_len * seed),int(side_len * seed))
                _cx = cx + np.random.randint(int(-cx * seed), int(cx * seed))
                _cy = cy + np.random.randint(int(-cy * seed), int(cy * seed))

                _x1 = _cx - _side_len / 2
                _y1 = _cy - _side_len / 2
                _x2 = _x1 + _side_len
                _y2 = _y1 + _side_len

                if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:
                    continue

                offset_x1 = (x1 - _x1) / _side_len
                offset_y1 = (y1 - _y1) / _side_len
                offset_x2 = (x2 - _x2) / _side_len
                offset_y2 = (y2 - _y2) / _side_len

                offset_px1 = (px1 - _x1) / _side_len
                offset_py1 = (py1 - _y1) / _side_len
                offset_px2 = (px2 - _x1) / _side_len
                offset_py2 = (py2 - _y1) / _side_len
                offset_px3 = (px3 - _x1) / _side_len
                offset_py3 = (py3 - _y1) / _side_len
                offset_px4 = (px4 - _x1) / _side_len
                offset_py4 = (py4 - _y1) / _side_len
                offset_px5 = (px5 - _x1) / _side_len
                offset_py5 = (py5 - _y1) / _side_len

                offset_ps=[offset_px1,offset_py1,offset_px2,offset_py2,offset_px3,
                    offset_py3,offset_px4,offset_py4,offset_px5,offset_py5]

                crop_box = [_x1, _y1, _x2, _y2]
                face_crop = img.crop(crop_box)
                face_resize = face_crop.resize((face_size, face_size))

                iou = utils.iou(crop_box, np.array(boxes))[0]
                if iou > 0.66:  # positive
                    positive_anno_file.write(
                        "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            positive_count, 1, offset_x1, offset_y1,
                            offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                            offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                    positive_anno_file.flush()
                    face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                    positive_count += 1
                elif 0.58 > iou > 0.35:  # part
                    part_anno_file.write(
                        "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            part_count, 2, offset_x1, offset_y1,offset_x2,
                            offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                            offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                    part_anno_file.flush()
                    face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                    part_count += 1
                elif iou < 0.1: # negative
                    negative_anno_file.write(
                        "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                    negative_anno_file.flush()
                    face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                    negative_count += 1
                
                is_landmark_example=True
                for offset_p in offset_ps:
                    if offset_p>0.9 or offset_p<0.1:
                        is_landmark_example=False
                if is_landmark_example:
                    landmark_anno_file.write(
                        "landmark/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                            landmark_count, 3, offset_x1, offset_y1,offset_x2,
                            offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                            offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                    landmark_anno_file.flush()
                    face_resize.save(os.path.join(landmark_image_dir, "{0}.jpg".format(landmark_count)))
                    landmark_count += 1


if __name__=='__main__':
    generate_data(12,0,172599,r"./Dataset/Train")
    generate_data(12,172599,202600,r"./Dataset/Test")

    generate_data(24,0,172599,r"./Dataset/Train")
    generate_data(24,172599,202600,r"./Dataset/Test")

    generate_data(48,0,172599,r"./Dataset/Train")
    generate_data(48,172599,202600,r"./Dataset/Test")