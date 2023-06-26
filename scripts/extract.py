import sys
import os
import re
from os import walk
import os.path as osp
import argparse
import random
import json

import cv2
import math
from tqdm import tqdm
from multiprocessing import Pool

# TODO: add additional supported video formats
supported_video_ext = ('.avi', '.mp4')

# TODO: add additional supported extracted frame formats
supported_frame_ext = ('.jpg', '.png')


class FrameExtractor:
    def __init__(self, video_file, output_dir, frame_number,frame_ext='.jpg', sampling=-1):
        """Extract frames from video file and save them under a given output directory.

        Args:
            video_file (str)  : input video filename
            output_dir (str)  : output directory where video frames will be extracted
            frame_ext (str)   : extracted frame file format
            sampling (int)    : sampling rate -- extract one frame every given number of seconds.
                                Default=-1 for extracting all available frames
        """
        # Check if given video file exists -- abort otherwise
        if osp.exists(video_file):
            self.video_file = video_file
        else:
            raise FileExistsError('Video file {} does not exist.'.format(video_file))

        self.sampling = sampling

        # Create output directory for storing extracted frames
        self.output_dir = output_dir
        if not osp.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Get extracted frame file format
        self.frame_ext = frame_ext
        if frame_ext not in supported_frame_ext:
            raise ValueError("Not supported frame file format: {}".format(frame_ext))
        else:
            self.frame_ext = frame_ext

        # Capture given video stream
        self.video = cv2.VideoCapture(self.video_file)

        # Get video fps
        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)

        # Get video length in frames
        self.video_length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sampling != -1:
            self.video_length = self.video_length // self.sampling
        
        #set frame
        self.frame=frame_number
        #self.frame=random.randint(0,self.video_length)
        if self.frame>=self.video_length:
            raise ValueError("frame number is bigger then the video length")
        
        
        self.frame_name=self.video_file
        self.frame_name=self.frame_name.removesuffix('.mp4')
        self.frame_name=self.frame_name.split('/')
        self.frame_name=self.frame_name[-1]
        self.frame_name=self.frame_name.removesuffix('/')
        self.frame_name=self.frame_name=self.frame_name+"_frame"+str(self.frame)+frame_ext


    def extract(self):
        self.video.set(1, self.frame)
        success, frame = self.video.read()
        curr_frame_filename = osp.join(self.output_dir, self.frame_name)
        cv2.imwrite(curr_frame_filename, frame)


def check_sampling_param(s):
    s_ = float(s)
    if (s_ <= 0) and (s_ != -1):
        raise argparse.ArgumentTypeError("Please give a positive number of seconds or -1 for extracting all frames.")
    return s_

class Startup:
    def __init__(self, args, video_list):
        self.args=args
        self.video_list=video_list
    
    def run_map(self):
        with Pool(processes=self.args.workers) as p:
            with tqdm(total=len(self.video_list)) as pbar:
                for i, _ in enumerate(p.imap_unordered(self.extract_video_frames, self.video_list)):
                    pbar.update()
    
    def extract_video_frames(self, video):
        extractor = FrameExtractor(video_file=video,
                                        output_dir=self.args.output_root,
                                        frame_number=self.args.frame_number,
                                        sampling=self.args.sampling)

        extractor.extract()


def main():
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("Extract frames from videos")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--video', type=str, help='set video filename')
    group.add_argument('--dir', type=str, help='set videos directory')
    parser.add_argument('--sampling', type=check_sampling_param, default=-1,
                        help="extract 1 frame every args.sampling seconds (default: extract all frames)")
    parser.add_argument('--output-root', type=str, default='extracted_frames', help="set output root directory")
    parser.add_argument('--workers', type=int, default=None, help="Set number of multiprocessing workers")
    parser.add_argument('--frame_number', type=int, default=0, help="Set number of the frame to extract")
    args = parser.parse_args()

    # Extract frames from a (single) given video file
    if args.video:
        # Setup video extractor for given video file
        video_basename = osp.basename(args.video).split('.')[0]
        # Check video file extension
        video_ext = osp.splitext(args.video)[-1]
        if video_ext not in supported_video_ext:
            raise ValueError("Not supported video file format: {}".format(video_ext))
        # Set extracted frames output directory
        output_dir = osp.join(args.output_root, '{}_frames'.format(video_basename))
        # Set up video extractor for given video file
        extractor = FrameExtractor(video_file=args.video, output_dir=output_dir, sampling=args.sampling)
        # Extract frames
        extractor.extract()

    # Extract frames from all video files found under the given directory (including all sub-directories)
    if args.dir:
        print("#. Extract frames from videos under dir : {}".format(args.dir))
        print("#. Store extracted frames under         : {}".format(args.output_root))
        if args.sampling == -1:
            print("#. Extract all available frames.")
        else:
            print("#. Extract one frame every {} seconds.".format(args.sampling))
        print("#. Scan for video files...")

        # Scan given dir for video files
        video_list = []
        for r, d, f in walk(args.dir):

            
            for file in f:
                file_path=os.path.join(r, file)
                video_list.append(file_path)
                

        # Extract frames from found video files
        global args_
        args_ = args

        start=Startup(args, video_list)
        start.run_map()
        

if __name__ == '__main__':
    main()
