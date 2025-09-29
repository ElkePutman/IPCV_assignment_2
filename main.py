import argparse
import sys
import os


from Processor_class import VideoProcessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='input video filename (zonder path)')
    parser.add_argument('-o', "--output", help='output video filename (zonder path)')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide input and output video filenames! See --help")

   
    BASE_INPUT_PATH = os.path.dirname(os.path.abspath(__file__))
    
    BASE_OUTPUT_PATH = os.path.join(BASE_INPUT_PATH,"Processed_videos")
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)

    input_file = os.path.join(BASE_INPUT_PATH, args.input)
    output_file = os.path.join(BASE_OUTPUT_PATH, args.output)
    

    process = VideoProcessor(input_file, output_file,down_fact=1)

    process.run(show_video=False)
    # process.debug_single_frame(19000, show_video=False,save_frame=True)