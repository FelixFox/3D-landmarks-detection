import click
import os
from utils.key_points_recognizer import KeyPointsRecognizer

@click.group()
def cli():
    pass

@cli.command()
@click.option("--video_path", "-v")
@click.option("--dest_path", "-d")
def video2plys(video_path, dest_path):
    recognizer = KeyPointsRecognizer(
        checkpoint_fp='./models/phase1_wpdc_vdc_v2.pth.tar', frames_step=1)

    if not os.path.exists(video_path):
        print("{} file not found".format(video_path))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    recognizer.get_key_points_from_video(video_fp=video_path,
                                         on_frame_processed=None,
                                         save_format=os.path.join(dest_path, "dsg_{}.ply"))


if __name__ == "__main__":
    cli()
