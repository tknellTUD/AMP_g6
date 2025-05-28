import os
from zipfile import ZipFile, ZIP_DEFLATED
from  argparse import ArgumentParser

def zip_res(res_folder, output_path="vod_submit.zip"):
    """
        res_folder: the folder of the output results
    """
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zf:
        for fname in os.listdir(res_folder):
            full_path = os.path.join(res_folder, fname)
            if os.path.isfile(full_path):
                zf.write(full_path, arcname=fname)
                
if __name__ == "__main__":     
    parser = ArgumentParser()
    parser.add_argument(
        "--res_folder", type=str, default="outputs/centerpoint_ro47020/preds", help="the folder of the output results"
    )
    parser.add_argument(
        "--output_path", type=str, default="submission.zip", help="the output zip file"
    )
    args = parser.parse_args()
    zip_res(args.res_folder, args.output_path)