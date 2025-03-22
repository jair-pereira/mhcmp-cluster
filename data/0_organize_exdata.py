import argparse
import glob
import os
import shutil
import time

def organize_exdata(path):
    # base directories
    os.rename(path, f"{path}_un") # rename "exdata" -> "exdata_un"
    os.mkdir(path) # make new "exdata"

    # get all algorithm experimental data directories
    dir_list = glob.glob(f"{path}_un/*")
    dir_list = [os.path.basename(d) for d in dir_list]

    # get base algorithm, param names
    alg_names = set([name.split("-")[0] for name in dir_list])

    # make directories inside the new "exdata"
    for name in alg_names:
        os.mkdir(f"{path}/{name}")

    # move data from exdata_un to exdata/{alg}
    for dir_name in dir_list:
        # move
        src = f"{path}_un/{dir_name}"
        dst = f"{path}/{dir_name.split('-')[0]}/"
        shutil.move(src, dst)

        # rename
        old = f"{dst}{dir_name}"
        new = f"{dst}"
        try:
            new += dir_name.split("-")[1]
        except IndexError:
            new += "000"

        os.rename(old, new)
        
    # remove exdata_un
    dir_deleted = False
    for _ in range(5):
        try:
            os.rmdir(f"{path}_un")
            dir_deleted = True
        except OSError as e:
            if e.errno==66:
                time.sleep(5)
    if not dir_deleted:
        print(f"Couldn't delete {path}_un.")
        print(e)
        print("Check if this directory is empty and manually delete it.")

if __name__ == "__main__":
    ## arguments ##
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', dest='path' , type=str, help="path to the exdata folder (e.g: 'results/20230731/exdata')")
    args = parser.parse_args()

    if args.path.split("/")[-1]=="exdata":
        organize_exdata(path=args.path)
        print("done")
    else:
        print("did nothing - not a exdata directory") # just to prevend messing up wrong directories accidentaly

