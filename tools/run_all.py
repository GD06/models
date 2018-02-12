#!/usr/bin/env python3
#- coding: utf-8 -

import os
import subprocess
import argparse

def main():

    parser = argparse.ArgumentParser(
        description="run all cg_run.sh under the specified directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('work_dir', help="specify the directory under which "
                        "all models will be executed.")

    args = parser.parse_args()

    work_status = {}

    for dirpath, dirnames, filenames in os.walk(args.work_dir):
        if 'cg_run.sh' in filenames:
            cmd = os.path.join(dirpath, 'cg_run.sh')
            try:
                subprocess.run(cmd, shell=True, cwd=dirpath, check=True)
                work_status[dirpath] = 'Accepted'
            except Exception as excep:
                work_status[dirpath] = 'Failed'

    succ_num = 0
    for wkey in work_status.keys():
        if work_status[wkey] == 'Accepted':
            succ_num += 1

    print('Summary')
    print('Running Total {} Models with {} Accepted'.format(
        len(work_status), succ_num))
    for wkey in work_status.keys():
        print('{}, {}'.format(wkey, work_status[wkey]))


if __name__ == '__main__':
    main()
