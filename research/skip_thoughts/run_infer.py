#!/usr/bin/env python3
#--- coding: utf-8 ---

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

import argparse

def main():

    parser = argparse.ArgumentParser(
        description="encoding sentences example for skip_thoughts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('vocab_file', help="specify the vocab_file")
    parser.add_argument('embedding_matrix_file', help='specify the embedding_matrix_file')
    parser.add_argument('checkpoint_path', help="specify the checkpoint_path")
    parser.add_argument('mr_data_dir', help="specify the mr_data_dir")
    parser.add_argument('--model_name', default="skip_thoughts")
    parser.add_argument('--bidirect', choices=["True", "False"],
                        default="False")

    args = parser.parse_args()

    if args.bidirect == "True":
        args.bidirect=True
    else:
        args.bidirect=False

    encoder = encoder_manager.EncoderManager(args.model_name)
    encoder.load_model(configuration.model_config(bidirectional_encoder=args.bidirect),
                       vocabulary_file=args.vocab_file,
                       embedding_matrix_file=args.embedding_matrix_file,
                       checkpoint_path=args.checkpoint_path)

    data = []
    with open(os.path.join(args.mr_data_dir, 'rt-polarity.neg'), 'rb') as f:
        data.extend([line.decode('latin-1').strip() for line in f])
    with open(os.path.join(args.mr_data_dir, 'rt-polarity.pos'), 'rb') as f:
        data.extend([line.decode('latin-1').strip() for line in f])


    encodings = encoder.encode(data)

    def get_nn(ind, num=10):
        encoding = encodings[ind]
        scores = sd.cdist([encoding], encodings, 'cosine')[0]
        sorted_ids = np.argsort(scores)
        print("Senetence:")
        print("", data[ind])
        print("\nNearest neighbors:")
        for i in range(1, num + 1):
            print(" %d. %s (%.3f)" %
                  (i, data[sorted_ids[i]], scores[sorted_ids[i]]))

    get_nn(0)

if __name__ == '__main__':
    main()
