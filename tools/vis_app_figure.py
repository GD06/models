#!/usr/bin/env python3
#--- coding: utf-8 ---

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

class AppPlot:
    def __init__(self, name='app_type'):
        self.name = name
        self.x_list = []
        self.y_list = []
        return

    def assign_attr(self, app_attr):
        self.marker = app_attr['marker']
        self.color = app_attr['color']
        if 'area' in app_attr:
            self.area = app_attr['area']
        else:
            self.area = np.pi * (4 ** 2)
        return

    def assign_set(self, app_set):
        self.app_set = app_set
        return


def draw_app_domain(app_dict, output_dir, output_fig):

    cv_app = AppPlot('CV')
    cv_app.assign_attr({'marker': 'o', 'color': 'r'})
    cv_app.assign_set({'delf_extract_features', 'faster_rcnn_inception_resnet_v2_atrous_coco',
                       'faster_rcnn_inception_v2_coco', 'faster_rcnn_nas_coco',
                       'faster_rcnn_resnet101_coco', 'faster_rcnn_resnet50_coco',
                       'rfcn_resnet101_coco', 'ssd_inception_v2_coco',
                       'ssd_mobilenet_v1_coco', 'real_nvp', 'slim_inception_resnet_v2',
                       'slim_inception_v4', 'slim_mobilenet_v1', 'slim_nasnet_large',
                       'slim_nasnet_mobile', 'slim_resnet_v2_101', 'slim_resnet_v2_152',
                       'slim_resnet_v2_50', 'slim_vgg_16', 'slim_vgg_19', 'tcn_inception_v3',
                       'tcn_resnet_v2', 'video_prediction_cdna', 'video_prediction_dna',
                       'video_prediction_stp', 'ptn_rotator', 'ptn'})

    nlp_app = AppPlot('NLP')
    nlp_app.assign_attr({'marker': '^', 'color': 'b'})
    nlp_app.assign_set({'lm_1b', 'namignize_small', 'namignize_large', 'swivel', 'textsum'})

    hybrid_app = AppPlot('CV + NLP')
    hybrid_app.assign_attr({'marker': 's', 'color': 'c'})
    hybrid_app.assign_set({'attention_ocr', 'im2txt_feed_image', 'im2txt_infer_step'})

    info_app = AppPlot('Information and Coding')
    info_app.assign_attr({'marker': '+', 'color': 'g'})
    info_app.assign_set({'adversarial_crypto_alice', 'adversarial_crypto_bob',
                         'adversarial_crypto_eve', 'adversarial_text', 'audioset',
                         'entropy_coder', 'image_encoder', 'image_decoder',
                         'skipt_thoughts_bi', 'skipt_thoughts_uni'})

    other_app = AppPlot('Others')
    other_app.assign_attr({'marker': '*', 'color': 'm'})
    other_app.assign_set({'cmp.lmap_Msc.clip5.sbpd_d_r2r', 'differential_privacy_sgd',
                          'gan_cifar_cond', 'gan_cifar_uncond', 'gan_mnist',
                          'learning_to_remember_rare_events',
                          'lfads_chaotic_rnn_inputs_g2p5', 'lfads_chaotic_rnn_multisession',
                          'lfads_itb_rnn', 'lfads_chaotic_rnns_labeled',
                          'pcl_rl_urex', 'pcl_rl_reinforce'})

    app_classes = [cv_app, nlp_app, hybrid_app, info_app, other_app]

    for app_name in app_dict.keys():
        found = False
        for each_app_class in app_classes:
            if app_name in each_app_class.app_set:
                found = True
                each_app_class.x_list.append(app_dict[app_name][1])
                each_app_class.y_list.append(app_dict[app_name][2])
        if not found:
            print('App name: {}'.format(app_name))
            raise NotImplementedError

    plt.figure()
    for each_app_class in app_classes:
        plt.scatter(each_app_class.x_list, each_app_class.y_list, s=each_app_class.area,
                    c=each_app_class.color, marker=each_app_class.marker,
                    label=each_app_class.name, alpha=0.8)

    plt.xlabel('$R_2$')
    plt.ylabel('$R_3$')
    plt.legend(loc='best')

    plt.savefig(os.path.join(output_dir, output_fig), format='png')
    return

def draw_app_fathom(app_dict, output_dir, output_fig):

    fathom_app = AppPlot('Fathom')
    fathom_app.assign_attr({'marker': 's', 'color': 'b'})
    fathom_app.assign_set({'alexnet', 'autoenc', 'deepq', 'memnet',
                           'residual', 'seq2seq', 'speech', 'vgg'})

    model_zoo_app = AppPlot('TF Model Zoo')
    model_zoo_app.assign_attr({'marker': 'o', 'color': 'r'})
    model_zoo_app.assign_set({})

    app_classes = [model_zoo_app, fathom_app]

    for app_name in app_dict.keys():
        found = False
        for each_app_class in app_classes:
            if app_name in each_app_class.app_set:
                found = True
                each_app_class.x_list.append(app_dict[app_name][1])
                each_app_class.y_list.append(app_dict[app_name][2])
        if not found:
            app_classes[0].x_list.append(app_dict[app_name][1])
            app_classes[0].y_list.append(app_dict[app_name][2])

    plt.figure()
    for each_app_class in app_classes:
        plt.scatter(each_app_class.x_list, each_app_class.y_list, s=each_app_class.area,
                    c=each_app_class.color, marker=each_app_class.marker,
                    label=each_app_class.name, alpha=0.8)

    plt.xlabel('$R_2$')
    plt.ylabel('$R_3$')
    plt.legend(loc='best')

    plt.savefig(os.path.join(output_dir, output_fig), format='png')

    return

def main():

    parser = argparse.ArgumentParser(
        description="draw application feature distribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help="the directory under which "
                        "profiling results are saved.")
    parser.add_argument('--suffix', default=".pickle", help="the suffix of result files.")
    parser.add_argument('--output_dir', default=None, help="the output directory "
                        "to store output results.")
    parser.add_argument('--output_fig', default="app_feature.png")
    parser.add_argument('--mode', default="domain", choices=['domain', 'fathom'])

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getenv('LOG_OUTPUT_DIR'), 'outputs')

    all_app_dict = {}
    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        for input_file in filenames:
            if args.suffix in input_file:

                with open(os.path.join(dirpath, input_file), 'rb') as f:
                    app_dict, result_dict = pickle.load(f)

                for app_key in app_dict.keys():
                    all_app_dict[app_key] = app_dict[app_key]

    if args.mode == 'domain':
        draw_app_domain(all_app_dict, args.output_dir, args.output_fig)
    elif args.mode == 'fathom':
        draw_app_fathom(all_app_dict, args.output_dir, args.output_fig)

if __name__ == '__main__':
    main()
