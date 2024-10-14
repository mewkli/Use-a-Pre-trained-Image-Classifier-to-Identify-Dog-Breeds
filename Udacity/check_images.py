#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/check_images.py
#
# TODO 0: Add your information below for Programmer & Date Created.                                                                             
# PROGRAMMER: Mehuli Saha                          
# REVISED DATE: 
# PURPOSE: Classifies pet images using a pretrained CNN model, compares these
#          classifications to the true identity of the pets in the images, and
#          summarizes how well the CNN performed on the image classification task. 
#          Note that the true identity of the pet (or object) in the image is 
#          indicated by the filename of the image. Therefore, your program must
#          first extract the pet image label from the filename before
#          classifying the images using the pretrained CNN model. With this 
#          program we will be comparing the performance of 3 different CNN model
#          architectures to determine which provides the 'best' classification.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images_solution.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Imports functions created for this program
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results

# Main program function defined below
def main():
    # Measure total program runtime by collecting start time
    start_time = time()
    in_arg = get_input_args()

    # Placeholder for get_pet_labels() - Replace with actual implementation
    results = get_pet_labels(in_arg.dir)
    classify_images(in_arg.dir, results, in_arg.arch)

    # Placeholder for adjust_results4_isadog() - Replace with actual implementation
    adjust_results4_isadog(results, in_arg.dogfile)
    results_stats = calculates_results_stats(results)

    # Placeholder for print_results() - Replace with actual implementation
    print_results(results, results_stats, in_arg.arch, True, True)
    end_time = time()
    tot_time = end_time - start_time
    print("\nTotal Elapsed Runtime:", tot_time, "seconds.")
    
    hours = int(tot_time // 3600)
    minutes = int((tot_time % 3600) // 60)
    seconds = int((tot_time % 3600) % 60)
    print("\nTotal Elapsed Runtime:", f"{hours:02}:{minutes:02}:{seconds:02}")

# Call to main function to run the program
if __name__ == "__main__":
    main()
