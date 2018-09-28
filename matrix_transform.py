'''
matrix transformation -- 


It defines classes_and_methods

@author:     Jerome

@copyright:  2018 All rights reserved.

@license:    license

@contact:    email
@deffield    
'''
import readline
import sys
import os
import numpy as np
import pandas as pd
from sys import argv


def run():
    # argument reading
    # index of starting task

    # ped file
    ped_file = args.ped_file
    #ped_file = "/ysm-gpfs/home/zy92/project/adversarial_autoencoders/ukb_data/UK_mini_v1.ped"
    
    # map file
    map_file = args.map_file
    #map_file = "/ysm-gpfs/home/zy92/project/adversarial_autoencoders/ukb_data/plink.frq"
    
    
    # output file 
    output_file = args.output_file
    #output_file = "/ysm-gpfs/home/zy92/project/adversarial_autoencoders/ukb_data/output_matrix.txt"
    
    # read table
    #df_ped = pd.read_table(ped_file, sep='\s+', header = None)
    
    
    # read the mapfile
    # map_dict = {}
    df_map = pd.read_table(map_file, sep='\s+', header = 0, low_memory = False)
    ref_series = df_map['A1']
    # open the outputfile
    fo = open(output_file, "w")
        
        
    # read the pedfile
    ct = 0
    with open(ped_file, "r") as fi_ped:
        for line in fi_ped:
            ct += 1
            print("INFO: line" + str(ct))
            tmp_line = line.strip().split()
            tmp_vec = []
            # number of snps
            n = int((len(tmp_line) - 6)/2)
            snp_dict = dict(zip(range(2*n), tmp_line[6:]))
            for j in range(n):
                pivot = 2*j
                #tmp_snp = [tmp_line[pivot], tmp_line[pivot + 1]]
                # snp == A1 => 1
                ref_snp = ref_series[j]
                #tmp_snp_t = [int(tmp_snp[0] == ref_snp), int(tmp_snp[1] == ref_snp)] 
                tmp_snp_v = str(int(snp_dict[pivot] == ref_snp) + int(snp_dict[pivot + 1] == ref_snp)) 
                #tmp_snp_v = str(0)
                #tmp_vec.append(str(sum(tmp_snp_t)))
                tmp_vec.append(tmp_snp_v)
            output_vec = tmp_line[:6] + tmp_vec
            # write to the output
            fo.write("\t".join(output_vec) + "\n")
    
    fo.close() 
    print("INFO: transform complete!" )
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Matrix Transformation')

    parser.add_argument("--verbosity",
                        help="Log verbosity level. 1 is everything being logged. 10 is only high level messages, above 10 will hardly log anything",
                        default = "10")

    parser.add_argument("-p",
                        "--ped_file",
                        help="path to .ped file",
                        default="./ukb_data/UK_mini_v1.ped")

    parser.add_argument("-m",
                        "--map_file",
                        help="path to .map file",
                        default="./ukb_data/UK.map")
    
    parser.add_argument("-o",
                        "--output_file",
                        help="path to output file",
                        default="./output_matrix.txt")

   

    args = parser.parse_args()
    
    # run the main function
    run()
