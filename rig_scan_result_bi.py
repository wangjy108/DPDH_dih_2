import pandas as pd
import os
import random
import math
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from decimal import Decimal
from decimal import ROUND_HALF_UP,ROUND_HALF_EVEN
from joblib import Parallel, delayed

def param(ini):
    with open(ini, "r") as f:
        content = [ff.strip() for ff in f.readlines() if len(ff.strip()) > 0]

    qm_opt_engine = [(line.split("="))[-1] for line in content if line.startswith("QM_OPT_ENGINE")][0]
    qm_scan_engine = [(line.split("="))[-1] for line in content if line.startswith("QM_SCAN_ENGINE")][0]
    charge = [int((line.split("="))[-1]) for line in content if line.startswith("CHARGE")][0]
    spin = [int((line.split("="))[-1]) for line in content if line.startswith("SPIN")][0]
    dih_1_string = [(line.split("="))[-1] for line in content if line.startswith("DIH1")][0]
    dih_2_string = [(line.split("="))[-1] for line in content if line.startswith("DIH2")][0]
    terval1 = [int((line.split("="))[-1]) for line in content if line.startswith("TERVAL1")][0]
    terval2 = [int((line.split("="))[-1]) for line in content if line.startswith("TERVAL2")][0]
    #bond = [(line.split("="))[-1] for line in content if line.startswith("BOND")][0]
    #rotable_angle = [int((line.split("="))[-1]) for line in content if line.startswith("ROTATE_ANGlE")][0]
    #rotable_terval = [int((line.split("="))[-1]) for line in content if line.startswith("ROTATE_TERVAL")][0]
    funtional = [(line.split("="))[-1] for line in content if line.startswith("FUNCTIONAL")][0]
    if_d3 = [int((line.split("="))[-1]) for line in content if line.startswith("D3")][0]
    opt_basis = [(line.split("="))[-1] for line in content if line.startswith("OPT_BASIS")][0]
    scan_basis = [(line.split("="))[-1] for line in content if line.startswith("SCAN_BASIS")][0]
    if_chk = [int((line.split("="))[-1]) for line in content if line.startswith("CHK")][0]
    #run_base = [(line.split("="))[-1] for line in content if line.startswith("RUN_BASE")][0]
    #torsion_A = [(line.split("="))[-1] for line in content if line.startswith("TORSION_A")][0]
    #torsion_B = [(line.split("="))[-1] for line in content if line.startswith("TORSION_B")][0]

    return qm_opt_engine, qm_scan_engine,charge, spin, dih_1_string, dih_2_string, terval1, terval2, \
           funtional, if_d3, opt_basis, scan_basis, if_chk


def get_initial_angle(file, atom_list_string):
    atom_list = [f"a{i}" for i in (atom_list_string.strip()).split(",")]
    content = ["2", "-9", " ".join(atom_list), "q", "-10", "q"]
    with open("log2dih", "w+") as cc:
        for line in content:
            cc.write(line + "\n")

    raw_prefix = (file.strip()).split(".")[0]
    os.system(f"Multiwfn {file} < log2dih | grep 'The dihedral angle is' > {raw_prefix}.dih")
    time.sleep(1)

    with open(f"{raw_prefix}.dih", "r") as f:
        line = [ff.strip() for ff in f.readlines()][0]

    dih = Decimal(line.split()[-2]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)

    os.system(f"rm -f {raw_prefix}.dih")
    os.system("rm -f log2dih")
    return int(str(dih))


def result_process(prefix, ini):
    _, _, _, _, dih_1_string, dih_2_string, _, _, _, _, _, _, _ = param(ini)
    #shift angle, minimun, maximum, ignore error, save conformation??
    log_file = [log for log in os.listdir() if ".log" in log and "rigid" in log]
    log_idx = set([int(((name.strip()).split(".")[0]).split("_")[-1]) for name in log_file])
    real_prefix = "_".join((log_file[0].split(".")[0]).split("_")[:-1])

    record = {}

    dih_1 = []
    dih_2 = []
    #dih_1_initial = get_initial_angle(f"{real_prefix}_1.log", dih_1_string)
    #dih_2_initial = get_initial_angle(f"{real_prefix}_1.log", dih_2_string)

    for idx in list(log_idx):
        ## normal_termination_flag

        os.system(f"grep 'Normal termination' {real_prefix}_{idx}.log | wc -l > tmp_vf_{prefix}_{idx}")
        with open(f"tmp_vf_{prefix}_{idx}", "r") as f:
            validation_flag = int([ff.strip() for ff in f.readlines()][0])

        ## energy calc
        if validation_flag:
            os.system(f"grep 'SCF Done' {real_prefix}_{idx}.log > tmp_scf_{prefix}_{idx}")
            with open(f"tmp_scf_{prefix}_{idx}", "r") as f2:
                record.setdefault(idx, float(([ff.strip() for ff in f2.readlines()][0]).split()[-5]))

            dih_1.append(get_initial_angle(f"{real_prefix}_{idx}.log", dih_1_string))
            dih_2.append(get_initial_angle(f"{real_prefix}_{idx}.log", dih_2_string))

    os.system("rm -f tmp_*")
    #return record, real_prefix, dih_1_initial, dih_2_initial
    return record, real_prefix, dih_1, dih_2, dih_1_string, dih_2_string

def get_conf(log_name, pdb_prefix):
    log2pdb = ["100", "2", "1", f"{pdb_prefix}.pdb", "0", "q"]
    with open(f"tmp_log2pdb_{pdb_prefix}", "w+") as c:
        for ss in log2pdb:
            c.write(ss + "\n")

    os.system(f"Multiwfn {log_name} < tmp_log2pdb_{pdb_prefix} > /dev/null")
    time.sleep(1)
    os.system(f"rm -f tmp_log2pdb_{pdb_prefix}")
    return

def stat_and_graph(prefix, ini):
    record, real_prefix, dih_1, dih_2, dih_1_string, dih_2_string = result_process(prefix, ini)
    #record, real_prefix, dih_1_initial, dih_2_initial = result_process(prefix, dih_1_string, dih_2_string)

    #start_angle_list = [int(i) for i in start_angle.strip().split(",")]
    ## energy barrier
    energy = [float(i) for i in list(record.values())]
    relative_energy = np.array([(i - min(energy))*627.51  for i in energy])
    #upper_level = math.ceil(relative_energy.max())
    #strip_zero = [i for i in energy if i != 0.0]
    barrier = (max(energy) - min(energy))* 627.15

    df = pd.DataFrame({f"Dih1:{'-'.join(dih_1_string.split(','))}/degree":dih_1,\
                       f"Dih2:{'-'.join(dih_2_string.split(','))}/degree":dih_2, \
                       "Energy/kcal.mol-1":relative_energy})
    df.to_csv(f"Data_RigidScan_{prefix}.csv",index=None)

    #terval = 360 // int(rotable_terval)

    unique_dih1 = []
    for i in range(len(dih_1)):
        if dih_1[i] not in unique_dih1:
            unique_dih1.append(dih_1[i])

    unique_dih2 = []
    for i in range(len(dih_2)):
        if dih_2[i] not in unique_dih2:
            unique_dih2.append(dih_2[i])


    dih_1_2 = []
    #dih_1_2 = list(zip(dih_1, dih_2))
    ##create zipped dih_pair
    for i in range(len(unique_dih1)):
        for j in range(len(unique_dih2)):
            dih_1_2.append(tuple([unique_dih1[i], unique_dih2[j]]))

    zipped_dih_pair = list(zip(dih_1, dih_2))
    zipped_content = list(zip(dih_1, dih_2, relative_energy))
    update_content = []

    for i in range(len(dih_1_2)):
        if dih_1_2[i] not in zipped_dih_pair:
            update_content.append(tuple([dih_1_2[i][0], dih_1_2[i][1], -0.1]))
        else:
            idx = zipped_dih_pair.index(dih_1_2[i])
            update_content.append(tuple([dih_1_2[i][0], dih_1_2[i][1], zipped_content[idx][-1]]))

    new_dih1, new_dih2, new_energy = zip(*(tuple(update_content)))

    new_dih1 = list(new_dih1)
    new_dih2 = list(new_dih2)
    new_energy = np.array(list(new_energy))

    upper_level = math.ceil(new_energy.max())

    torsion_angle_A = []
    for i in range(len(new_dih1)):
        if new_dih1[i] not in torsion_angle_A:
            torsion_angle_A.append(new_dih1[i])

    torsion_angle_B = []
    for i in range(len(new_dih2)):
        if new_dih2[i] not in torsion_angle_B:
            torsion_angle_B.append(new_dih2[i])

    #torsion_angle_A = set([(int(i) // 18)*20 + int(dih_1_initial) for i in list(record.keys())[:-1]])
    #torsion_angle_B = set([((int(i)-1) % 18)*20 + int(dih_2_initial) for i in list(record.keys())])

    #X_smooth = np.linspace(max(X_torsion_angle), min(X_torsion_angle), 200)
    #Y_smooth = make_interp_spline(X_torsion_angle,relative_energy)(X_smooth)
    if upper_level > 30.0:
        ## save graph
        plt.figure(figsize=(12,8), dpi=200)
        #X, Y = np.meshgrid(list(torsion_angle_A), list(torsion_angle_B))
        Z = new_energy.reshape(len(torsion_angle_A), len(torsion_angle_B))
        level_basic = [i*0.1 for i in range(-1, 201)]
        level_extend = [20] + [i*2+0.5 for i in range(10, (upper_level // 2)+2)]
        a = plt.contourf(Z, levels=level_basic, cmap=plt.cm.rainbow)
        a1 = plt.contourf(Z, levels=level_extend, cmap=plt.cm.autumn)

        #plt.plot(X_torsion_angle, relative_energy, 'bo')
        #plt.plot(X_smooth, Y_smooth, 'g')
        #y_ticks = list(set([(int(i) // (18+1))*20 + int(dih_1_initial) for i in list(record.keys())]))
        #x_ticks = list(set([((int(i)-1) % 18)*20 + int(dih_2_initial) for i in list(record.keys())]))
        #x_ticks.sort()
        #y_ticks.sort()
        y_ticks = torsion_angle_A
        x_ticks = torsion_angle_B
        plt.xticks([i for i in range(len(x_ticks))], x_ticks)
        plt.yticks([i for i in range(len(y_ticks))], y_ticks)

        plt.ylabel(f"Dih1:{'-'.join(dih_1_string.split(','))}/degree")
        plt.xlabel(f"Dih2:{'-'.join(dih_2_string.split(','))}/degree")

        #plt.legend()

        #y_teval = (max(energy) - min(energy)) // 5
        #x_terval = (max(X_torsion_angle) - min(X_torsion_angle)) // 8

        #plt.xticks(["{:.0f}".format((min(X_torsion_angle) -  x_terval/2) + i*x_terval) for i in range(10)], \
        #           [(min(X_torsion_angle) - x_terval/2) + i*x_terval for i in range(10)])

        #plt.yticks(["{:.2f}".format((min(energy) - y_teval/2) + i*y_teval) for i in range(7)], \
        #           [(min(energy) - y_teval/2) + i*y_teval for i in range(7)])
        plt.colorbar(a)
        #plt.savefig(f"Contour_RigidScan_{prefix}-1.png")
        plt.colorbar(a1)
        plt.savefig(f"Contour_RigidScan_{prefix}.png")

    else:
        plt.figure(figsize=(12,8), dpi=200)
        #X, Y = np.meshgrid(list(torsion_angle_A), list(torsion_angle_B))
        Z = new_energy.reshape(len(torsion_angle_A), len(torsion_angle_B))
        level_basic = [i*0.1 for i in range(-1, upper_level*10+1)]
        a = plt.contourf(Z, levels=level_basic, cmap=plt.cm.rainbow)
        #y_ticks = list(set([(int(i) // (18+1))*20 + int(dih_1_initial) for i in list(record.keys())]))
        #x_ticks = list(set([((int(i)-1) % 18)*20 + int(dih_2_initial) for i in list(record.keys())]))
        #x_ticks.sort()
        #y_ticks.sort()
        y_ticks = torsion_angle_A
        x_ticks = torsion_angle_B

        plt.xticks([i for i in range(len(x_ticks))], x_ticks)
        plt.yticks([i for i in range(len(y_ticks))], y_ticks)

        plt.ylabel(f"Dih1:{'-'.join(dih_1_string.split(','))}/degree")
        plt.xlabel(f"Dih2:{'-'.join(dih_2_string.split(','))}/degree")
        plt.colorbar(a)
        plt.savefig(f"Contour_RigidScan_{prefix}.png")


    ## find the  minimum and maximun
    minima = {}
    maxima = {}

    ##expand Z
    pre_exp_Z = np.concatenate([Z, Z, Z], axis=1)
    exp_Z = np.concatenate([pre_exp_Z, pre_exp_Z, pre_exp_Z], axis=0)
    #print(Z.shape, exp_Z.shape)

    i = 0
    while i < len(torsion_angle_A):
        shift_i = i + len(torsion_angle_A)
        ## maxtrix
        j = 0
        while j < len(torsion_angle_B):
            pp_8s = []
            ## should always be 3x3 dimensional to define the local minima and maxima
            ## local should be defined by sorrounding 8 points
            shift_j = j + len(torsion_angle_B)

            p11 = exp_Z[shift_i-1, shift_j-1]
            if p11 < 0:
                f1 = 1
                while f1 < min(len(torsion_angle_A), len(torsion_angle_B))-1:
                    if exp_Z[shift_i-1-f1, shift_j-1-f1] >=0:
                        p11 = exp_Z[shift_i-1-f1, shift_j-1-f1]
                        break
                    else:
                        f1 += 1
            pp_8s.append(p11)

            p12 = exp_Z[shift_i-1, shift_j]
            if p12 < 0:
                f2 = 1
                while f2 < min(len(torsion_angle_A), len(torsion_angle_B))-1:
                    if exp_Z[shift_i-1-f2, shift_j] >=0:
                        p12 = exp_Z[shift_i-1-f2, shift_j]
                        break
                    else:
                        f2 += 1
            pp_8s.append(p12)

            p13 = exp_Z[shift_i-1, shift_j+1]
            if p13 < 0:
                f3 = 1
                while f3 < min(len(torsion_angle_A), len(torsion_angle_B))-1:
                    if exp_Z[shift_i-1-f3, shift_j+1+f3] >=0:
                        p13 = exp_Z[shift_i-1-f3, shift_j+1+f3]
                        break
                    else:
                        f3+=1
            pp_8s.append(p13)

            p21 = exp_Z[shift_i, shift_j-1]
            if p21 < 0:
                f21 = 1
                while f21 < min(len(torsion_angle_A), len(torsion_angle_B))-1:
                    if exp_Z[shift_i, shift_j-1-f21] >= 0:
                        p21 = exp_Z[shift_i, shift_j-1-f21]
                        break
                    else:
                        f21 += 1
            pp_8s.append(p21)

            p23 = exp_Z[shift_i, shift_j+1]
            if p23 < 0:
                f23 = 1
                while f23 < min(len(torsion_angle_A), len(torsion_angle_B))-1:
                    if exp_Z[shift_i, shift_j+1+f23] >= 0:
                        p23 = exp_Z[shift_i, shift_j+1+f23]
                        break
                    else:
                        f23 += 1
            pp_8s.append(p23)

            p31 = exp_Z[shift_i+1, shift_j-1]
            if p31 < 0:
                f31 = 1
                while f31 < min(len(torsion_angle_A), len(torsion_angle_B))-1:
                    if exp_Z[shift_i+1+f31, shift_j-1-f31] >= 0:
                        p31 = exp_Z[shift_i+1+f31, shift_j-1-f31]
                        break
                    else:
                        f31 += 1
            pp_8s.append(p31)

            p32 = exp_Z[shift_i+1, shift_j]
            if p32 < 0:
                f32 = 1
                while f32 < min(len(torsion_angle_A), len(torsion_angle_B))-1:
                    if exp_Z[shift_i+1+f32, shift_j] >= 0:
                        p32 = exp_Z[shift_i+1+f32, shift_j]
                        break
                    else:
                        f32 += 1
            pp_8s.append(p32)

            p33 = exp_Z[shift_i+1, shift_j+1]
            if p33 < 0:
                f33 = 1
                while f33 < min(len(torsion_angle_A), len(torsion_angle_B))-1:
                    if exp_Z[shift_i+1+f33, shift_j+1+f33] >= 0:
                        p33 = exp_Z[shift_i+1+f33, shift_j+1+f33]
                        break
                    else:
                        f33 += 1
            pp_8s.append(p33)

            if exp_Z[shift_i, shift_j] >=0 and exp_Z[shift_i, shift_j] < min(pp_8s) and exp_Z[shift_i, shift_j] not in minima.values():
                minima.setdefault(f"{i}_{j}", exp_Z[shift_i, shift_j])
            if exp_Z[shift_i, shift_j] > max(pp_8s) and exp_Z[shift_i, shift_j] not in maxima.values():
                maxima.setdefault(f"{i}_{j}", exp_Z[shift_i, shift_j])

            j += 1
        i += 1

    sorted_minima = sorted(minima.items(), key=lambda d:d[1])
    sorted_maxima = sorted(maxima.items(), key=lambda d:d[1])

    global_minima = sorted_minima[0]
    global_maxima = sorted_maxima[-1]

    #print(minima, maxima)
    #print(global_minima, global_maxima)

    ##save global minima
    ##find the properate idx from record

    gmmin_idx = int(global_minima[0].split('_')[0])*len(torsion_angle_B) + int(global_minima[0].split('_')[-1])
    gmmin_energy = new_energy[gmmin_idx]
    get_real_idx_gmmin =[idx for idx in range(len(list(record.keys()))) \
                          if (dih_1[idx] == new_dih1[gmmin_idx]) and (dih_2[idx] == new_dih2[gmmin_idx])][0]
    find_true_gmmin_idx = list(record.keys())[get_real_idx_gmmin]
    #find_true_gmmin_idx = [idx for idx in record.keys() if (dih_1[idx-1] == new_dih1[gmmin_idx]) and (dih_2[idx-1] == new_dih2[gmmin_idx])][0]
    global_minima_log = f"rigid_{prefix}_{find_true_gmmin_idx}.log"
    global_minima_pdb_prefix = f"Conf_GlobalMinima_Dih1:{new_dih1[gmmin_idx]}_Dih2:{new_dih2[gmmin_idx]}"
    get_conf(global_minima_log, global_minima_pdb_prefix)

    ## save globale maxima
    gmmax_idx = int(global_maxima[0].split('_')[0])*len(torsion_angle_B) + int(global_maxima[0].split('_')[-1])
    gmmax_energy = new_energy[gmmax_idx]
    get_real_idx_gmmax = [idx for idx in range(len(list(record.keys()))) \
                         if (dih_1[idx] == new_dih1[gmmax_idx]) and (dih_2[idx] == new_dih2[gmmax_idx])][0]
    find_true_gmmax_idx = list(record.keys())[get_real_idx_gmmax]
    #find_true_gmmax_idx = [idx for idx in record.keys() if (dih_1[idx-1] == new_dih1[gmmax_idx]) and (dih_2[idx-1] == new_dih2[gmmax_idx])][0]
    global_maxima_log = f"rigid_{prefix}_{find_true_gmmax_idx}.log"
    global_maxima_pdb_prefix = f"Conf_GlobalMaxima_Dih1:{new_dih1[gmmax_idx]}_Dih2:{new_dih2[gmmax_idx]}"
    get_conf(global_maxima_log, global_maxima_pdb_prefix)

    if len(sorted_minima) > 1:
        for point in sorted_minima[1:]:
            minima_idx = int(point[0].split('_')[0])*len(torsion_angle_B) + int(point[0].split('_')[-1])
            minima_energy = new_energy[minima_idx]
            get_real_idx_min = [idx for idx in range(len(list(record.keys()))) \
                               if (dih_1[idx] == new_dih1[minima_idx]) and (dih_2[idx] == new_dih2[minima_idx])][0]
            find_true_minima_idx = list(record.keys())[get_real_idx_min]
            #find_true_minima_idx = [idx for idx in record.keys() if (dih_1[idx-1] == new_dih1[minima_idx]) and (dih_2[idx-1] == new_dih2[minima_idx])][0]
            minima_log = f"rigid_{prefix}_{find_true_minima_idx}.log"
            minima_pdb_prefix = f"Conf_LocalMinima_Dih1:{new_dih1[minima_idx]}_Dih2:{new_dih2[minima_idx]}"
            get_conf(minima_log, minima_pdb_prefix)

    if len(sorted_maxima) > 1:
        for point2 in sorted_maxima[0:-1]:
            maxima_idx = int(point2[0].split('_')[0])*len(torsion_angle_B) + int(point2[0].split('_')[-1])
            maxima_energy = new_energy[maxima_idx]
            get_real_idx_max = [idx for idx in range(len(list(record.keys()))) \
                                if (dih_1[idx] == new_dih1[maxima_idx]) and (dih_2[idx] == new_dih2[maxima_idx])][0]
            find_true_maxima_idx = list(record.keys())[get_real_idx_max]
            #find_true_maxima_idx = [idx for idx in record.keys() if (dih_1[idx-1] == new_dih1[maxima_idx]) and (dih_2[idx-1] == new_dih2[maxima_idx])][0]
            minima_log = f"rigid_{prefix}_{find_true_maxima_idx}.log"
            minima_pdb_prefix = f"Conf_LocalMaxima_Dih1:{new_dih1[maxima_idx]}_Dih2:{new_dih2[maxima_idx]}"
            get_conf(minima_log, minima_pdb_prefix)


    with open("Result_summary.txt", "w+") as cc:
        cc.write(f"Done with system {prefix}, rigid energy barrier: {barrier:.3f} /kcal.mol-1 ")
        cc.write(f"Scan result has been saved in Data_RigidScan_{prefix}.csv\n")
        cc.write(f"Graph has been saved in Graph_RigidScan_{prefix}.png\n")
        cc.write("Conformation for local minima points are saved as Conf_LocalMinima_*.pdb, if any\n")
        cc.write("Conformation for local maxima points are saved as Conf_LocalMaxima_*.pdb, if any\n")
        cc.write(f"Conformation for global minima points are saved as {global_minima_pdb_prefix}.pdb \n")
        cc.write(f"Conformation for global maxima points are saved as {global_maxima_pdb_prefix}.pdb \n")
    #print(f"Conformations with highest energy have been saved in Conf_{prefix}_{max_key}.pdb")
    #print(f"Conformations with lowest energy have been saved in Conf_{prefix}_{min_key}.pdb")
    os.system("rm -f tmp_*")
    if not os.path.exists("./Results"):
        os.mkdir("./Results")
    os.system("mv *.pdb ./Results")
    os.system("mv *.png ./Results")
    os.system("mv *.txt ./Results")
    os.system("mv *.csv ./Results")

    return print("Check Result_summary.txt in ./Results for detailed results")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='processing rigid scan results')
    parser.add_argument('--prefix', type=str, required=True, \
                        help='system prefix')
    parser.add_argument('--ini', type=str, required=True, \
                        help='paramater file name')

    """
    parser.add_argument('--dih_1_string', type=str, required=True, \
                        help='torsion atom index, seperated by comma, sample: 1,2,5,7')
    parser.add_argument('--dih_2_string', type=str, required=True, \
                        help='torsion atom index, seperated by comma, sample: 5,7,10,11')
    """
    args = parser.parse_args()

    #stat_and_graph(args.prefix,args.dih_1_string, args.dih_2_string)
    stat_and_graph(args.prefix, args.ini)
