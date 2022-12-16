import pandas as pd
import os
import random
import math
import numpy as np
import time
import argparse
import sys

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

def mol2xyz(prefix):
    file = [name.strip() for name in os.listdir() if prefix in name][0]
    mol2xyz = ["100", "2", "2", f"{prefix}.mol2xyz", "0", "q"]
    with open("tmp_mol2xyz", "w+") as c:
        for ss in mol2xyz:
            c.write(ss + "\n")

    os.system(f"Multiwfn {file} < tmp_mol2xyz > /dev/null")
    time.sleep(20)
    return

def log2xyz(prefix):
    log2xyz = ["100", "2", "2", f"{prefix}.log2xyz", "0", "q"]
    with open("tmp_log2xyz", "w+") as c:
        for ss in log2xyz:
            c.write(ss + "\n")

    os.system(f"Multiwfn opt_{prefix}.log < tmp_log2xyz > /dev/null")
    time.sleep(20)
    return

def pyscf_coord2xyz(prefix):
    ## collect np.array

    with open(f"pyscf_opted_{prefix}.coords", "r") as f:
        opted_coords = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

    mol2xyz(prefix)
    with open(f"{prefix}.mol2xyz", "r") as f_xyz:
        content = [line.strip() for line in f_xyz.readlines() if len(line.strip()) > 0]

    atom_xyz = content[2:]
    header = content[:2]

    with open(f"opt_{prefix}.coords2xyz", "w+") as c:
        for line in header:
            c.write(line + "\n")
        for i in range(len(opted_coords)):
            atom = (atom_xyz[i]).split()[0]
            xyz_coords = [eval(coords) for coords in opted_coords[i].split()]
            c.write(f"{atom:<2s}{xyz_coords[0]:>16.8f}{xyz_coords[1]:>16.8f}{xyz_coords[2]:>16.8f}\n")
    return


def prepare_opt_gaussian(prefix, ini):
    qm_opt_engine, qm_scan_engine,charge, spin, dih_1_string, dih_2_string, terval1, terval2, \
           funtional, if_d3, opt_basis, scan_basis, if_chk = param(ini)
    ## collect xyz for gjf preparation
    #run_at(ini)

    mol2xyz(prefix)

    with open(f"{prefix}.mol2xyz", "r") as f_xyz:
        atom_xyz = [line.strip() for line in f_xyz.readlines() if len(line.strip()) > 0][2:]

    with open(f"opt_{prefix}.gjf", "w+") as c:
        if if_chk:
            c.write(f"%chk=opt_{prefix}.chk\n")
        else:
            pass
        c.write("%mem=10GB\n")
        c.write("%nproc=14\n")
        if if_d3:
            cmd = f"# opt {funtional}/{opt_basis} em=gd3bj nosymm \n"
        else:
            cmd = f"# opt {funtional}/{opt_basis} nosymm \n"

        c.write(cmd)
        c.write("\n")
        c.write(f"opt {prefix}\n")
        c.write("\n")
        c.write(f"{charge} {spin}\n")

        for line in atom_xyz:
            c.write(line + "\n")

        c.write("\n")
        c.write("\n")
        c.write("\n")

    os.system("rm -f *xyz")
    return print(f"Finish prepare opt_{prefix}.gjf for g16 run")

def prepare_opt_pycsf(prefix, ini):
    qm_opt_engine, qm_scan_engine,charge, spin, dih_1_string, dih_2_string, terval1, terval2, \
           funtional, if_d3, opt_basis, scan_basis, if_chk = param(ini)

    mol2xyz(prefix)

    with open(f"{prefix}.mol2xyz", "r") as f_xyz:
        atom_xyz = [line.strip() for line in f_xyz.readlines() if len(line.strip()) > 0][2:]

    with open(f"opt_pyscf_{prefix}.xyz", "w+") as py_xyz:
        for line in atom_xyz:
            py_xyz.write(line + "\n")
    os.system(f"cat ./pyscf_opt_header.py > opt_pyscf_{prefix}.py")
    with open(f"opt_pyscf_{prefix}.py", "a+") as c:
        c.write("\n")
        c.write("mol = gto.Mole() \n")
        c.write ("mol.verbose=5 \n")
        c.write("\n")

        c.write(f"mol.atom = open('opt_pyscf_{prefix}.xyz').read()\n")
        c.write(f"mol.basis = '{opt_basis}' \n")
        c.write("mol.charge = " + str(charge) + "\n")
        c.write("mol.symmetry = False \n")
        c.write("mol.spin = " + str(spin-1) + "\n")
        c.write("mol.build() \n")
        c.write("\n")

        c.write(f"opt = dft.RKS(mol, xc='{funtional}')\n")

    os.system(f"cat ./pyscf_opt_geomparam.py >> opt_pyscf_{prefix}.py")

    with open(f"opt_pyscf_{prefix}.py", "a+") as c2:
        c2.write("mol_eq = geometric_solver.optimize(opt, **conv_params_geomtric)\n")
        if if_d3:
            c2.write(f"opt = dftd3(dft.RKS(mol_eq, xc='{funtional}')).run()\n")
        else:
            c2.write(f"opt = dft.RKS(mol_eq, xc='{funtional}').run()\n")
        c2.write("coords = mol_eq.atom_coords()\n")
        c2.write(f"np.savetxt('pyscf_opted_{prefix}.coords', coords)\n")

    os.system("rm -f *mol2xyz")
    return print(f"Finish prepare opt_pyscf_{prefix}.xyz and opt_pyscf_{prefix}.py for pyscf run")

def prepare_scan_gaussian(prefix, ini):
    qm_opt_engine, qm_scan_engine,charge, spin, dih_1_string, dih_2_string, terval1, terval2, \
           funtional, if_d3, opt_basis, scan_basis, if_chk = param(ini)


    #run_at(ini)
    if qm_opt_engine == 'g16':
        log2xyz(prefix)

        with open(f"{prefix}.log2xyz", "r") as xyz:
            N = len([ff.strip() for ff in xyz.readlines() if len(ff.strip()) > 0]) - 2

        os.system(f"mv {prefix}.log2xyz xyz")
        #with open("xyz", "w+") as new_xyz:
        #    for line in atom_xyz:
        #        new_xyz.write(line + "\n")

    elif qm_opt_engine == 'pyscf':
        pyscf_coord2xyz(prefix)
        with open(f"opt_{prefix}.coords2xyz", "r") as xyz:
            N = len([ff.strip() for ff in xyz.readlines() if len(ff.strip()) > 0]) - 2

        os.system(f"mv opt_{prefix}.coords2xyz xyz")

    else:
        return print("Provide available qm method")

    simplified_bond1="-".join(dih_1_string.strip().split(",")[1:3])
    simplified_bond2="-".join(dih_2_string.strip().split(",")[1:3])

    with open("gentor.ini", "w+") as cc:
        cc.write(simplified_bond1 + "\n")
        run_terval1= f"e{terval1}"
        cc.write(run_terval1 + "\n")
        cc.write(simplified_bond2 + "\n")
        run_terval2= f"e{terval1}"
        cc.write(run_terval2 + "\n")


    #with open("gentor.ini", "w+") as cc:
    #    cc.write(bond + "\n")
    #    run_terval = f"e{rotable_terval}"
    #    cc.write(run_terval + "\n")

    #os.system(f"cp {tool_file}/run_gentor ./")
    os.system("./run_gentor > /dev/null")
    time.sleep(3)

    with open("traj.xyz", "r") as traj:
        content = [line.strip() for line in traj.readlines()]

    conformations = [line for line in content if len(line.split()) == 4]
    n_frame = len([line for line in content if "Conformation" in line])

    #n_frame = rotable_angle // rotable_terval

    assemble_atom = {}

    i = 1
    while i <= n_frame:
        current = conformations[(i-1)*N:i*N]
        assemble_atom.setdefault(i, current)
        i += 1

    for key, value in assemble_atom.items():
        with open(f"rigid_{prefix}_{key}.gjf", "w+") as c:
            if if_chk:
                c.write(f"%chk=rigid_{prefix}_{key}.chk\n")
            else:
                pass
            c.write("%mem=10GB\n")
            c.write("%nproc=14\n")
            if if_d3:
                cmd = f"# {funtional}/{scan_basis} em=gd3bj nosymm \n"
            else:
                cmd = f"# {funtional}/{scan_basis} nosymm \n"

            c.write(cmd)
            c.write("\n")
            c.write(f"rigd_scan_{key}\n")
            c.write("\n")
            c.write(f"{charge} {spin}\n")

            for line in value:
                c.write(line + "\n")

            c.write("\n")
            c.write("\n")
            c.write("\n")

    ## compose run script
    return print(f"Done with rigid scan preparison, save as rigid_{prefix}_*.gjf, {n_frame} gjf file in total")

def prepare_scan_pyscf(prefix, ini):
    qm_opt_engine, qm_scan_engine,charge, spin, dih_1_string, dih_2_string, terval1, terval2, \
           funtional, if_d3, opt_basis, scan_basis, if_chk = param(ini)

    if qm_opt_engine == 'g16':
        log2xyz(prefix)
        os.system(f"mv {prefix}.log2xyz xyz")

    elif qm_opt_engine == 'pyscf':
        pyscf_coord2xyz(prefix) ##opt_{prefix}.coords2xyz
        os.system(f"mv opt_{prefix}.coords2xyz  xyz")

    else:
        return print("Provide available qm method")

    simplified_bond1="-".join(dih_1_string.strip().split(",")[1:3])
    simplified_bond2="-".join(dih_2_string.strip().split(",")[1:3])

    with open("gentor.ini", "w+") as cc:
        cc.write(simplified_bond1 + "\n")
        run_terval1= f"e{terval1}"
        cc.write(run_terval1 + "\n")
        cc.write(simplified_bond2 + "\n")
        run_terval2= f"e{terval1}"
        cc.write(run_terval2 + "\n")

    #with open("gentor.ini", "w+") as cc:
    #    cc.write(bond + "\n")
    #    run_terval = f"e{rotable_terval}"
    #    cc.write(run_terval + "\n")

    #os.system(f"cp {tool_file}/run_gentor ./")
    os.system("./run_gentor > /dev/null")
    time.sleep(2)

    with open("xyz", "r") as xyz:
        N = len([ff.strip() for ff in xyz.readlines() if len(ff.strip()) > 0]) - 2

    with open("traj.xyz", "r") as traj:
        content = [line.strip() for line in traj.readlines()]

    conformations = [line for line in content if len(line.split()) == 4]
    n_frame = len([line for line in content if "Conformation" in line])

    assemble_atom = {}

    i = 1
    while i <= n_frame:
        current = conformations[(i-1)*N:i*N]
        assemble_atom.setdefault(i, current)
        i += 1

    for key, value in assemble_atom.items():
        with open(f"rs_pyscf_{prefix}_{key}.xyz", "w+") as ccc:
            for line in value:
                ccc.write(line + "\n")

    os.system(f"cat ./pyscf_opt_header.py > rs_pyscf_{prefix}.py")
    with open(f"rs_pyscf_{prefix}.py", "a+") as scan:
        scan.write("\n")
        scan.write("mol = gto.Mole() \n")
        scan.write ("mol.verbose=5 \n")
        scan.write("\n")

        scan.write(f"mol.atom = open(sys.argv[-1]).read()\n")
        scan.write(f"mol.basis = '{scan_basis}'\n")
        scan.write("mol.charge = " + str(charge) + "\n")
        scan.write("mol.symmetry = False \n")
        scan.write("mol.spin = " + str(spin-1) + "\n")
        scan.write("mol.build() \n")
        scan.write("\n")

        if if_d3:
            scan.write(f"sp = dftd3(dft.RKS(mol, xc='{funtional}'))\n")
        else:
            scan.write(f"sp = dft.RKS(mol, xc='{funtional}')\n")

        scan.write("sp.kernel()\n")

    return print(f"Done with rigid scan preparison, save as rs_pyscf_{prefix}_*.xyz, {n_frame} xyz file in total")

def prepare_opt(prefix,ini):
    qm_opt_engine, qm_scan_engine,charge, spin, dih_1_string, dih_2_string, terval1, terval2, \
           funtional, if_d3, opt_basis, scan_basis, if_chk = param(ini)

    ## opt_preparation
    if qm_opt_engine == 'g16':
        prepare_opt_gaussian(prefix, ini)
    elif qm_opt_engine == 'pyscf':
        prepare_opt_pycsf(prefix, ini)
    else:
        return print("provide suitable qm method")

    return

def prepare_scan(prefix, ini):
    qm_opt_engine, qm_scan_engine,charge, spin, dih_1_string, dih_2_string, terval1, terval2, \
           funtional, if_d3, opt_basis, scan_basis, if_chk = param(ini)

    if qm_scan_engine == 'g16':
        prepare_scan_gaussian(prefix, ini)
    elif qm_scan_engine == 'pyscf':
        prepare_scan_pyscf(prefix, ini)
    else:
        return print("provide suitable qm method")

    return

def run(mode, prefix, ini):
    if mode == 'opt':
        prepare_opt(prefix, ini)
    elif mode == 'scan':
        prepare_scan(prefix, ini)
    else:
        return print("Nothing should to do")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare/run system for rigid dihedral scan based on gaussian16')
    parser.add_argument('--mode', type=str, required=True, \
                        help='prepare mode, can be "opt" or "scan"')
    parser.add_argument('--prefix', type=str, required=True, \
                        help='system prefix')

    parser.add_argument('--ini', type=str, required=True, \
                        help='paramater file name')
    args = parser.parse_args()

    run(args.mode, args.prefix, args.ini)
