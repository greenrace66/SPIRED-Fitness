from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import click
import esm
import numpy as np
import pandas as pd
import torch

from scripts.model import SPIRED_Fitness_Union, SPIRED_Stab
from scripts.utils_train_valid import getDataTest, getStabDataTest

from .utils import load_records, unique_record_name
from .weights import model_path

AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")
AA_DICT = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "ALA",
}

DOUBLE_MUT_LIST = list(itertools.product(AMINO_ACIDS, AMINO_ACIDS, repeat=1))
DOUBLE_MUT_DICT = {index: "".join(value) for index, value in enumerate(DOUBLE_MUT_LIST)}
DOUBLE_MUT_DICT_INVERSE = {value: index for index, value in DOUBLE_MUT_DICT.items()}


@dataclass
class FitnessResources:
    model: SPIRED_Fitness_Union
    esm2_650m: torch.nn.Module
    esm2_3b: torch.nn.Module
    esm2_batch_converter: object
    esm1v_models: tuple[torch.nn.Module, ...]
    esm1v_alphabet: object
    esm1v_batch_converter: object


@dataclass
class StabResources:
    model: SPIRED_Stab
    esm2_650m: torch.nn.Module
    esm2_3b: torch.nn.Module
    esm2_batch_converter: object


def load_fitness_resources(model_dir: Path | None = None) -> FitnessResources:
    model = SPIRED_Fitness_Union(device_list=["cpu", "cpu", "cpu", "cpu"])
    state_dict = torch.load(model_path("SPIRED-Fitness.pth", model_dir), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    esm2_650m, _ = esm.pretrained.esm2_t33_650M_UR50D()
    esm2_650m.eval()
    esm2_3b, esm2_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    esm2_3b.eval()
    esm2_batch_converter = esm2_alphabet.get_batch_converter()

    esm1v_1, _ = esm.pretrained.esm1v_t33_650M_UR90S_1()
    esm1v_2, _ = esm.pretrained.esm1v_t33_650M_UR90S_2()
    esm1v_3, _ = esm.pretrained.esm1v_t33_650M_UR90S_3()
    esm1v_4, _ = esm.pretrained.esm1v_t33_650M_UR90S_4()
    esm1v_5, esm1v_alphabet = esm.pretrained.esm1v_t33_650M_UR90S_5()
    for esm1v_model in (esm1v_1, esm1v_2, esm1v_3, esm1v_4, esm1v_5):
        esm1v_model.eval()
    esm1v_batch_converter = esm1v_alphabet.get_batch_converter()

    return FitnessResources(
        model=model,
        esm2_650m=esm2_650m,
        esm2_3b=esm2_3b,
        esm2_batch_converter=esm2_batch_converter,
        esm1v_models=(esm1v_1, esm1v_2, esm1v_3, esm1v_4, esm1v_5),
        esm1v_alphabet=esm1v_alphabet,
        esm1v_batch_converter=esm1v_batch_converter,
    )


def load_stab_resources(model_dir: Path | None = None) -> StabResources:
    model = SPIRED_Stab(device_list=["cpu", "cpu", "cpu", "cpu"])
    state_dict = torch.load(model_path("SPIRED-Stab.pth", model_dir), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    esm2_650m, _ = esm.pretrained.esm2_t33_650M_UR50D()
    esm2_650m.eval()
    esm2_3b, esm2_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    esm2_3b.eval()
    esm2_batch_converter = esm2_alphabet.get_batch_converter()

    return StabResources(model=model, esm2_650m=esm2_650m, esm2_3b=esm2_3b, esm2_batch_converter=esm2_batch_converter)


def run_fitness(fasta_file: Path, output_dir: Path, model_dir: Path | None = None) -> Path:
    resources = load_fitness_resources(model_dir)
    records = load_records(fasta_file)
    if not records:
        raise click.ClickException(f"No FASTA records found in {fasta_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = set()

    with torch.no_grad():
        for index, record in enumerate(records):
            name = unique_record_name(record, index, seen_names)
            protein_dir = output_dir / name
            _write_fitness_record(str(record.seq), protein_dir, resources)

    return output_dir


def run_stab(fasta_file: Path, output_dir: Path, model_dir: Path | None = None) -> Path:
    resources = load_stab_resources(model_dir)
    records = load_records(fasta_file)
    if len(records) != 2:
        raise click.ClickException("SPIRED-Stab expects exactly two FASTA records: wild-type followed by mutant.")

    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        _write_stab_pair(str(records[0].seq), str(records[1].seq), output_dir, resources)

    return output_dir


def run_structure(fasta_file: Path, output_dir: Path, model_dir: Path | None = None) -> Path:
    resources = load_fitness_resources(model_dir)
    records = load_records(fasta_file)
    if not records:
        raise click.ClickException(f"No FASTA records found in {fasta_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = set()
    with torch.no_grad():
        for index, record in enumerate(records):
            name = unique_record_name(record, index, seen_names)
            protein_dir = output_dir / name
            _write_structure_record(str(record.seq), protein_dir, resources)

    return output_dir


def _write_fitness_record(seq: str, protein_dir: Path, resources: FitnessResources) -> None:
    protein_dir.mkdir(parents=True, exist_ok=True)
    (protein_dir / "CA_structure").mkdir(exist_ok=True)
    (protein_dir / "GDFold2").mkdir(exist_ok=True)
    (protein_dir / "features_for_downstream").mkdir(exist_ok=True)

    f1d_esm2_3b, f1d_esm2_650m, esm1v_single_logits, esm1v_double_logits, target_tokens = getDataTest(
        seq,
        resources.esm2_3b,
        resources.esm2_650m,
        *resources.esm1v_models,
        resources.esm1v_batch_converter,
        resources.esm1v_alphabet,
        resources.esm2_batch_converter,
    )

    single_pred, double_pred, predxyz, _, plddt, _, _, _, _, _, phi_psi_1d, _, _ = resources.model(
        target_tokens, f1d_esm2_3b, f1d_esm2_650m, esm1v_single_logits, esm1v_double_logits
    )

    _write_structure_outputs(seq, protein_dir, predxyz, plddt, phi_psi_1d)
    torch.save(predxyz["4th"][-1][0].permute(1, 2, 0).detach().cpu().clone(), protein_dir / "features_for_downstream" / "3d.pt")
    torch.save(plddt["4th"][-1][0].detach().cpu().clone(), protein_dir / "features_for_downstream" / "plddt.pt")
    torch.save(double_pred[0].detach().cpu().clone(), protein_dir / "double_mut_pred.pt")
    _write_single_mutation_tables(seq, single_pred[0].detach().cpu().clone(), protein_dir)
    _write_double_mutation_tables(seq, double_pred[0].detach().cpu().clone(), protein_dir)


def _write_structure_record(seq: str, protein_dir: Path, resources: FitnessResources) -> None:
    protein_dir.mkdir(parents=True, exist_ok=True)
    (protein_dir / "CA_structure").mkdir(exist_ok=True)
    (protein_dir / "GDFold2").mkdir(exist_ok=True)

    f1d_esm2_3b, f1d_esm2_650m, esm1v_single_logits, esm1v_double_logits, target_tokens = getDataTest(
        seq,
        resources.esm2_3b,
        resources.esm2_650m,
        *resources.esm1v_models,
        resources.esm1v_batch_converter,
        resources.esm1v_alphabet,
        resources.esm2_batch_converter,
    )
    _, _, predxyz, _, plddt, _, _, _, _, _, phi_psi_1d, _, _ = resources.model(
        target_tokens, f1d_esm2_3b, f1d_esm2_650m, esm1v_single_logits, esm1v_double_logits
    )

    _write_structure_outputs(seq, protein_dir, predxyz, plddt, phi_psi_1d)


def _write_stab_pair(wt_seq: str, mut_seq: str, output_dir: Path, resources: StabResources) -> None:
    (output_dir / "wt" / "CA_structure").mkdir(parents=True, exist_ok=True)
    (output_dir / "wt" / "GDFold2").mkdir(parents=True, exist_ok=True)
    (output_dir / "mut" / "CA_structure").mkdir(parents=True, exist_ok=True)
    (output_dir / "mut" / "GDFold2").mkdir(parents=True, exist_ok=True)

    mut_pos_torch_list = torch.tensor((np.array(list(wt_seq)) != np.array(list(mut_seq))).astype(int).tolist())

    wt_f1d_esm2_3b, wt_f1d_esm2_650m, wt_target_tokens = getStabDataTest(
        wt_seq, resources.esm2_3b, resources.esm2_650m, resources.esm2_batch_converter
    )
    mut_f1d_esm2_3b, mut_f1d_esm2_650m, mut_target_tokens = getStabDataTest(
        mut_seq, resources.esm2_3b, resources.esm2_650m, resources.esm2_batch_converter
    )

    wt_data = {"target_tokens": wt_target_tokens, "esm2-3B": wt_f1d_esm2_3b, "embedding": wt_f1d_esm2_650m}
    mut_data = {"target_tokens": mut_target_tokens, "esm2-3B": mut_f1d_esm2_3b, "embedding": mut_f1d_esm2_650m}
    ddg, dtm, wt_features, mut_features = resources.model(wt_data, mut_data, mut_pos_torch_list)

    pd.DataFrame({"ddG": ddg.item(), "dTm": dtm.item()}, index=[0]).to_csv(output_dir / "pred.csv", index=False)
    _write_structure_outputs(wt_seq, output_dir / "wt", wt_features["Predxyz"], wt_features["Plddt"], wt_features["phi_psi_1D"])
    _write_structure_outputs(mut_seq, output_dir / "mut", mut_features["Predxyz"], mut_features["Plddt"], mut_features["phi_psi_1D"])


def _write_structure_outputs(seq: str, protein_dir: Path, predxyz, plddt, phi_psi_1d) -> None:
    predxyz_4th = predxyz["4th"][-1]
    plddt_value = plddt["4th"][-1][0]
    plddt_value_l = torch.mean(plddt_value, dim=1)
    plddt_top8_idx = torch.topk(plddt_value_l, 8)[-1]
    plddt_value_top8 = plddt_value[plddt_top8_idx, :]
    xyz_top8 = predxyz_4th[0, :, plddt_top8_idx, :]
    xyz_top8 = xyz_top8.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
    phi_psi_1d = phi_psi_1d[0].permute(1, 0).cpu().detach().numpy().astype(np.float32)
    plddt_top8_idx = plddt_top8_idx.cpu().detach().numpy().astype(np.int32)
    plddt_value_top8 = plddt_value_top8.cpu().detach().numpy().astype(np.float32)
    np.savez(
        protein_dir / "GDFold2" / "input.npz",
        reference=plddt_top8_idx,
        translation=xyz_top8,
        dihedrals=phi_psi_1d,
        plddt=plddt_value_top8,
    )

    n_residues, length, _ = xyz_top8.shape
    for n in range(min(n_residues, 8)):
        xyz_l = xyz_top8[n, ...]
        with open(protein_dir / "CA_structure" / f"{n}.pdb", "w") as handle:
            for i in range(length):
                amino_acid = AA_DICT[seq[i]]
                xyz_ca = xyz_l[i, ...]
                x, y, z = (round(float(xyz_ca[0]), 3), round(float(xyz_ca[1]), 3), round(float(xyz_ca[2]), 3))
                handle.write(
                    "ATOM  {:>5} {:<4} {} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}  \n".format(
                        int(i + 1), "CA", amino_acid, int(i + 1), x, y, z, 1.0, 0.0, "C"
                    )
                )


def _write_single_mutation_tables(seq: str, pred: torch.Tensor, protein_dir: Path) -> None:
    data = pd.DataFrame(columns=AMINO_ACIDS)
    for i in range(len(seq)):
        for j, amino_acid in enumerate(AMINO_ACIDS):
            data.loc[i, amino_acid] = pred[i, j].item()
    data.index = list(seq)
    data.to_csv(protein_dir / "single_mut_pred_for_heatmap.csv")

    data = pd.DataFrame(columns=["pred_score"])
    for mut_pos, wt_res in enumerate(seq):
        for mut_res in AMINO_ACIDS:
            mut_info = f"{wt_res}{mut_pos}{mut_res}"
            if wt_res != mut_res:
                data.loc[mut_info, "pred_score"] = pred[mut_pos, AMINO_ACIDS.index(mut_res)].item()
    data.to_csv(protein_dir / "single_mut_pred.csv")


def _write_double_mutation_tables(seq: str, pred: torch.Tensor, protein_dir: Path) -> None:
    upper_bound = torch.topk(pred.flatten(), k=2000, largest=True)[0][-1]
    lower_bound = torch.topk(pred.flatten(), k=2000, largest=False)[0][-1]

    data = pd.DataFrame(columns=["pred_score"])
    x, y, z = torch.where(pred >= upper_bound)
    x, y, z = x.tolist(), y.tolist(), z.tolist()
    for index in range(len(x)):
        mut_info = seq[x[index]] + str(x[index]) + DOUBLE_MUT_DICT[z[index]][0] + "," + seq[y[index]] + str(y[index]) + DOUBLE_MUT_DICT[z[index]][-1]
        if x[index] != y[index] and seq[x[index]] != DOUBLE_MUT_DICT[z[index]][0] and seq[y[index]] != DOUBLE_MUT_DICT[z[index]][-1]:
            mut_info_inverse = seq[y[index]] + str(y[index]) + DOUBLE_MUT_DICT[z[index]][-1] + "," + seq[x[index]] + str(x[index]) + DOUBLE_MUT_DICT[z[index]][0]
            score = pred[x[index], y[index], z[index]].item()
            score += pred[y[index], x[index], DOUBLE_MUT_DICT_INVERSE[DOUBLE_MUT_DICT[z[index]][-1] + DOUBLE_MUT_DICT[z[index]][0]]].item()
            score /= 2
            if mut_info not in data.index and mut_info_inverse not in data.index:
                data.loc[mut_info, "pred_score"] = score
    data.to_csv(protein_dir / "double_mut_pred_top_k.csv")

    data = pd.DataFrame(columns=["pred_score"])
    x, y, z = torch.where(pred <= lower_bound)
    x, y, z = x.tolist(), y.tolist(), z.tolist()
    for index in range(len(x)):
        mut_info = seq[x[index]] + str(x[index]) + DOUBLE_MUT_DICT[z[index]][0] + "," + seq[y[index]] + str(y[index]) + DOUBLE_MUT_DICT[z[index]][-1]
        if x[index] != y[index] and seq[x[index]] != DOUBLE_MUT_DICT[z[index]][0] and seq[y[index]] != DOUBLE_MUT_DICT[z[index]][-1]:
            mut_info_inverse = seq[y[index]] + str(y[index]) + DOUBLE_MUT_DICT[z[index]][-1] + "," + seq[x[index]] + str(x[index]) + DOUBLE_MUT_DICT[z[index]][0]
            score = pred[x[index], y[index], z[index]].item()
            score += pred[y[index], x[index], DOUBLE_MUT_DICT_INVERSE[DOUBLE_MUT_DICT[z[index]][-1] + DOUBLE_MUT_DICT[z[index]][0]]].item()
            score /= 2
            if mut_info not in data.index and mut_info_inverse not in data.index:
                data.loc[mut_info, "pred_score"] = score
    data.to_csv(protein_dir / "double_mut_pred_last_k.csv")
