# Literal model inputs for dataset row 498

Every CSV contains only numeric values; row and column indices start at 0. These arrays are copied or derived from `selected_inputs.npz` and written with nine significant digits, enough to recover their float32 values. For one-dimensional masks and `aux`, the values occupy one CSV row.

`atom_X_41x18.csv` columns: element_C, element_N, element_O, element_S, element_F, element_Cl, site_class_C, site_class_N_pyridine, site_class_N_pyrrole, site_class_N_nitro, site_class_O_carbonyl, site_class_O_ether, site_class_O_anion, site_class_S_thio, formal_charge/2, atom_degree/4, site_pi_electrons/2, H_diagonal/2.

`pi_X_40x3.csv` columns: H_diagonal/2, site_pi_electrons/2, nonzero_H_neighbor_count/4. `matrix_rows_40x41.csv` columns 0–39 are H row entries; column 40 is site_pi_electrons/2. The first 10 atoms/sites are active for this molecule. Zero padding is excluded from the models by the masks.
