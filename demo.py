#!/usr/bin/env python3

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def compute_similarity_matrix(
    assay_csv, 
    smiles_col="Smiles",
    sep=";", 
    output_csv="similarity_matrix.csv"
):
    """
    Reads an assay CSV, computes the Morgan-fingerprint-based similarity matrix via RDKit,
    and saves it to a CSV. Also returns the similarity DataFrame.

    Parameters
    ----------
    assay_csv : str
        Path to the assay CSV file.
    smiles_col : str, optional
        Column name in the CSV containing SMILES. Default "Smiles".
    sep : str, optional
        Delimiter used in CSV. Default ";".
    output_csv : str, optional
        File path to save the resulting similarity matrix. Default "similarity_matrix.csv".

    Returns
    -------
    similarity_df : pd.DataFrame
        NxN similarity matrix (index=SMILES, columns=SMILES).
    """
    df = pd.read_csv(assay_csv, sep=sep)
    smiles_list = df[smiles_col].tolist()

    # Convert SMILES to RDKit Mol objects
    molecules = [Chem.MolFromSmiles(s) for s in smiles_list]

    # Compute Morgan fingerprints
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        for mol in molecules
    ]

    n = len(fingerprints)
    sim_matrix = np.zeros((n, n))

    # Calculate pairwise similarities
    for i in range(n):
        for j in range(i, n):
            similarity = FingerprintSimilarity(fingerprints[i], fingerprints[j])
            sim_matrix[i, j] = similarity
            sim_matrix[j, i] = similarity

    similarity_df = pd.DataFrame(sim_matrix, index=smiles_list, columns=smiles_list)
    similarity_df.to_csv(output_csv)
    return similarity_df



def load_assay_data(
    assay_path, 
    sep=';', 
    feature_col='Standard Value', 
    transform=True
):
    """
    Loads assay data from a CSV file and optionally applies a -log transform to the target feature.

    Parameters
    ----------
    assay_path : str
        Path to the assay CSV file.
    sep : str, optional
        CSV delimiter. Default ";".
    feature_col : str, optional
        Column with bioactivity or target values. Default "Standard Value".
    transform : bool, optional
        If True, apply -log transform (e.g., pIC50). Default True.

    Returns
    -------
    assay_df : pd.DataFrame
        The loaded assay data.
    target : pd.Series
        The extracted target values (optionally -log transformed).
    """
    assay_df = pd.read_csv(assay_path, sep=sep)
    target = assay_df[feature_col]

    if transform:
        target = -np.log(target)  # e.g., converting IC50 -> pIC50

    return assay_df, target


def train_random_forest(
    sim_df, 
    target, 
    test_size=0.2, 
    random_state=20, 
    n_estimators=100
):
    """
    Splits data, trains a Random Forest, and fits another on the full dataset
    to extract feature importances.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Similarity matrix as features (rows=molecules, columns=molecules).
    target : pd.Series
        Target (bioactivity) values.
    test_size : float, optional
        Fraction for test set. Default=0.2.
    random_state : int, optional
        Seed for reproducibility. Default=42.
    n_estimators : int, optional
        Number of trees in RandomForest. Default=100.

    Returns
    -------
    full_regressor : RandomForestRegressor
        Trained on the full dataset (for feature importances).
    feature_importances : np.ndarray
        The feature importance scores.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        sim_df, 
        target, 
        test_size=test_size, 
        random_state=random_state
    )

    # Train on the train split
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators, 
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)

    # Now train on the full dataset for feature importances
    full_regressor = RandomForestRegressor(
        n_estimators=n_estimators, 
        random_state=random_state
    )
    full_regressor.fit(sim_df, target)

    feature_importances = full_regressor.feature_importances_
    return full_regressor, feature_importances


def get_top_important_molecules(
    assay_df, 
    sim_df, 
    feature_importances, 
    smiles_col='Smiles',
    top_n=20
):
    """
    Sorts feature importances and merges them with the original assay data.

    Parameters
    ----------
    assay_df : pd.DataFrame
        The assay data (must contain 'Smiles').
    sim_df : pd.DataFrame
        Similarity matrix (columns named by SMILES).
    feature_importances : np.ndarray
        Feature importance scores for each column in `sim_df`.
    smiles_col : str, optional
        Column name for SMILES in `assay_df`. Default='Smiles'.
    top_n : int, optional
        How many of the top features to return. Default=20.

    Returns
    -------
    top_molecules_df : pd.DataFrame
        DataFrame of top N molecules with SMILES, importance, and any other metadata.
    """
    smiles_list = sim_df.columns
    importance_dict = {
        smiles: imp
        for smiles, imp in zip(smiles_list, feature_importances)
    }

    sorted_importance = sorted(
        importance_dict.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    importance_df = pd.DataFrame(sorted_importance, columns=['Smiles', 'Importance'])

    # Merge with assay data to retrieve additional info
    if smiles_col in assay_df.columns:
        merged_df = pd.merge(
            importance_df, 
            assay_df, 
            left_on='Smiles', 
            right_on=smiles_col,
            how='left'
        )
    else:
        merged_df = importance_df

    return merged_df.head(top_n)


def generate_similar_molecules_gpt(
    top_molecules_df, 
    n_to_generate=10000,
    openai_api_key=None
):
    """
    Calls a GPT-like API to generate new molecules (SMILES) similar to the top molecules.
    Uses a refined, detailed prompt for clarity and specificity.

    Parameters
    ----------
    top_molecules_df : pd.DataFrame
        DataFrame containing top molecules (must have 'Smiles' column).
    n_to_generate : int, optional
        Total # of new molecules to generate. Default=10000.
    openai_api_key : str, optional
        GPT/OpenAI API key or token. For demonstration only.

    Returns
    -------
    generated_df : pd.DataFrame
        DataFrame of newly generated molecules with one column: ['Generated_SMILES'].
    """
    # Example refined prompt:
    # (1) Provide the reference SMILES
    # (2) Request a specific count per reference
    # (3) Emphasize chemical viability, correct SMILES format, and a tabular (CSV-like) output
    # (4) Restrict extra commentary

    top_smiles_list = top_molecules_df['Smiles'].tolist()
    if len(top_smiles_list) == 0:
        return pd.DataFrame(columns=['Generated_SMILES'])

    # We'll assume we want (n_to_generate / len(top_smiles_list)) per reference molecule
    molecules_per_ref = n_to_generate // len(top_smiles_list)
    generated_smiles = []

    # ------------------------------------------------------------------------
    # import openai
    # openai.api_key = openai_api_key
    # for ref_smiles in top_smiles_list:
    #     prompt = f"""
    #     Below is a list of molecules in their SMILES format. I want to generate 
    #     molecules that are structurally similar to the above molecules. 
    #     Generate me {molecules_per_ref} such candidate molecules in SMILES format. 
    #     Ensure that these are chemically viable AND valid SMILES such that they 
    #     could theoretically be produced in a wet lab. 
    #
    #     Output these generated SMILES strings in a table (one SMILES per line, 
    #     with no additional commentary).
    #
    #     Reference Molecule:
    #     {ref_smiles}
    #     """
    #
    #     response = openai.ChatCompletion.create(
    #         model="gpt-4",
    #         messages=[{"role": "system", "content": "You are a chemistry AI."},
    #                   {"role": "user", "content": prompt}],
    #         temperature=0.7,
    #         max_tokens=1500
    #     )
    #     # Parse SMILES from response text...
    #     # new_smiles_list = parse_response_for_smiles(response)
    #     # generated_smiles.extend(new_smiles_list)
    #
    # ------------------------------------------------------------------------
    
    return generated_smiles

def filter_valid_smiles(
    smiles_df, 
    smiles_col='Generated_SMILES'
):
    """
    Filters a DataFrame of SMILES for basic RDKit validity.

    Parameters
    ----------
    smiles_df : pd.DataFrame
        DataFrame containing a column of SMILES to check.
    smiles_col : str, optional
        Column name with SMILES. Default='Generated_SMILES'.

    Returns
    -------
    valid_df : pd.DataFrame
        Subset of rows where SMILES can be parsed by RDKit.
    """
    valid_indices = []
    for idx, row in smiles_df.iterrows():
        s = row[smiles_col]
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid_indices.append(idx)

    valid_df = smiles_df.loc[valid_indices].reset_index(drop=True)
    return valid_df


def compute_similarity_generated(
    generated_df, 
    original_df,
    generated_smiles_col='Generated_SMILES',
    original_smiles_col='Smiles'
):
    """
    Computes the similarity between generated molecules and the original assay molecules.

    Parameters
    ----------
    generated_df : pd.DataFrame
        DataFrame of generated SMILES.
    original_df : pd.DataFrame
        Original assay DataFrame with SMILES.
    generated_smiles_col : str, optional
        Column name of generated SMILES. Default='Generated_SMILES'.
    original_smiles_col : str, optional
        Column name of assay SMILES. Default='Smiles'.

    Returns
    -------
    sim_generated_df : pd.DataFrame
        Similarity matrix (rows=generated, cols=original).
    """
    gen_smiles_list = generated_df[generated_smiles_col].tolist()
    orig_smiles_list = original_df[original_smiles_col].tolist()

    gen_mols = [Chem.MolFromSmiles(s) for s in gen_smiles_list]
    orig_mols = [Chem.MolFromSmiles(s) for s in orig_smiles_list]

    gen_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in gen_mols]
    orig_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in orig_mols]

    sim_matrix = np.zeros((len(gen_fps), len(orig_fps)))

    for i in range(len(gen_fps)):
        for j in range(len(orig_fps)):
            sim_matrix[i, j] = FingerprintSimilarity(gen_fps[i], orig_fps[j])

    sim_generated_df = pd.DataFrame(
        sim_matrix, 
        index=gen_smiles_list, 
        columns=orig_smiles_list
    )
    return sim_generated_df


def predict_bioactivity_for_generated(model, sim_generated_df):
    """
    Uses a trained model to predict bioactivity for generated molecules.

    Parameters
    ----------
    model : sklearn regressor
        Fitted model (e.g., RandomForest).
    sim_generated_df : pd.DataFrame
        Similarity matrix [#generated x #original]. Must align with training columns.

    Returns
    -------
    predictions : np.ndarray
        Predicted bioactivity values.
    """
    return model.predict(sim_generated_df)


def main():
    sim_df = compute_similarity_matrix(
        assay_csv="CBL-2.csv",
        smiles_col="Smiles",
        sep=";",
        output_csv="similarity_matrix.csv"
    )

    assay_df, target = load_assay_data(
        assay_path="CBL-2.csv",
        sep=";",
        feature_col="Standard Value",
        transform=True
    )

    original_smiles = sim_df.columns
    sim_df_numeric = sim_df.reset_index(drop=True)
    sim_df_numeric.columns = range(len(sim_df_numeric.columns))

    model, feature_importances = train_random_forest(
        sim_df_numeric, 
        target, 
        test_size=0.2, 
        random_state=42, 
        n_estimators=100
    )

    top_20_df = get_top_important_molecules(
        assay_df=assay_df, 
        sim_df=sim_df, 
        feature_importances=feature_importances, 
        smiles_col='Smiles',
        top_n=20
    )
    print("\nTop 20 Important Molecules:")
    print(top_20_df[['Smiles', 'Importance']])

    generated_df = generate_similar_molecules_gpt(
        top_molecules_df=top_20_df, 
        n_to_generate=10000,
        openai_api_key=None # insert API key
    )
    print(f"\nGenerated {len(generated_df)} molecules from GPT call.")

    valid_generated_df = filter_valid_smiles(generated_df, smiles_col='Generated_SMILES')
    print(f"After filtering, we have {len(valid_generated_df)} valid generated molecules.")

    generated_sim_df = compute_similarity_generated(
        generated_df=valid_generated_df,
        original_df=assay_df,
        generated_smiles_col='Generated_SMILES',
        original_smiles_col='Smiles'
    )

    generated_sim_df_ordered = generated_sim_df[original_smiles]
    generated_sim_df_numeric = generated_sim_df_ordered.reset_index(drop=True)
    generated_sim_df_numeric.columns = range(len(generated_sim_df_numeric.columns))

    predictions = predict_bioactivity_for_generated(
        model=model,
        sim_generated_df=generated_sim_df_numeric
    )

    candidate_molecules_df = valid_generated_df.copy()
    candidate_molecules_df['Predicted_Bioactivity'] = predictions

    candidate_molecules_df.sort_values(by='Predicted_Bioactivity', ascending=False, inplace=True)

    print("\nSample of generated candidate molecules with predicted bioactivity:")
    print(candidate_molecules_df.head(10))

    candidate_molecules_df.to_csv("candidate_molecules_with_predictions.csv", index=False)


if __name__ == "__main__":
    main()
