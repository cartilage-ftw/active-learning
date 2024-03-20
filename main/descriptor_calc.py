import localcider
import pandas as pd

from localcider.sequenceParameters import SequenceParameters


def estimate_descriptors(df):
    """
    Arguments:
        df: A (pandas) DataFrame object, which contains a column containing sequences; it's fine to pass
        your dataframe which additionally has both sequences and B22, this method just appends the descriptors
        in separate columns in the same dataframe

    Returns:
        A new dataframe with additional columns containing each of the descriptors
    """

    # define a dictionary that will hold the descriptor values for each seq
    descriptor_dict = {
        'Fraction Charged Residues': [], # initialize to be empty
        'Net Charge Per Residue': [],
        'Fraction Negatively Charged': [],
        'Fraction Positively Charged': [],
        'Fraction Expanding': [], # residues that contribute to expansion of the chain
        'Disorder Promoting Residues': [],
        'Kappa': [],
        'Omega': [],
        'Mean Hydropathy': []
    } 

    for i in range(len(df)):
        seq = df.iloc[i]['Sequence']
        # create an object localCIDER uses
        seq_obj = SequenceParameters(seq)
        descriptor_dict['Fraction Charged Residues'].append(seq_obj.get_FCR())
        descriptor_dict['Net Charge Per Residue'].append(seq_obj.get_NCPR(pH=None))
        descriptor_dict['Fraction Negatively Charged'].append(seq_obj.get_fraction_negative())
        descriptor_dict['Fraction Positively Charged'].append(seq_obj.get_fraction_positive())
        descriptor_dict['Fraction Expanding'].append(seq_obj.get_fraction_expanding(pH=None))
        descriptor_dict['Disorder Promoting Residues'].append(seq_obj.get_fraction_disorder_promoting())
        descriptor_dict['Kappa'].append(seq_obj.get_kappa())
        descriptor_dict['Omega'].append(seq_obj.get_Omega())
        descriptor_dict['Mean Hydropathy'].append(seq_obj.get_mean_hydropathy())

    descriptor_df = pd.DataFrame(data=descriptor_dict)
    combined_df = pd.concat([df, descriptor_df], axis=1)
    
    return combined_df