"""
The permit predictor is an example application using a trained model to predict
the confidence that a building permit is issued
"""

import numpy as np
from data_io import restore_object
from constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PCA_PICKLE_FILE, ENCODER_PICKLE_FILE, MODEL_PICKLE_FILE


def main():
    print("""\n                  *%%.  %%%#,
             .%%%%%%%,  %%%%(  %%
          .%%%%/    %%  /%%%(  %%  /.
        .%%%%*      *%/  #%%(  %%  /%%.
       %%%%(         %%*  .%(  %%  *%%%%
      %%%%%%%.       *%%%      %%     %%%
     %%%%%  %%%%.     ,%%%%%   %%  (/   %%
    *%%%#   #          ,%%%%(  %%  (%%*  #*
    %%%#                 %%%(  %%  ,%%%#  %
    %%%,             #%*   %(  %%    #%%/
    %%%%%            %%%       %%  /  .%%,
    %%%%%            %%%  /%(  %%  (%  #%%
    *%%%%            %%%  /%(  %%  (%/  %%*
     %%%(           #%%%  /%(  %%  (%%  %%
      %%%%%%%%%%%%%%%%%%  /%(  %%  (%%. (
       %%%%%%%       %%%  /%(  %%  (%%(
        .%%%%%%      %%%  /%(  %%  (%%.
          .%%%%%     #%%  /%(  %%  (.
             .%%,    %%%  /%(  %%          """)
    print('\nWelcome to the Seattle Building Permit Predictor')
    print('Please enter the permit details below')

    calc_confidence()


def calc_confidence():
    """ Calculates the confidence that a building permit is issued based on the
    feature values entered by the user """
    permit = np.array([[
        input('Permit class: '),
        input('Permit type: '),
        input('Type mapped: '),
        input('Housing Units: '),
        input('Year: '),
        input('Neighborhood: ')
    ]])

    # Restore required objects
    encoder = restore_object(ENCODER_PICKLE_FILE)
    pca = restore_object(PCA_PICKLE_FILE)
    clf = restore_object(MODEL_PICKLE_FILE)

    # Transform permit application
    permit = np.concatenate((encoder.transform(permit[:, CATEGORICAL_FEATURES]).toarray(),
                             permit[:, NUMERIC_FEATURES].astype(float)), axis=1)
    permit = pca.transform(permit)

    # Predict confidence that the permit will be issued
    confidence_issued = clf.predict_proba(permit)[0][1]

    print('The permit will be issued with a confidence of: ' + str(confidence_issued))

    if input("Would you want to predict another permit? (y/n)") == 'y':
        calc_confidence()


if __name__ == '__main__':
    main()
