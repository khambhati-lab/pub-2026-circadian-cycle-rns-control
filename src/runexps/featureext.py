import sys
import os
import multiprocessing
import datetime
import logging

logging.basicConfig(filename=f"logs/log_{datetime.datetime.now()}.txt", level=logging.DEBUG)
logging.captureWarnings(True)
logging.info('Running from src/runexps/featureext')


from src.dataproc import rnsfeatureext

'''
Main routine to execute several feature extraction tasks in parallel. It takes the patient's npcode and machine
on which to execute the feature extraction task and launches jobs on the machine. Features are extracted from 
RNS ECoG data.

Things to check before executing the function:
- list of patients 
- machine name
- number of processes to run parallely
- feature extraction function that is executed in each job

Date, Author: --, Krishnakant V. Saboo
'''

if __name__ == "__main__":

    print('In main.')

    machine = 'cluster' # 'local' or 'cluster'
    patient_list = ['NPE%04d'%(ii+1) for ii in range(140)]

    args_list = [{'machine':machine, 'npcode':patient_id} for patient_id in patient_list]

    pool = multiprocessing.Pool(processes=20)
    recresult_list = pool.map(rnsfeatureext.main_featext, args_list)
    
    pool.close()
    print('Pool computation started')
    pool.join()
    print('Pool computation ended')

    
