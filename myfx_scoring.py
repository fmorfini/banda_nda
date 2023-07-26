# francesca morfini (f.morfini.work@gmail.com; morfini.f@northeastern.edu) 
# keara greene (k.greene@northeastern.edu)
# susan whitfield-gabrieli lab
# northeastern university, Boston, US

## =========================================================== ##
# scoring and checks for BANDA NDA data
## =========================================================== ##

# INFO
# this set of functions scores clinical questionnaires acquired under the HCP for disease BANDA project
# all functions are independent from one another

# each function:
# - summarizes info regarding original questionnaire and scoring, NDA translation of questionnaire and how to score accordingly,
# - calculates subscales and total scores
# - runs basic reality checks (min max of scored subscales are within theoretical range, and that all items have been used)
# - drops item-level columns as specified by {item_level = 'drop'} when calling the fx

# inputs: item level data as downloaded from the NDA and after dropping columns that contains all NaNs
# outputs: original dataframe minus item-level columns, plus 3 columns for each subscale/tot {one with theoretical number of items; one with nan count; one with scores}
# options: 
# - df : pandas dataframe with original item-level data as downloaded from NDA, with the following set up as a multiindex ['src_subject_id', 'visit','respondent','subjectkey','sex','interview_date','interview_age']. note, this can be one dataframe with all data from NDA concatenated together
# - item_level: 'drop' or none, flags whether to drop or keep item-level columns after scoring
# - grab_item_count: 'yes' or none, flags whether to save a column reporting theoretical number of items that make up a subscale (one for every questionnaire-subscale)

# this file includes one function to score each of the following:
# bisbas
# cbcl (scoring info is proprietary so not including here)
# chapman handedness
# cssrs
# ksads01 (technically not a scoring, here just creating grouping based on specific aims of our study)
# ksadsp201 (technically not a scoring, currently dropping this info)
# masq
# mfq
# nffi
# rbqa
# rcads
# rmbi
# shaps
# stai (work in progress)
# strain (scoring info is proprietary so not including here)
# tanner
# wasi
# PENN and NIH tasks (not scoring, just relabelling columns)

## ================================================= ##
# more info about the BANDA dataset, dataset manual, scoring, crosswalk etc can be found at
# https://nda.nih.gov/edit_collection.html?id=3037
# https://www.humanconnectome.org/storage/app/media/documentation/BANDA1.0/BANDA_Release_1.0_Manual.pdf
# https://www.humanconnectome.org/storage/app/media/documentation/BANDA1.0/BANDA1.0_Crosswalk.csv

## ================================================= ##
#conda activate mri
import pandas as pd
import numpy as np

## ================================================= ##
def score_bisbas(df, item_level, grab_item_count):
    '''
    questionnaire: Behavioral Approach System/Behavioral Inhibition System (BIS/BAS) Scales 
    respondent: child
    4 subscales and no total score
    subscales: bis, bas drive, bas fun seeking, bas reward responsiveness
    total: none
    filler items: 1, 6, 11, 17
    reverse scored items: all items are reverse except 2 and 22
    notes from BANDA1.0_crosswalk.csv: all items (except 2 and 22) have already been reverse coded, and therefore will not need to be reverse scored in this script
    questionnaire and scoring: https://local.psy.miami.edu/people/faculty/ccarver/availbale-self-report-instruments/bisbas-scales/
    NDA data dictionary: https://nda.nih.gov/data_structure.html?short_name=bisbas01
    NDA values (reversed items): 4= Very true for me; 3= Somewhat true for me; 2= Somewhat false for me; 1= Very false for me
    original questionnaire values (and NDA non-reversed items 2 & 22): 1 = very true for me; 2 = somewhat true for me ; 3 = somewhat false for me; 4 = very false for me
    note: NDA had already released scored subscales ['bissc_total', 'bas_drive', 'bas_fs', 'bas_rr']. here dropping and recalculate
    scores interpretation: "A higher level of BAS activity is defined by more erratic, risk-taking behavior and positive emotion, whereas higher levels of BIS activity is an indicator of a more risk-averse behavior and negative emotion." (from https://en.wikiversity.org/wiki/Behavioral_Inhibition_and_Behavioral_Activation_System_(BIS/BAS)_Scales)
    '''

    # questionnaire info
    name = 'bisbas'
    filename =f'{name}01'
    n_items = 24
    reverse_items = [ ] #NDA already reverse coded
    fillers_items = [1, 6, 11, 17]
    rules = {'1' : 1, '2' : 2, '3' : 3, '4': 4 ,'NaN': np.nan}

    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 
    cols_items = [f'{name}{i}' for i in range(1,n_items+1)]  # returns list of column names
    df[cols_items] = df[cols_items].replace(rules) # re-codes 

    orig_subscales = ['bissc_total', 'bas_drive', 'bas_fs', 'bas_rr']
    df = df.drop(columns = orig_subscales)
    
    # note: no need to reverse scoring since NDA data had already been reversed

    # subscales info
    # subscalename = [['subscalename', ['items numbers'], ['placeholder for cols corresponding to items'], 'n items in subscale', 'min score' (added later), 'max score' (added later)]]
    subscale_1 = ['bis', [2, 8, 13, 16, 19, 22, 24], [ ], 7, [], []] 
    subscale_2 = ['bas_drive', [3, 9, 12, 21], [ ], 4, [], []]
    subscale_3 = ['bas_fun_seeking', [5, 10, 15, 20], [ ], 4, [], []]
    subscale_4 = ['bas_reward_respon', [4, 7, 14, 18, 23], [ ], 5, [], []]
       
    subscales = [subscale_1, subscale_2, subscale_3, subscale_4]
        
    # scoring and reality checks
    for scale in subscales:
        scale[4] = scale[3] * 1  # add min
        scale[5] = scale[3] * 4  # add max
        scale[2] = [f'{name}{item}' for item in scale[1]] # returns list of column names corresponding to only the items that make up the subscale
        df[f'{name}_{scale[0]}_nan_count'] = df[scale[2]].isna().sum(axis=1) # count and store number of item-level nan answers        
        if grab_item_count == 'yes':
            df[f'{name}_{scale[0]}_items_count'] = len(scale[2]) # store theoretical count of items making up the subscale
        df[f'{name}_{scale[0]}'] = df[scale[2]].sum(axis=1, min_count = 1) # score subscale
    
        # calculating min/max i) subscale value and ii) the min/max possible score they would have gotten if they had answered the quesionts they had as nans
        scale_min = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 1 )).min()
        scale_max = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 4 )).max()

        #scale[4] and scale[5] indicate the theoretical ranges; scale_min|_max are actual min and max of data
        assert (scale_min >= scale[4]) and (scale_max <= scale[5]), f'Watch out: values for subscale {name}_{scale[0]} are outside of theoretical range for some participants. Values should be between {scale[4]} and {scale[5]} but are between {scale_min} and {scale_max} instead'

        # check that overall number of items grabbed is correct
        n_items_used = [scale[3] for scale in subscales]

        assert sum(n_items_used) == n_items - len(fillers_items), 'The number of items used to calculate the subscales and total score is either more or less than what was expected'

    # remove item-level columns from original df
    if item_level == 'drop':
        df = df.drop(columns = cols_items)
    
    print(f'Scored {name}')
    return df

# ================================================= #
# score_cbcl(df, item_level, grab_item_count)
# scoring info is proprietary so not including here

## ================================================= ##
def score_chaphand(df, item_level, grab_item_count):
    '''
    questionnaire: Chapman Handedness
    respondent: child
    0 subscales and 1 total score
    subscales: none
    total: sum of all items
    fillers items: none
    reverse scored items: none
    notes from BANDA1.0_Crosswalk.csv file: Recoded 1 to 0, 3 to 1
    questionnaire & scoring: (Chapman and Chapman, 1987) See /work/swglab/data/BANDA/NDA/BANDAImgManifestBeh/info_questionnaires_manual_crosswalk/CHAPMAN HANDEDNESS.pdf
    NDA data dictionary: https://nda.nih.gov/data_structure.html?short_name=chaphand01
    NDA values:  0=Left; 1= Either Hand; 2=Right
    original questionnaire values (from Table 1 in Chapman & Chapman 1987): 1=Left; 3= Either Hand; 2=Right
    scoring: 3=Left; 2= Either Hand; 1=Right
    scores interpretation: 13-17 = right-handed, 33-39 left handed, 18-32 ambilateral.
    '''

    # questionnaire info
    name = 'chaphand'
    filename =f'{name}01'
    n_items = 13
    reverse_items = []
    fillers_items = []
    rules = {0 : 3, 1 : 2, 2 : 1, 'NaN': np.nan}

    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 
    
    cols_items = df.columns[df.columns.str.startswith('hu0')]  # returns list of column names
    df[cols_items] = df[cols_items].replace(rules) # re-codes 

    # subscales info
    # subscalename = [['subscalename', ['items numbers'], ['placeholder for cols corresponding to items'], 'n items in subscale', 'min score' (added later), 'max score' (added later)]]
    tot = ['tot', cols_items, [], n_items, [], []]
   
    subscales = [tot]
    
    # scoring and reality checks
    for scale in subscales:
        scale[4] = scale[3] * 1  # add min
        scale[5] = scale[3] * 3  # add max
        scale[2] = scale[1] # keep for consistency with other functions, but here no need since column names were hardcoded and same as column items
        df[f'{name}_{scale[0]}_nan_count'] = df[scale[2]].isna().sum(axis=1) # count and store number of item-level nan answers

        if grab_item_count == 'yes': 
            df[f'{name}_{scale[0]}_items_count'] = len(scale[2]) # store theoretical count of items making up the subscale
        df[f'{name}_{scale[0]}'] = df[scale[2]].sum(axis=1, min_count = 1) # subscale score
    
        # calculating min/max i) subscale value and ii) the min/max possible score they would have gotten if they had answered the quesionts they had as nans
        scale_min = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 1 )).min()
        scale_max = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 3 )).max()

        #scale[4] and scale[5] indicate the theoretical ranges; scale_min|_max are actual min and max of data     
        assert (scale_min >= scale[4]) and (scale_max <= scale[5]), f'Watch out: values for subscale {name}_{scale[0]} are outside of theoretical range for some participants. values should be between {scale[4]} and {scale[5]} but are between {scale_min} and {scale_max} instead'

        # check that overall number of items grabbed is correct
        n_items_used = [scale[3] for scale in subscales]
        assert sum(n_items_used) == n_items - len(fillers_items), 'The number of items used to calculate the subscales and total score is either more or less than what was expected' 
    
    # remove item-level columns from original df
    if item_level == 'drop':
        df = df.drop(columns = cols_items)
    
    print(f'scored {name}')
    return df

## ================================================= ##
def score_cssrs(df, item_level, grab_item_count):
    '''
    questionnaire: Columbia-Suicide Severity Rating Scale (C-SSRS)
    respondent: child
    5 subscales and no total score
    subscales: Suicidal Ideation; Suicidal Behavior; Medical Damage for Attempt; Potential Lethality  *Note: did not include Intensity of ideation subscale because all intensity questions not represented in the data
    total: no
    fillers items: none
    reverse scored items: none
    notes from BANDA1.0_Crosswalk.csv file: certain questions changed 0s to 2s
    questionnaire: https://cssrs.columbia.edu/wp-content/uploads/C-SSRS1-14-09-BaselineScreening.pdf
    scoring: https://dphhs.mt.gov/assets/suicideprevention/basicscoringguideforclinicians.pdf
    NDA data dictionary: https://nda.nih.gov/data_structure.html?short_name=cssrs01
    notes on values: highly dependent on question, so best to look at NDA data dictionary and original questionnaire
    scoring: Suicidal Ideation (Highest Level Endorsed 1-5); Suicidal Behavior (present during time period) Y/N; Medical Damage for Attempt (0-5); Potential Lethality (if medical damage = 0) (0-2)    
    values and scoring for Suicidal Ideation subscale: individual items are 0=No; 1=Yes and suicidal Ideation subscale is most severe ideation endorsed: 1=Wish to be dead: Past Month; 2=Non-specific suicidal thoughts: Past Month; 3=Active suicidal ideation with any methods (no plan) without intent to act: Past Month; 4=Active suicidal ideation with some intent to act, without specific plan: Past Month; 5=Active suicidal ideation with specific plan and intent: Past Month
    values for Suicidal Behavior subscale: individual items for past 3 months 1=Yes; 2=No and suicidal behavior subscale is 1=Yes (ANY individual items endorsed); 0=No (no items endorsed)
    values for Medical Damage for Attempt subscale: if there was an attempt, 0=No physical damage or very minor physical damage (e.g., surface scratches).; 1=Minor physical damage (e.g., lethargic speech first-degree burns mild bleeding sprains).; 2=Moderate physical damage medical attention needed (e.g., conscious but sleepy, somewhat responsive second-degree burns bleeding of major vessel).; 3=Moderately severe physical damage medical hospitalization and likely intensive care required (e.g., comatose with reflexes intact third-degree burns less than 20% of body extensive blood loss but can recover major fractures).; 4=Severe physical damage medical hospitalization with intensive care required (e.g., comatose without reflexes third-degree burns over 20% of body extensive blood loss with unstable vital signs major damage to a vital area); 5=Death
    values for Potential Lethality subscale: if there was an attempt AND actual damage=0, then 0=Behavior not likely to result in injury; 1=Behavior likely to result in injury but not likely to cause death; 2=Behavior likely to result in death despite available medical care

    # subscales interpretation (as per this function's outcomes)
    suicidal ideation:       0 means no ideation, higher values mean more severe ideation
    suicidal behavior:       0 means no behavior present, 1 means had some sort of suicidal behavior
    suicidal medical damage: 0 means no damage/very minor damage, higher values means more damage
    suicidal pot. lethality: 0 means not likely to result in injury, higher values mean more potential for lethality
    '''

    # questionnaire info
    name = 'cssrs'
    filename =f'{name}01'

    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 

    ###########################
    ### subscale suicideal ideation
    cols_ideation = ['css_sim1', 'css_sim2', 'css_sim3', 'css_sim4', 'css_sim5']
    #reverting order will help grab index of most severe type of ideation in the next step, since .idxmax() returns only the index of the first match (which is 1 if cols are listed 1to5, but which is 5 if cols are listed from 5to1)
    cols_ideation.reverse() 
    # next, if sum of cols_ideation is ==0, assign 0 to cssrs_ideation, otherwise assign index of most severe column (where css_sim1 is least and css_sim5 most severe)
    df[f'{name}_ideation'] = np.where(df[cols_ideation].sum(axis= 'columns') ==0, 0, df[cols_ideation].idxmax(axis="columns").str[-1:].astype(float)) 

    assert df[f'{name}_ideation'].min() >= 0 and df[f'{name}_ideation'].max() <= 5, f'Watch out: {name}_ideation outside the 0-5 value ranges for some participants'

    ###########################
    ### subscale suicidal behavior
    cols_behave = ['sbaap3m', 'sbiap3m', 'sbasiap3m', 'sbpabp3m']
    rules_behave = {1 : 1, 2 : 0, 'NaN': np.nan}
    df[cols_behave] = df[cols_behave].replace(rules_behave) # re-codes 

    conditions = [df[cols_behave].any(axis = 'columns'), #assign 1 if any of selected colums has a 1
                  (df[cols_behave].sum(axis = 'columns', min_count = 1) == 0), #assign 0 if reporting 0s
                  df[cols_behave].isna().all(axis = 'columns') # assign nan if all questions were nan
                 ]
    newvalues = [1, 0, np.nan]
    df[f'{name}_behave'] = np.select(conditions, newvalues, default=np.nan) # assign nan to all other cases

    assert df[f'{name}_behave'].min() >= 0 and df[f'{name}_behave'].max() <= 1, f'Watch out: {name}_behave outside the 0-1 value ranges for some participants'

    ###########################
    ### subcale medical damage
    df[f'{name}_damage'] = df['actlthl1']
    assert df[f'{name}_damage'].min() >= 0 and df[f'{name}_damage'].max() <= 5, f'Watch out: {name}_behave outside the 0-5 value ranges for some participants'

    ###########################
    ### subcale potential lethality
    df[f'{name}_lethality'] = df['potlthl1']
    assert df[f'{name}_lethality'].min() >= 0 and df[f'{name}_lethality'].max() <= 2, f'Watch out: {name}_behave outside the 0-2 value ranges for some participants'

    ###########################
    # remove item-level columns from original df
    if item_level == 'drop':
        cols_items = ['si1l', 'si2l', 'si3l', 'si4l', 'si5l', 'sb2l', 'sb3l', 'sb4l',
           'sb5l', 'sb6l', 'rctattdt', 'actlthl1', 'potlthl1', 'lthldt',
           'actlthl2', 'potlthl2', 'initatdt', 'actlthl3', 'potlthl3',
           'cssrs_base_06', 'cssrs_base_06a', 'css_sim1', 'css_sim2',
           'css_sim3', 'css_sim4', 'css_sim5', 'iilmsi', 'iirmsi', 'sbaap3m',
           'sbtnal', 'sbtnap3m', 'sbnssibl', 'sbnssibp3m', 'sbsibiul',
           'sbsibiup3m', 'sbiap3m', 'sbiatnal', 'sbiatnap3m', 'sbpabp3m',
           'sbasiap3m', 'sbasiatnal', 'sbasiatnap3m',
           'sb_present_interview_3m']
        df = df.drop(columns = cols_items) #dropping colums

    # note: the option grab_item_count has no meaning as per questionnaire instructions, so not calculating it
    print(f'scored {name}')
    return df
                  
## ================================================= ##
def score_ksads01(df, item_level, grab_item_count):
    '''
    questionnaire: Kiddie Schedule for Affective Disorders and Schizophrenia 
    respondent: child
    filler items: none
    reverse scored items: none
        
    notes from BANDA1.0_Crosswalk.csv file: some diagnosis have been recoded from 3 to 4 and 4 to 3, however since here we are interested in both 3s and 4s we are not re-coding this
    questionnaire: https://www.pediatricbipolar.pitt.edu/sites/default/files/KSADS_DSM_5_SCREEN_Final.pdf
    questionnaire: https://www.pediatricbipolar.pitt.edu/sites/default/files/KSADS_DSM_5_SummaryChecklists_Final%20%281%29.pdf
    questionnaire: https://insideoutinstitute.org.au/assets/kiddie%20sads%20present%20and%20lifetime%20version%20k%20sads%20pl.pdf
    scoring: done directly by clinician - here only applying study-specific rules to generate diagnostic grouping
    NDA data dictionary: https://nda.nih.gov/data_structure.html?short_name=ksads_diagnoses01
    NDA values for the diagnosis considered (note, this is not consistent across diagnosis): 0=Incomplete information; 1= Not present ; 2= Sub-clinical ; 3= Present but just meets diagnostic criteria ; 4= Present severe

    # STUDY SPECIFIC INFO ## for our group only
    notes from BANDA1.0_Crosswalk.csv file: some diagnosis have been recoded from 3 to 4 and 4 to 3, however since here we are interested in both 3s and 4s we are not re-coding this
    # here we are creating a 4-way grouping variable: healthy, depressed, anxious, comorbid (i.e., dep and anx)
    # considering participants with either 3 (present) or 4 (severe), based on the following:
    depressive= anyone with a current diagnosis of any of the following: major depressive disorder; dysthymia; depression not oherwise specified
    anxious= anyone with a current diagnosis of any of the following: generalized anxiety disorder; social phobia; specific phobia; separaion anxiety; panic disorder; agoraphobia
    comorbid= anyone with at least one depressive and one anxious diagnosis
    healthy= anyone without any depressive nor anxious diagnosis
    '''

    # questionnaire info
    name = 'ksads'
    filename =f'{name}_diagnoses01'
    rules = {4 : 1, 3 : 1, 2 : 0, 1 : 0, 0 : 0, 'NaN': np.nan, 999.0: np.nan} # grab only those with presence even if just above threshod (3) or severe (4)

    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 
    
    dep = ['mddcurrent','dysthymiacurrent','depnoscurrent'] # as per NDA dictionary, these could be considered corresponding to dsm5 too
    anx = ['gadcurrent', 'panicdisordercurrent', 'agoraphobiacurrent', 'separationcurrent', 'socialphobiacurrent', 'simplephobiacurrent']
    
    cols_all = ['depressive_disorder_nos', 'mania', 'hypomania', 'bipolar_nos','bipolar_i', 'bipolar_ii', 'schizoaffective_disorder_mania',
                'schizophrenia', 'schizophreniform_disorder','brief_reactive_psychosis', 'avoidant_disorder_childhood',
                'overanxious_disorder', 'post_traumatic_stress_disorder','acute_stress_disorder', 'adj_disorder_wanxious_mood', 'enuresis',
                'encopresis', 'attention_deficit_disorder', 'adj_disorder_dist_conduct','adj_dis_mixed_mood_conduct', 'tourettes',
                'chronic_motor_voc_tic_disorder', 'transient_tic_disorder','alcohol_abuse', 'mental_retardation', 'other_psychiatric_disorder',
                'no_psychiatric_disorder', 'relationship', 'kssp_q2_p', 'kssp_q8_p','kssp_q10_p', 'kssp_q11_p', 'mddpast', 'mddcurrent', 'dysthymiapast',
                'dysthymiacurrent', 'depnospast', 'depnoscurrent', 'cyclothymiapast','cyclothymiacurrent', 'bipolarnospast', 'bipolarnoscurrent',
                'bipolaripast', 'bipolaricurrent', 'bipolariipast', 'bipolariicurrent','panicdisorderpast', 'panicdisordercurrent', 'separationpast',
                'separationcurrent', 'simplephobiapast', 'simplephobiacurrent','socialphobiapast', 'socialphobiacurrent', 'agoraphobiapast',
                'agoraphobiacurrent', 'gadpast', 'gadcurrent', 'ocdpast', 'ocdcurrent','ptsdpast', 'ptsdcurrent', 'adhdpast', 'adhdcurrent', 'oddpast',
                'oddcurrent', 'conductpast', 'conductcurrent', 'anorexiapast','anorexiacurrent', 'bulimiapast', 'bulimiacurrent', 'alcoholabusepast',
                'alcoholdependencepast', 'alcoholdependencecurrent','substanceabusepast', 'substanceabusecurrent','substancedependencepast','substancedependencecurrent']

    cols_items = dep + anx # cols used to generate diagnostic grouping
    df[cols_all] = df[cols_all].replace(rules) # re-codes

    # generate custom grouping based on chosen diagnoses
    conditions =[  df[dep].T.any()  & ~(df[anx].T.any()), #depressed: grab individuals with any dep diagnosis, but none of the anxious
                 ~(df[dep].T.any()) &   df[anx].T.any(),  #anxious: no dep, any anx
                   df[dep].T.any()  &   df[anx].T.any(),  #comorbid: any dep AND any anx
                 ~(df[dep].T.any()) & ~(df[anx].T.any()) & ~(df[dep+anx].isna().all(axis='columns')), #hc: no dep, no anx
                 df[dep+anx].isna().all(axis='columns')   #nan
                ]
    newvalues = ["depressed","anxious","comorbid","hc",np.nan]
    df['ksads_group'] = np.select(conditions, newvalues, default=np.nan)
    
    # remove item-level columns
    if item_level == 'drop':
        cols_drop =[i for i in cols_all if i not in cols_items] #grab all cols except those used to generate the grouping (I want to keep for later reporting dis-aggregated stats)
        df = df.drop(columns = cols_drop) #dropping colums
        df = df.rename({col:f'{name}_'+ col for col in cols_all if col not in cols_drop},axis=1) # renaming surviving cols to find more easily later
    else:
        df = df.rename({col:f'{name}_'+ col for col in cols_all},axis=1) # renaming cols to find more easily later

     # note: the option grab_item_count has no meaning as per questionnaire instructions, so not calculating it        
    print(f'scored {name}')
    return df

## ================================================= ##
def score_ksadsp201(df, item_level, grab_item_count):
    '''
    currently just dropping all of these
    '''
    
    name = 'ksadsp201'
    filename ='ksads_diagnosesp201'
    rules = {4 : 1, 3 : 1, 2 : 0, 1 : 0, 0 : 0, 'NaN': np.nan} # grab only those with presence even if just above threshod (3) or severe (4)
    
    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 

    #all columns
    cols_all = ['hypomaniapast', 'schizophreniapast', 'schizoaffectivepast','avoidantchildhoodpast', 'acutestresspast', 'adjustmentanxiouspast',
       'adjustmentconductpast', 'adjustmentmixedpast', 'tourettespast','chronicticpast', 'transientticpast', 'mentalretardationpast',
       'otherpsychiatricdisorderpast', 'briefreactivepsychosispast','schziophreniformpast', 'adjdiswdepcurr', 'adjdiswdeppast', 'sldc194',
       'sldc232', 'depressotherspeccurrent_dsm5', 'depressotherspecpast_dsm5','disruptmooddysregpast_dsm5', 'premenstrualdysphcurrent_dsm5',
       'premenstrualdysphpast_dsm5', 'bipolarotherspeccurrent_dsm5','bipolarotherspecpast_dsm5', 'schizophreniapast_dsm5',
       'schizoaffectivecurrent_dsm5', 'schizoaffectivepast_dsm5','schizophreniformpast_dsm5', 'briefreactivepsychpast_dsm5',
       'schizophreniaothspeccurr_dsm5', 'schizophreniaothspecpast_dsm5','schizophreniaunspeccurr_dsm5', 'schizophreniaunspecpast_dsm5',
       'illnessanxietycurrent_dsm5', 'illnessanxietypast_dsm5','otherspecanxietycurrent_dsm5', 'otherspecanxietypast_dsm5',
       'unspecanxietycurrent_dsm5', 'unspecanxietypast_dsm5','selectivemutismcurrent_dsm5', 'selectivemutismpast_dsm5',
       'hoardingcurrent_dsm5', 'hoardingpast_dsm5','trichotillomaniacurrent_dsm5', 'trichotillomaniapast_dsm5',
       'excoriationcurrent_dsm5', 'excoriationpast_dsm5','bodydysmorphiccurrent_dsm5', 'bodydysmorphicpast_dsm5',
       'ocdotherspeccurrent_dsm5', 'ocdotherspecpast_dsm5','ocdunspeccurrent_dsm5', 'ocdunspecpast_dsm5', 'acutestresspast_dsm5',
       'bingeeatingcurrent_dsm5', 'bingeeatingpast_dsm5','adhdotherspeccurrent_dsm5', 'adhdotherspecpast_dsm5',
       'adhdunspeccurrent_dsm5', 'adhdunspecpast_dsm5','intermittentexplosivecurr_dsm5', 'intermittentexplosivepast_dsm5',
       'pyromaniacurrent_dsm5', 'pyromaniapast_dsm5','kleptomaniacurrent_dsm5', 'kleptomaniapast_dsm5',
       'disruptiveotherspeccurr_dsm5', 'disruptiveotherspecpast_dsm5','disruptiveunspeccurrent_dsm5', 'disruptiveunspecpast_dsm5',
       'tourettespast_dsm5', 'chronicticpast_dsm5', 'transientticpast_dsm5','alcoholusepast_dsm5', 'alcoholunspeccurrent_dsm5',
       'alcoholunspecpast_dsm5', 'substanceusecurrent_dsm5','substanceusepast_dsm5']
    
    df[cols_all] = df[cols_all].replace(rules) # re-codes

    # remove item-level columns
    if item_level == 'drop':
        df = df.drop(columns = cols_all) #dropping colums
    else:
        df = df.rename({col:f'{name}_'+ col for col in cols_all},axis=1) # renaming cols to find more easily later

    # note: the option grab_item_count has no meaning as per questionnaire instructions, so not calculating it
    print(f'scored {name}')
    return df

## ================================================= ##
def score_masq(df, item_level, grab_item_count):
    '''
    questionnaire: Mood and Anxiety Symptom Questionnaire 
    respondent: parent self-report
    4 subscales and no total score
    subscales: general distress anxious symptoms, anxious arousal, general distress depressive symptoms, anhedonic depression
    total: none
    
    fillers items: none
    
    reverse scored items: 3, 7, 10, 15, 22, 27, 39, 43, 47, 49, 53, 56, 58, 60
    Notes from BANDA1.0_Crosswalk.csv file: 43 is already reverse scored; essentially 3 groups (values 1-5; values 0-4; and strings only)
    questionnaire: https://github.com/seschneck/arc_scoring_code/raw/main/MASQ/MASQ62item.pdf
    scoring: https://github.com/seschneck/arc_scoring_code/raw/main/MASQ/ScoringNotes.pdf
    Extra info: https://arc.psych.wisc.edu/self-report/mood-and-anxiety-symptom-questionnaire-masq/
    NDA data dictionary: https://nda.nih.gov/data_structure.html?short_name=masq01
    NDA values: Highly depends on the question, but essentially 3 groups (values 1-5, values 0-4, and strings only)
    NDA 1-5: 1=Not at all or very slightly; 2=Mildly; 3=Moderately; 4=Quite a bit; 5=Very much
    NDA 0-4: 0=Not at all; 1=A little bit; 2=Moderately; 3=Quite a bit; 4=Extremely
    NDA strings only: not at all; a little bit; moderately; quite a bit; extremely
    Original questionnaire values: 1=Very slightly or not at all; 2=A little; 3=Moderately; 4=Quite a bit; 5=Extremely
    
    Interpretation of Scores: higher score indicates higher level of behavior problem of subscale
    '''

    # questionnaire info
    name = 'masq'
    filename =f'{name}01'
    n_items = 62
    reverse_items = [3, 7, 10, 15, 22, 27, 39, 47, 49, 53, 56, 58, 60] #does not include 43 because already reverse coded in NDA
    fillers_items = []
    rules_ques = {1 : 1, 2 : 2, 3 : 3, 4 : 4, 5 : 5, 'NaN': np.nan} # for items in NDA that follow original questionnaire values
    rules_plus1 = {0 : 1, 1 : 2, 2 : 3, 3 : 4, 4 : 5, 'NaN': np.nan} # for items in NDA that were put on 0 to 4 scale
    rules_str = {'not at all' : 1, 'a little bit' : 2, 'moderately' : 3, 'quite a bit' : 4, 'extremely' : 5, 'NaN': np.nan} # for items in NDA that were made to be strings

    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 
    
    # column names of all columns, listed in order from first to last questionnaire's item
    cols_items = ['wn12', 'masq_02', 'poms038', 'afraid', 'idas_40', 'shakyhnd', 'masq_09', 'masq_diar', 'masq_04', 'masq_06', 
                  'masq_nerv', 'idas_8', 'masq_21', 'masq_un', 'idas_10', 'lumpthrt', 'masq_faint', 'masq90_q86', 'masq_20', 
                  'bsil23', 'stai25', 'masq_11', 'masq_12', 'sweaty', 'impac_q9', 'masq_onedge', 'masq_d30_26', 'masq_15', 'masq_23', 
                  'masq_30', 'masq90_q23', 'relax', 'scl14', 'atq_b_10d', 'masq_05', 'masq_10', 'masq_08', 'fatexp45', 'masq_22', 
                  'masq_18', 'masq90_q10', 'masq_chok', 'idas_27', 'twitch', 'masq_17', 'masq_dm', 'idas_59', 'masq_afr2die', 'masq_14', 
                  'masq90_q71', 'masq90_q35', 'masq_27', 'masq_16', 'bsil33', 'bsil38', 'idas_50', 'pclc_9', 'masq90_q32', 'masq_24', 
                  'masq_29', 'baseline_h_017', 'frequri']

    #column items in NDA that follow original questionnaire values
    cols_items_ques = ['wn12', 'idas_40', 'idas_8', 'idas_10', 'stai25', 'impac_q9', 'atq_b_10d', 'fatexp45', 
                       'idas_27', 'idas_59', 'idas_50', 'pclc_9']

    #column items in NDA that were put on 0 to 4 scale
    cols_items_plus1 = ['masq_02', 'poms038', 'afraid', 'shakyhnd', 'masq_09', 'masq_diar', 'masq_04', 'masq_06', 'masq_nerv', 'masq_21', 
                        'masq_un', 'lumpthrt', 'masq_faint', 'masq90_q86', 'masq_20', 'bsil23', 'masq_11', 'masq_12', 'sweaty','masq_onedge', 'masq_d30_26','masq_15', 
                        'masq_23', 'masq_30', 'masq90_q23', 'relax', 'masq_05', 'masq_10', 'masq_08', 'masq_22', 'masq_18', 'masq90_q10', 
                        'masq_chok', 'twitch', 'masq_17', 'masq_dm', 'masq_afr2die', 'masq_14', 'masq90_q71', 'masq90_q35', 'masq_27', 'masq_16',
                        'bsil33', 'bsil38', 'masq90_q32', 'masq_24', 'masq_29', 'baseline_h_017', 'frequri']

    #column items in NDA that were made to be strings
    cols_items_str = ['scl14']

    #re-code all variables
    df[cols_items_ques] = df[cols_items_ques].replace(rules_ques)    #items in NDA that follow original questionnaire values
    df[cols_items_plus1] = df[cols_items_plus1].replace(rules_plus1) #items in NDA that were put on 0 to 4 scale
    df[cols_items_str] = df[cols_items_str].replace(rules_str)  #items in NDA that were made to be strings

    # reverse scoring
    for i in reverse_items:
        grab_col = cols_items[i-1] # name of column corresponding to the i-th item
        df[grab_col] = 5 + 1 - df[grab_col] # 5 = max item value, 1 = constant, then subtracting from original un-reverted scores
        
    # subscales info
    # subscalename = [['subscalename', ['items numbers'], ['placeholder for cols corresponding to items'], 'n items in subscale', 'min score' (added later), 'max score' (added later)]]
    subscale_1 = ['gd_anxious_symptoms', [4, 8, 11, 14, 16, 20, 26, 32, 35, 55, 59], [], 11, [], []] # question numbers here refer to the i-th item (as reported in order in {cols_items} )
    subscale_2 = ['anxious_arousal', [2, 6, 13, 17, 19, 24, 28, 30, 37, 40, 42, 44, 46, 48, 52, 54, 62], [], 17, [], []]
    subscale_3 = ['gd_depressive_symptoms', [1, 5, 9, 12, 21, 23, 29, 31, 34, 36, 38, 45], [], 12, [], []]
    subscale_4 = ['anhedonic_depression', [3, 7, 10, 15, 18, 22, 25, 27, 33, 39, 41, 43, 47, 49, 50, 51, 53, 56, 57, 58, 60, 61], [], 22, [], []]
    tot = ['tot', list( range(1, n_items+1) ), [], n_items, [], []]
   
    subscales = [subscale_1, subscale_2, subscale_3, subscale_4, tot]
    
    # scoring and reality checks
    for scale in subscales:
        scale[4] = scale[3] * 1  # add min
        scale[5] = scale[3] * 5  # add max
        scale[2] = [cols_items[item-1] for item in scale[1]] # returns list of column names corresponding to only the items that make up the subscale. note, indexing [item -1] because the subscales refer to item number but python starts from 0
        df[f'{name}_{scale[0]}_nan_count'] = df[scale[2]].isna().sum(axis=1) # count and store number of item-level nan answers
        if grab_item_count == 'yes': 
            df[f'{name}_{scale[0]}_items_count'] = len(scale[2]) # store theoretical count of items making up the subscale
        df[f'{name}_{scale[0]}'] = df[scale[2]].sum(axis=1, min_count = 1) # score subscale

        # calculating min/max i) subscale value and ii) the min/max possible score they would have gotten if they had answered the quesionts they had as nans
        scale_min = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 1 )).min()
        scale_max = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 5 )).max()

        #scale[4] and scale[5] indicate the theoretical ranges; scale_min|_max are actual min and max of data     
        assert (scale_min >= scale[4]) and (scale_max <= scale[5]), f'Watch out: values for subscale {name}_{scale[0]} are outside of theoretical range for some participants. Values should be between {scale[4]} and {scale[5]} but are between {scale_min} and {scale_max} instead'

        # check that overall number of items grabbed is correct
        n_items_used = [scale[3] for scale in subscales]
        assert sum(n_items_used) == 2*n_items - len(fillers_items), 'The number of items used to calculate the subscales and total score is either more or less than what was expected' #note: doing 2* because subscales includes the total subscale too, which here corresponds to the sum of all items
        
    # remove item-level columns from original df
    if item_level == 'drop':
        df = df.drop(columns = cols_items)
        
    print(f'Scored {name}')
    return df

## ================================================= ##
def score_mfq(df, item_level, grab_item_count):
    '''
    questionnaire: Moods and Feelings questionnaire 
    respondent: child or parent; Child self-report, parent reporting about child
    0 subscales and 1 total score
    subscales: none
    total: sum of all items
    filler items: none
    reverse scored items: none
    notes from BANDA1.0_Crosswalk.csv file: all items plus 1
    questionnaire: https://psychiatry.duke.edu/research/research-programs-areas/assessment-intervention/developmental-epidemiology-instruments-0
    scoring: https://www.corc.uk.net/outcome-experience-measures/mood-and-feelings-questionnaire-mfq/
    NDA data dictionary: https://nda.nih.gov/data_structure.html?short_name=mfq01
    NDA values:  1=not true, 2=sometimes true, 3=true
    original questionnaire values: 0=not true, 1=sometimes true, 2=true, 
    scores interpretation: higher score indicates higher level of depressive symptoms
    '''

    # questionnaire info
    name = 'mfq'
    filename =f'{name}01'
    rules = {1 : 0,  2 : 1, 3 : 2, 'NaN': np.nan} 

    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 
    
    cols_items = df.columns[df.columns.str.contains(f'{name}')]
    df[cols_items] = df[cols_items].replace(rules) # re-codes 

    scale = ['tot', cols_items]

    # note: no need to reverse scoring since no items needs to be reversed as per the original questionnaire 
    
    # scoring
    df[f'{name}_{scale[0]}_nan_count'] = df[cols_items].isna().sum(axis=1) # count and store number of item-level nan answers
    # fixing nan counts just for Child who have one fewer item - so nan count is considering one extra nan than necessary
    for i in df.query(' respondent == "Child"').index:
        # df[f'{name}_count_nan'].loc[i] = df[f'{name}_count_nan'].loc[i] -1
        df.loc[i,f'{name}_{scale[0]}_nan_count'] = df.loc[i,f'{name}_{scale[0]}_nan_count'] -1
    df[f'{name}_{scale[0]}'] = df[scale[1]].sum(axis=1, min_count = 1) # score subscale  

    ## reality checks 
    version_1 = ['Parent', len(cols_items), 34] # this represents [respondent, how many cols grabbred, n items]
    version_2 = ['Child', len(cols_items) -1 , 33]
    versions = [version_1, version_2]
    
    # calculating min/max i) subscale value and ii) the min/max possible score they would have gotten if they had answered the quesionts they had as nans
    for v in versions:
        tmp = df.query(f' respondent == "{v[0]}"').copy()
        scale_min = (tmp[f'{name}_{scale[0]}'] + ( tmp[f'{name}_{scale[0]}_nan_count'] * 0 )).min()
        scale_max = (tmp[f'{name}_{scale[0]}'] + ( tmp[f'{name}_{scale[0]}_nan_count'] * 2 )).max()
        
        # check that overall number of items grabbed is correct
        assert v[1] == v[2], f'Watch out: there are more columns than those used'

        #v[2]*0 and v[2]*2 indicate the theoretical ranges as n_items x min_item_value; scale_min|_max are actual min and max of data     
        assert (scale_min >= v[2]*0) and (scale_max <= v[2]*2), f'Watch out: values of the {v[0]} version are outside of theoretical range. Min is {scale_min} instead of {scale[3]}; max is {scale_max} instead of {scale[4]}'

        if grab_item_count == 'yes': 
            df[f'{name}_{scale[0]}_items_count'] =  v[2] # store theoretical count of items making up the subscale
  
    # remove item-level columns from original df
    if item_level == 'drop':
        df = df.drop(columns = cols_items)

    print(f'Scored {name}')
    return df

## ================================================= ##
def score_nffi(df, item_level, grab_item_count):
    '''
    questionnaire: NEO-Five Factor Inventory [Neuroticism scale only]
    respondent: child
    1 subscales and no total score
    subscales: neuroticism
    total: none
    fillers items: none
    reverse scored items: 1, 16, 31, 46
    notes from BANDA1.0_Crosswalk.csv file: none
    questionnaire & scoring info: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3417236/pdf/main.pdf#page=3
    More scoring info: https://pages.uoregon.edu/gsaucier/NEO-FFI%20subcomponent%20norms%20and%20scoring.htm
    More info: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5639474/#:~:text=Measures,are%20reverse%2Dworded
    NDA data dictionary: https://nda.nih.gov/data_structure.html?short_name=nffi01
    NDA and original questionnaire values: 1=Strongly Disagree; 2=Disagree; 3=Neutral; 4=Agree; 5=Strongly agree
    scores interpretation: higher score means higher neuroticism
    '''

    # questionnaire info
    name = 'nffi'
    filename =f'{name}01'
    n_items = 12
    reverse_items = ['nffi_1', 'nffi_16', 'nffi_31', 'nffi_46'] 
    
    fillers_items = []
    rules = {1 : 1, 2 : 2, 3 : 3, 4 : 4, 5 : 5, 'NaN': np.nan}

    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 
    
    cols_items = ['nffi_1', 'nffi_6', 'nffi_11', 'nffi_16', 'nffi_21', 'nffi_26',
       'nffi_31', 'nffi_36', 'nffi_41', 'nffi_46', 'nffi_51', 'nffi_56']
    
    df[cols_items] = df[cols_items].replace(rules) # re-codes 

    # reverse scoring
    for grab_col in reverse_items:
        df[grab_col] = 5 + 1 - df[grab_col] # 5 = max item value, 1 = constant, then subtracting from original un-reverted scores

    # subscales info
    # subscalename = [['subscalename', ['items numbers'], ['placeholder for cols corresponding to items'], 'n items in subscale', 'min score' (added later), 'max score' (added later)]]
    subscale_1 = ['neuroticism', ['nffi_1', 'nffi_6', 'nffi_11', 'nffi_16', 'nffi_21', 'nffi_26','nffi_31', 'nffi_36', 'nffi_41', 'nffi_46', 'nffi_51', 'nffi_56'], [], 12, [], []] 
       
    subscales = [subscale_1]
    
    # scoring and reality checks
    for scale in subscales:
        scale[4] = scale[3] * 0  # add min
        scale[5] = scale[3] * 5  # add max
        scale[2] = scale[1] # keep for consistency with other functions, but here no need since column names were hardcoded and same as column items
        df[f'{name}_{scale[0]}_nan_count'] = df[scale[2]].isna().sum(axis=1) # count and store number of item-level nan answers
        if grab_item_count == 'yes': 
            df[f'{name}_{scale[0]}_items_count'] = len(scale[2]) # store theoretical count of items making up the subscale
        df[f'{name}_{scale[0]}'] = df[scale[2]].sum(axis=1, min_count = 1) # score subscale
    
        # calculating min/max i) subscale value and ii) the min/max possible score they would have gotten if they had answered the quesionts they had as nans
        scale_min = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 0 )).min()
        scale_max = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 5 )).max()

        #scale[4] and scale[5] indicate the theoretical ranges; scale_min|_max are actual min and max of data     
        assert (scale_min >= scale[4]) and (scale_max <= scale[5]), f'Watch out: values for subscale {name}_{scale[0]} are outside of theoretical range for some participants. values should be between {scale[4]} and {scale[5]} but are between {scale_min} and {scale_max} instead'

        # check that overall number of items grabbed is correct
        n_items_used = [scale[3] for scale in subscales]
        assert sum(n_items_used) == n_items - len(fillers_items), 'The number of items used to calculate the subscales and total score is either more or less than what was expected' 

    # remove item-level columns from original df
    if item_level == 'drop':
        df = df.drop(columns = cols_items)    
        
    print(f'scored {name}')
    return df

## ================================================= ##
def score_rbqa(df, item_level, grab_item_count):
    '''
    questionnaire: Risky Behavior questionnaire for Adolescents (RBQA)
    respondent: child
    0 subscales and 1 total score
    subscales: none
    total: sum of all items
    fillers items: none
    reverse scored items: none
    notes from BANDA1.0_Crosswalk.csv file: none
    questionnaire: https://cdasr.mclean.harvard.edu/wp-content/uploads/2017/09/Auerbach_2012_BRT.pdf#page=7
    scoring: https://cdasr.mclean.harvard.edu/wp-content/uploads/2017/09/Auerbach_2012_BRT.pdf#page=3
    NDA data dictionary: https://nda.nih.gov/data_structure.html?short_name=rbqa01
    NDA and original questionnaire values: 0 = Never; 1 = Almost Never; 2 = Sometimes; 3 = Almost Always; 4 = Always
    scores interpretation: higher score indicates higher level of behavior problem of subscale
    '''

    # questionnaire info
    name = 'rbqa'
    filename =f'{name}01'
    n_items = 20
    reverse_items = []
    fillers_items = []
    rules = {'0' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, 'NaN': np.nan}

    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 
    
    cols_items = [f'{name}{i}' for i in range(1,n_items+1)] # returns list of column names
    df[cols_items] = df[cols_items].replace(rules) # re-codes    
    
    # note: no need to reverse scoring since no items needs to be reversed as per the original questionnaire 
    
    # subscales info
    # subscalename = [['subscalename', ['items numbers'], ['placeholder for cols corresponding to items'], 'n items in subscale', 'min score' (added later), 'max score' (added later)]]
    tot = ['tot', list( range(1, n_items+1) ), [], n_items, [], []]

    subscales = [tot]

    # scoring and reality checks
    for scale in subscales:
        scale[4] = scale[3] * 0  # add min
        scale[5] = scale[3] * 4  # add max
        scale[2] = [f'{name}{item}' for item in scale[1]] # returns list of column names corresponding to only the items that make up the subscale
        df[f'{name}_{scale[0]}_nan_count'] = df[scale[2]].isna().sum(axis=1) # count and store number of item-level nan answers
        if grab_item_count == 'yes': 
            df[f'{name}_{scale[0]}_items_count'] = len(scale[2]) # store theoretical count of items making up the subscale
        df[f'{name}_{scale[0]}'] = df[scale[2]].sum(axis=1, min_count = 1) # subscale score

        # calculating min/max i) subscale value and ii) the min/max possible score they would have gotten if they had answered the quesionts they had as nans
        scale_min = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 0 )).min()
        scale_max = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 4 )).max()

        #scale[4] and scale[5] indicate the theoretical ranges; scale_min|_max are actual min and max of data     
        assert (scale_min >= scale[4]) and (scale_max <= scale[5]), f'Watch out: values for subscale {name}_{scale[0]} are outside of theoretical range for some participants. values should be between {scale[4]} and {scale[5]} but are between {scale_min} and {scale_max} instead'

        # check that overall number of items grabbed is correct
        n_items_used = [scale[3] for scale in subscales]
        assert sum(n_items_used) == n_items - len(fillers_items), 'The number of items used to calculate the subscales and total score is either more or less than what was expected'
    
    # remove item-level columns from original df
    if item_level == 'drop':
        df = df.drop(columns = cols_items)

    print(f'scored {name}')
    return df

## ================================================= ##
def score_rcads(df, item_level, grab_item_count):
    '''
    questionnaire: Revised Children's Anxiety and Depression Scale (RCADS)
    respondent: child
    6 subscales and 1 total score
    subscales: ocd, general anxiety, social anxiety, panic, separation anxiety, depression
    total: sum of all items
    filler items: none
    reverse scored items: none
    notes from BANDA1.0_Crosswalk.csv file: none
    questionnaire and scoring: https://www.corc.uk.net/outcome-experience-measures/revised-childrens-anxiety-and-depression-scale-rcads/
    extra scoring info : https://www.childfirst.ucla.edu/wp-content/uploads/sites/163/2018/03/RCADSUsersGuide20150701.pdf
    NDA data dictionary: https://nda.nih.gov/data_structure.html?short_name=rcads01
    NDA values: text only- Never; Sometimes; Often; Always 
    original questionnaire values: 0=Never; 1=Sometimes; 2=Often; 3=Always
    scores interpretation: higher score indicates higher level of behavior problem of subscale
    '''

    # questionnaire info
    name = 'rcads'
    filename =f'{name}01'
    n_items = 47
    reverse_items = []
    fillers_items = []
    rules = {'Never' : 0, 'Sometimes' : 1, 'Often' : 2, 'Always' : 3, 'NaN': np.nan}

    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 
    
    cols_items = [f'{name}_{i}' for i in range(1,n_items+1)]  # returns list of column names
    df[cols_items] = df[cols_items].replace(rules) # re-codes 

    # note: no need to reverse scoring since no items needs to be reverted as per the original questionnaire 
        
    # subscales info
    # subscalename = [['subscalename', ['items numbers'], ['placeholder for cols corresponding to items'], 'n items in subscale', 'min score' (added later), 'max score' (added later)]]
    subscale_1 = ['social', [4, 7, 8, 12, 20, 30, 32, 38, 43], [], 9, [], []] 
    subscale_2 = ['panic', [3, 14, 24, 26, 28, 34, 36, 39, 41], [], 9, [], []]
    subscale_3 = ['mdd', [2, 6, 11, 15, 19, 21, 25, 29, 40, 47], [], 10, [], []]
    subscale_4 = ['sad', [5, 9, 17, 18, 33, 45, 46], [], 7, [], []]
    subscale_5 = ['gad', [1, 13, 22, 27, 35, 37], [], 6, [], []]
    subscale_6 = ['ocd', [10, 16, 23, 31, 42, 44], [], 6, [], []]
    tot = ['tot', list( range(1, n_items+1) ), [], n_items, [], []]
   
    subscales = [subscale_1, subscale_2, subscale_3, subscale_4, subscale_5, subscale_6, tot]
    
    # scoring and reality checks
    for scale in subscales:
        scale[4] = scale[3] * 0  # add min
        scale[5] = scale[3] * 3  # add max
        scale[2] = [f'{name}_{item}' for item in scale[1]] # returns list of column names corresponding to only the items that make up the subscale
        df[f'{name}_{scale[0]}_nan_count'] = df[scale[2]].isna().sum(axis=1) # count and store number of item-level nan answers
        if grab_item_count == 'yes': 
            df[f'{name}_{scale[0]}_items_count'] = len(scale[2]) # store theoretical count of items making up the subscale
        df[f'{name}_{scale[0]}'] = df[scale[2]].sum(axis=1, min_count = 1) # score subscale
    
        # calculating min/max i) subscale value and ii) the min/max possible score they would have gotten if they had answered the quesionts they had as nans
        scale_min = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 0 )).min()
        scale_max = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 3 )).max()

        #scale[4] and scale[5] indicate the theoretical ranges; scale_min|_max are actual min and max of data     
        assert (scale_min >= scale[4]) and (scale_max <= scale[5]), f'Watch out: values for subscale {name}_{scale[0]} are outside of theoretical range for some participants. values should be between {scale[4]} and {scale[5]} but are between {scale_min} and {scale_max} instead'

        # check that overall number of items grabbed is correct
        n_items_used = [scale[3] for scale in subscales]
        assert sum(n_items_used) == 2*n_items - len(fillers_items), 'The number of items used to calculate the subscales and total score is either more or less than what was expected' #note: doing 2* because subscales includes the total subscale too, which here corresponds to the sum of all items

    # remove item-level columns from original df
    if item_level == 'drop':
        df = df.drop(columns = cols_items)
    
    print(f'scored {name}')
    return df

# ================================================= #
def score_rmbi(df, item_level, grab_item_count):
    '''
    questionnaire: Retrospective Measure of Behavioral Inhibition (RMBI)
    respondent: parent (about their child, not self-report)
    4 subscales and 1 total score
    subscales: fearful inhibition, non-approach, risk avoidance, shyness&sensitivity
    total: sum of all items
    fillers items: none
    reverse scored items: 4, 5, 7, 11, 13, 15
    notes from BANDA1.0_Crosswalk.csv file: items are already reverse scored
    questionnaire: similar version: https://scholarship.shu.edu/cgi/viewcontent.cgi?referer=&httpsredir=1&article=1239&context=theses#page=47
    scoring: https://www.sciencedirect.com/science/article/pii/S0165178105001022
    NDA data dictionary: https://nda.nih.gov/data_structure.html?short_name=rmbi01
    NDA data scoring: 0 = no/hardly ever; 1 = some of the time; 2 = yes/most of the time; 3 = do not remember at all; #this vary based on specific items (see items_a and items_b)
    original questionnaire scoring: 0 = “no/hardly ever”; 1 = “some of the time”, or 2 = “yes/most of the time”
    scores interpretation: higher score indicates higher level of behavior problem of subscale
    '''

    # questionnaire info
    name = 'rmbi'
    filename =f'{name}01'
    n_items = 18
    reverse_items = []
    fillers_items = []
    
    items_a = [1, 2, 3, 6, 8, 9, 10, 12, 14, 16, 17, 18]
    rules_a = {0 : 0, 1 : 1, 2 : 2, 3 : np.nan, 'NaN': np.nan}

    items_b = [4, 5, 7, 11, 13, 15]
    rules_b = {0 : np.nan, 1 : 0, 2 : 1, 3 : 2, 'NaN': np.nan}

    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 

    cols_items_a = [f'{name}{i}' for i in range(1,len(items_a)+1)]  # returns list of column names
    df[cols_items_a] = df[cols_items_a].replace(rules_a) # re-codes 
    
    cols_items_b = [f'{name}{i}' for i in range(1,len(items_b)+1)]  # returns list of column names
    df[cols_items_b] = df[cols_items_b].replace(rules_b) # re-codes 

    # subscales info
    # subscalename = [['subscalename', ['items numbers'], ['placeholder for cols corresponding to items'], 'n items in subscale', 'min score' (added later), 'max score' (added later)]]
    # subscale_1 = ['fear_inhib', [1, 6, 10, 16, 18], [], 5, [], []] 
    # subscale_2 = ['non_approach', [2, 4, 5, 9, 11, 15], [], 6, [], []]
    # subscale_3 = ['risk_avoid', [7, 8, 13], [], 3, [], []] 
    # subscale_4 = ['shy_sensit', [3, 12, 14, 17], [], 4, [], []] 
    tot = ['tot', list( range(1, n_items+1) ), [], n_items, [], []]
   
    # subscales = [subscale_1, subscale_2, subscale_3, subscale_4, tot]
    subscales = [tot]
    
    # scoring and reality checks
    for scale in subscales:
        scale[4] = scale[3] * 0  # add min
        scale[5] = scale[3] * 2  # add max
        scale[2] = [f'{name}{item}' for item in scale[1]] # returns list of column names corresponding to only the items that make up the subscale
        df[f'{name}_{scale[0]}_nan_count'] = df[scale[2]].isna().sum(axis=1) # count and store number of item-level nan answers
        if grab_item_count == 'yes': 
            df[f'{name}_{scale[0]}_items_count'] = len(scale[2]) # store theoretical count of items making up the subscale
        df[f'{name}_{scale[0]}'] = df[scale[2]].sum(axis=1, min_count = 1) # score subscale
    
        # calculating min/max i) subscale value and ii) the min/max possible score they would have gotten if they had answered the quesionts they had as nans
        scale_min = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 0 )).min()
        scale_max = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 2 )).max()

        #scale[4] and scale[5] indicate the theoretical ranges; scale_min|_max are actual min and max of data     
        assert (scale_min >= scale[4]) and (scale_max <= scale[5]), f'Watch out: values for subscale {name}_{scale[0]} are outside of theoretical range for some participants. values should be between {scale[4]} and {scale[5]} but are between {scale_min} and {scale_max} instead'

        # check that overall number of items grabbed is correct
        n_items_used = [scale[3] for scale in subscales]
        assert sum(n_items_used) == n_items - len(fillers_items), 'The number of items used to calculate the subscales and total score is either more or less than what was expected' #note: doing 2* because subscales includes the total subscale too, which here corresponds to the sum of all items

    # remove item-level columns from original df
    if item_level == 'drop':
        df = df.drop(columns = cols_items_a + cols_items_b)
    
    print(f'scored {name}')
    return df

## ================================================= ##
def score_shaps(df, item_level, grab_item_count):
    '''
    questionnaire: Snaith-Hamilton Pleasure Scale (SHAPS)
    respondent: child
    0 subscales and 1 total score
    subscales: none
    total: sum of all items
    fillers items: none
    reverse scored items: 2, 4, 5, 7, 9, 12, 14
    notes from BANDA1.0_Crosswalk.csv file: items [2, 4, 5, 7, 9, 12, 14] are already reversed scored, and all items plus 1
    questionnaire: https://www.phenxtoolkit.org/toolkit_content/PDF/PX710601.pdf
    scoring info: https://datashare.nida.nih.gov/instrument/snaith-hamilton-pleasure-scale
    NDA data dictionary: https://nda.nih.gov/data_structure.html?short_name=shaps01
    NDA values: 1= Strongly disagree; 2= Disagree; 3= Agree; 4= Strongly agree
    original questionnaire values: 0= Strongly disagree; 1= Disagree; 2= Agree; 3= Strongly agree
    scoring: 1= Strongly disagree; 1= Disagree; 0= Agree; 0= Strongly agree
    scores interpretation: higher score means higher level of anhedonia. "A score of 2 or less constitutes a “normal” score, while an “abnormal” score is defined as 3 or more" from scoring info (URL above)
    '''

    # questionnaire info
    name =  'shaps'
    filename =f'{name}01'
    n_items = 14
    reverse_items = []
    fillers_items = []
    rules = {1 : 1, 2 : 1, 3 : 0, 4 : 0, 'NaN': np.nan, 999: np.nan, -9: np.nan}

    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 
    
    cols_items = [f'{name}{i}' for i in range(1,n_items+1)]  # returns list of column names
    df[cols_items] = df[cols_items].replace(rules) # re-codes 

    # note: no need to reverse scoring since NDA data had already been reversed
    
    # subscales info
    # subscalename = [['subscalename', ['items numbers'], ['placeholder for cols corresponding to items'], 'n items in subscale', 'min score' (added later), 'max score' (added later)]]
    tot = ['tot', list( range(1, n_items+1) ), [], n_items, [], []]
   
    subscales = [tot]
            
    # scoring and reality checks
    for scale in subscales:
        scale[4] = scale[3] * 0  # add min
        scale[5] = scale[3] * 1  # add max
        scale[2] = [f'{name}{item}' for item in scale[1]] # returns list of column names corresponding to only the items that make up the subscale
        df[f'{name}_{scale[0]}_nan_count'] = df[scale[2]].isna().sum(axis=1) # count and store number of item-level nan answers
        if grab_item_count == 'yes': 
            df[f'{name}_{scale[0]}_items_count'] = len(scale[2]) # store theoretical count of items making up the subscale
        df[f'{name}_{scale[0]}'] = df[scale[2]].sum(axis=1, min_count = 1) # score subscale
    
        # calculating min/max i) subscale value and ii) the min/max possible score they would have gotten if they had answered the quesionts they had as nans
        scale_min = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 0 )).min()
        scale_max = (df[f'{name}_{scale[0]}'] + ( df[f'{name}_{scale[0]}_nan_count'] * 1 )).max()

        #scale[4] and scale[5] indicate the theoretical ranges; scale_min|_max are actual min and max of data     
        assert (scale_min >= scale[4]) and (scale_max <= scale[5]), f'Watch out: values for subscale {name}_{scale[0]} are outside of theoretical range for some participants. values should be between {scale[4]} and {scale[5]} but are between {scale_min} and {scale_max} instead'

        # check that overall number of items grabbed is correct
        n_items_used = [scale[3] for scale in subscales]
        assert sum(n_items_used) == n_items - len(fillers_items), 'The number of items used to calculate the subscales and total score is either more or less than what was expected'
    
    # remove item-level columns from original df
    if item_level == 'drop':
        df = df.drop(columns = cols_items)

    print(f'scored {name}')
    return df

## ================================================= ##
# def score_stai(df, item_level, grab_item_count):
# work in progress

## ================================================= ##
# score_strain(df, item_level, grab_item_count)
# scoring info is proprietary so not including here

## ================================================= ##
def score_tanner(df, item_level, grab_item_count):
    '''
    questionnaire: Tanner Staging Sexual Maturity Scale
    respondent: child
    1 total score
    total: sum of all items
    fillers items: none
    reverse scored items: none
    notes from BANDA1.0_Crosswalk.csv file: none
    questionnaire: for full pdf go to /work/swglab/data/BANDA/NDA/BANDAImgManifestBeh/info_questionnaires_manual_crosswalk/TANNER.pdf
    A similar version can be found online here 1-s2.0-S0022347617304626-ympd9106-fig-0001.jpg but does not contain 3 subjective questions. 
    Info on Tanner stages here: https://www.ncbi.nlm.nih.gov/books/NBK544322/figure/article-20323.image.f1/?report=objectonly
    NDA data dictionary: https://nda.nih.gov/data_structure.html?short_name=tanner_sms01
    NDA & original questionnaire values: dependent on question, see NDA link above. NDA and questionnaire values match. 
    scores interpretation: higher values means higher pubertal stage (i.e more development/more adult like)
    '''

    # questionnaire info
    name = 'tanner'
    filename =f'{name}_sms01'
    n_items = 5
    reverse_items = []
    fillers_items = []
    rules = {1.0 : 1, 2.0 : 2, 3.0 : 3, 4.0 : 4, 5.0 : 5, 'NaN': np.nan, 9999.0: np.nan}

    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 
    
    cols_items = ['tsf1', 'tsf2', 'tsf3', 'tsftsg', 'tsftphg']
    df[cols_items] = df[cols_items].replace(rules) # re-codes 

    # note: no need to reverse scoring since no items needs to be reversed as per the original questionnaire 
    
    # subscales info
    # subscalename = [['subscalename', ['items numbers'], ['placeholder for cols corresponding to items'], 'n items in subscale', 'min score' (added later), 'max score' (added later)]]
    tot = ['tot', cols_items, [], n_items, [], []]
   
    subscales = [tot] #keep for consistency
    
    # scoring and reality checks
    for scale in subscales:
        scale[4] = 1  # add min
        scale[5] = 5  # add max
        scale[2] = scale[1]  # keep for consistency with other functions, but here no need since column names were hardcoded
        df[f'{name}_{scale[0]}_nan_count'] = df[scale[2]].isna().sum(axis=1) # count and store number of item-level nan answers
        if grab_item_count == 'yes': 
            df[f'{name}_{scale[0]}_items_count'] = len(scale[2]) # store theoretical count of items making up the subscale
        df[f'{name}_{scale[0]}'] = df[scale[2]].mean(axis=1, skipna = True ) # score subscale
    
        # calculating min/max i) subscale value and ii) the min/max possible score they would have gotten if they had answered the quesionts they had as nans
        scale_min = df[f'{name}_{scale[0]}'].min()
        scale_max = df[f'{name}_{scale[0]}'].max()

        #scale[4] and scale[5] indicate the theoretical ranges; scale_min|_max are actual min and max of data     
        assert (scale_min >= scale[4]) and (scale_max <= scale[5]), f'Watch out: values for subscale {name}_{scale[0]} are outside of theoretical range for some participants. values should be between {scale[4]} and {scale[5]} but are between {scale_min} and {scale_max} instead'

        # check that overall number of items grabbed is correct
        n_items_used = [scale[3] for scale in subscales]
        assert sum(n_items_used) == n_items - len(fillers_items), 'The number of items used to calculate the subscales and total score is either more or less than what was expected' #note: doing 2* because subscales includes the total subscale too, which here corresponds to the sum of all items    

    # remove item-level columns from original df
    if item_level == 'drop':
        df = df.drop(columns = cols_items)

    print(f'scored {name}')
    return df

## ================================================= ##
def score_wasi(df, item_level, grab_item_count):
    '''
    currently just relabelling columns since NDA released aggregated data and no item-level info   
    '''

    # questionnaire info
    name = 'wasi'
    filename =f'{name}201'
    rules = {'NaN': np.nan, 999.0: np.nan}

    cols_items = df.columns[df.columns.str.contains(f'{filename}')]  # returns list of column names
    df[cols_items] = df[cols_items].replace(rules) # re-codes
    
    # undoing the customizing of column names (for consistency with NDA)
    df.columns = df.columns.str.replace(f'{filename}_','') #had previously appended filename_ to each variable coming from a file, here removing 
 

    # note: NDA had no item_level for the wasi, so not really dropping any cols here
    # note: the option grab_item_count has no meaning here since no item_level info was available under NDA
    
    print(f'scored {name}')
    return df

## ================================================= ##
def score_penn_nih_tasks(df, item_level, grab_item_count):
    '''
    currently just relabelling columns since NDA released aggregated data and no item-level info   
    '''
    new_names ={'pwmt01' : 'penntask_wordmem',
                 'pmat01' : 'penntask_matreason',
                 'er4001' : 'penntask_emorecog',
                 'deldisk01' : 'penntask_delaydisc',
                 "dccs01_nih_dccs" : "nihtoolbox_dimenscardsort",
                 'dccs01' : 'nihtoolbox_dimenscardsort',
                 'lswmt01' : 'nihtoolbox_listsort',
                 'flanker01' : 'nihtoolbox_flanker',
                 'orrt01' : 'nihtoolbox_oralreadrec',
                 'pcps01' : 'nihtoolbox_patterncompar',
                }

    #rename columns
    for k in new_names.keys():
        df.columns = df.columns.str.replace(k,new_names[k])
    
    # note: NDA had no item_level for the strain, so not really dropping any cols here
    # note: the option grab_item_count has no meaning here since no item_level info was available under NDA
    

    rules = {'NaN': np.nan, 999.0: np.nan, '999.0': np.nan}
    cols_items = df.max()[df.max().values == 999.0].keys().to_list()  # grabbing columns if their maximum values is 999.0 should have probably been a NaN eg delaydisc_sv_*
    cols_items = [i for i in cols_items if 'nihtoolbox' in i or 'penntask' in i] #making sure to do it only for penn_ or nihtlbox columns
    df[cols_items] = df[cols_items].replace(rules) # re-codes 
    
    print('Remaned PENN tasks and NIH toolbox tasks')
    return df
